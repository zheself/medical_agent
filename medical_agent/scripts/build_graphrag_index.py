"""
scripts/build_graphrag_index.py — 构建 GraphRAG 离线索引

流程:
    1. 从 Neo4j 拉取全图
    2. 用 Leiden 做三层社区检测
    3. 对每个社区生成结构化摘要 (LLM 调用)
    4. 向量化摘要 (bge-m3)
    5. 写回 Neo4j (节点 community 属性) + 向量库 (摘要向量)

预期耗时: ~5 小时 (4 小时社区检测 + 1 小时摘要生成)

使用:
    python scripts/build_graphrag_index.py \
        --neo4j-uri bolt://localhost:7687 \
        --neo4j-password yourpassword \
        --llm-endpoint http://localhost:8000/v1 \
        --output-dir ./graphrag_index

实现状态: 🟡 骨架完整 / ⚪ 需在真实环境跑通
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-password", required=True)
    p.add_argument("--llm-endpoint", required=True, help="vLLM 服务地址")
    p.add_argument("--llm-model", default="qwen2.5-7b-medical-lora")
    p.add_argument("--embedder-model", default="BAAI/bge-m3")
    p.add_argument("--output-dir", default="./graphrag_index")
    p.add_argument("--resolutions", type=str, default="0.5,1.0,1.5",
                   help="Leiden resolution 参数 (对应粗/中/细)")
    return p.parse_args()


# ============================================================
# Step 1: 拉取 Neo4j 全图
# ============================================================

def fetch_graph_from_neo4j(uri: str, user: str, password: str) -> Tuple[List, List]:
    """返回 (nodes, edges)"""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    nodes = []
    edges = []
    
    with driver.session() as session:
        # 拉节点
        result = session.run("MATCH (n) RETURN n.name AS name, n.type AS type, id(n) AS id")
        for r in result:
            nodes.append({"id": r["id"], "name": r["name"], "type": r["type"]})
        
        # 拉边
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN id(a) AS src, id(b) AS dst, type(r) AS rel,
                   COALESCE(r.weight, 1.0) AS weight
        """)
        for r in result:
            edges.append({
                "src": r["src"], "dst": r["dst"],
                "rel": r["rel"], "weight": r["weight"]
            })
    
    driver.close()
    return nodes, edges


# ============================================================
# Step 2: Leiden 三层社区检测
# ============================================================

def detect_communities(nodes: List, edges: List, resolutions: List[float]) -> Dict[int, Dict]:
    """
    跑 Leiden，返回 {node_id: {"l0": int, "l1": int, "l2": int}}
    """
    import igraph as ig
    import leidenalg
    
    logger.info(f"Building igraph: {len(nodes)} nodes, {len(edges)} edges")
    
    # 构建 igraph
    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    
    # 建立 neo4j_id -> igraph_idx 映射
    id_map = {n["id"]: i for i, n in enumerate(nodes)}
    
    edge_list = [(id_map[e["src"]], id_map[e["dst"]]) for e in edges
                 if e["src"] in id_map and e["dst"] in id_map]
    weights = [e["weight"] for e in edges
               if e["src"] in id_map and e["dst"] in id_map]
    g.add_edges(edge_list)
    g.es["weight"] = weights
    
    # 三层社区检测
    membership_by_level = {}
    for level, res in enumerate(resolutions):
        logger.info(f"Running Leiden at level {level} (resolution={res})...")
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=res,
            weights="weight"
        )
        membership_by_level[f"l{level}"] = partition.membership
        logger.info(f"  Level {level}: {len(set(partition.membership))} communities")
    
    # 整理回 node_id -> levels
    result = {}
    for i, node in enumerate(nodes):
        result[node["id"]] = {
            level: membership_by_level[level][i]
            for level in membership_by_level
        }
    return result


# ============================================================
# Step 3: 社区摘要生成
# ============================================================

COMMUNITY_SUMMARY_PROMPT = """你是医学知识图谱摘要专家。请基于下方社区内的实体和关系，生成结构化摘要。

# 社区核心实体（按 PageRank 重要性排序）
{core_entities}

# 社区内关键关系（top 20）
{key_relations}

# 严格要求
1. 只使用上述提供的实体和关系，**不要引入任何外部知识**
2. 输出必须是合法 JSON，字段如下：
{{
  "theme": "社区主题，简短 (如 '中枢神经系统感染')",
  "core_entities": [复述核心实体列表],
  "key_relations": [复述关键关系列表],
  "differential_diagnosis": ["疾病A vs 疾病B 的鉴别要点", ...] (如适用),
  "treatment_principles": "总体治疗原则" (如适用),
  "narrative": "自然语言描述，3-5 句，将用于向量检索"
}}

# 输出 (仅 JSON):
"""


def generate_community_summary(
    community_entities: List[Dict],
    community_relations: List[Dict],
    llm_client,
    model: str,
) -> Dict:
    """为一个社区生成摘要"""
    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        core_entities=", ".join(e["name"] for e in community_entities[:30]),
        key_relations="\n".join(
            f"  {r['src']} -[{r['rel']}]-> {r['dst']}"
            for r in community_relations[:20]
        ),
    )
    
    response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.1,
    )
    response_text = response.choices[0].message.content
    
    # 解析 JSON（用 outlines 更稳，这里简化）
    import re
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Failed to parse JSON from: {response_text[:200]}")


def verify_summary(summary: Dict, original_entities: set) -> Dict:
    """
    回填验证：摘要中提到的实体必须在原社区中
    
    返回:
        {"is_hallucinated": bool, "hallucination_ratio": float, "details": ...}
    """
    # 提取 narrative 中的实体（生产环境用 NER；这里用简单字符串匹配）
    narrative = summary.get("narrative", "")
    
    # 这里需要 NER 来准确提取实体，简化版用枚举
    hallucinated = []
    for ent in summary.get("core_entities", []):
        if ent not in original_entities:
            hallucinated.append(ent)
    
    ratio = len(hallucinated) / max(len(summary.get("core_entities", [])), 1)
    return {
        "is_hallucinated": ratio > 0.1,
        "hallucination_ratio": ratio,
        "hallucinated_entities": hallucinated,
    }


# ============================================================
# Step 4: 向量化
# ============================================================

def embed_summaries(summaries: Dict[str, Dict], embedder_model: str) -> Dict[str, List[float]]:
    """对所有摘要的 narrative 字段做向量化"""
    from sentence_transformers import SentenceTransformer
    
    logger.info(f"Loading embedder: {embedder_model}")
    model = SentenceTransformer(embedder_model)
    
    embeddings = {}
    texts = []
    keys = []
    for cid, summary in summaries.items():
        keys.append(cid)
        texts.append(summary.get("narrative", ""))
    
    logger.info(f"Embedding {len(texts)} narratives...")
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True)
    
    for k, v in zip(keys, vectors):
        embeddings[k] = v.tolist()
    return embeddings


# ============================================================
# Step 5: 写回 Neo4j + Milvus
# ============================================================

def write_back_communities_to_neo4j(driver, memberships: Dict):
    """把 community membership 写回 Neo4j 节点属性"""
    cypher = """
    UNWIND $batch AS row
    MATCH (n) WHERE id(n) = row.id
    SET n.community_l0 = row.l0,
        n.community_l1 = row.l1,
        n.community_l2 = row.l2
    """
    batch = [
        {"id": nid, "l0": levels.get("l0"), "l1": levels.get("l1"), "l2": levels.get("l2")}
        for nid, levels in memberships.items()
    ]
    
    BATCH_SIZE = 1000
    with driver.session() as session:
        for i in range(0, len(batch), BATCH_SIZE):
            chunk = batch[i:i + BATCH_SIZE]
            session.run(cypher, batch=chunk)


def write_to_milvus(summaries: Dict, embeddings: Dict, collection_name: str):
    """写入 Milvus / Zilliz"""
    # 生产实现：
    # from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
    # connections.connect(...)
    # collection = Collection(collection_name)
    # collection.insert([
    #     [cid for cid in summaries],
    #     [embeddings[cid] for cid in summaries],
    #     [json.dumps(summaries[cid]) for cid in summaries],
    # ])
    raise NotImplementedError("生产环境实现 Milvus 写入")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: 拉图
    logger.info("=" * 60)
    logger.info("Step 1: Fetching graph from Neo4j")
    nodes, edges = fetch_graph_from_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    logger.info(f"Fetched {len(nodes)} nodes, {len(edges)} edges")
    
    # Step 2: Leiden
    logger.info("=" * 60)
    logger.info("Step 2: Leiden community detection")
    resolutions = [float(r) for r in args.resolutions.split(",")]
    memberships = detect_communities(nodes, edges, resolutions)
    
    # 保存中间结果
    with open(output_dir / "memberships.json", "w") as f:
        json.dump({str(k): v for k, v in memberships.items()}, f)
    
    # Step 3: 按社区分组生成摘要
    logger.info("=" * 60)
    logger.info("Step 3: Generating community summaries")
    
    from openai import OpenAI
    llm_client = OpenAI(base_url=args.llm_endpoint, api_key="EMPTY")
    
    # 按 level 1 (中粒度) 分组
    by_l1 = {}
    for node_id, levels in memberships.items():
        l1 = levels["l1"]
        by_l1.setdefault(l1, []).append(node_id)
    
    summaries = {}
    for community_id, node_ids in by_l1.items():
        community_entities = [n for n in nodes if n["id"] in node_ids]
        community_relations = [
            e for e in edges
            if e["src"] in node_ids and e["dst"] in node_ids
        ]
        
        # 尝试 3 次（含幻觉检测和重试）
        original_entity_set = {n["name"] for n in community_entities}
        for attempt in range(3):
            try:
                summary = generate_community_summary(
                    community_entities, community_relations,
                    llm_client, args.llm_model
                )
                verify = verify_summary(summary, original_entity_set)
                if not verify["is_hallucinated"]:
                    summaries[f"L1_{community_id}"] = summary
                    break
                else:
                    logger.warning(
                        f"Community {community_id} attempt {attempt+1}: "
                        f"hallucination_ratio={verify['hallucination_ratio']:.2f}"
                    )
            except Exception as e:
                logger.warning(f"Community {community_id} attempt {attempt+1}: {e}")
        else:
            logger.error(f"Community {community_id} failed after 3 attempts, marking for review")
    
    # 保存
    with open(output_dir / "summaries.json", "w") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    
    # Step 4: 向量化
    logger.info("=" * 60)
    logger.info("Step 4: Embedding summaries")
    embeddings = embed_summaries(summaries, args.embedder_model)
    
    with open(output_dir / "embeddings.json", "w") as f:
        json.dump(embeddings, f)
    
    # Step 5: 写回
    logger.info("=" * 60)
    logger.info("Step 5: Writing back to Neo4j + Milvus")
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    write_back_communities_to_neo4j(driver, memberships)
    driver.close()
    
    # write_to_milvus(summaries, embeddings, "medical_communities")  # 真实环境启用
    
    logger.info("✅ GraphRAG index build complete!")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Summaries: {len(summaries)}")


if __name__ == "__main__":
    main()
