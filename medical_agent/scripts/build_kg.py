"""
scripts/build_kg.py — 从 Huatuo-26M-Lite 构建医学 KG 并导入 Neo4j

使用流程:
    # 1. 准备数据
    # 下载 Huatuo-26M-Lite 数据集
    
    # 2. 启动 Neo4j
    docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/yourpassword neo4j:5.x
    
    # 3. 跑构建
    python scripts/build_kg.py \
        --input-path ./data/huatuo_26m_lite.json \
        --neo4j-uri bolt://localhost:7687 \
        --neo4j-user neo4j \
        --neo4j-password yourpassword

预期产出:
    - 4.5 万实体（疾病、症状、药物、治疗方法）
    - 31 万关系（典型症状、并发症、推荐药物等）

实现状态: 🟡 骨架 / ⚪ 需根据 Huatuo 实际 schema 完善
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


# ============================================================
# 实体类型定义
# ============================================================

ENTITY_TYPES = {
    "Disease": ["疾病", "病症"],
    "Symptom": ["症状"],
    "Drug": ["药物", "药品"],
    "Treatment": ["治疗", "疗法", "手术"],
    "Examination": ["检查", "检验"],
    "Department": ["科室"],
    "BodyPart": ["部位"],
}

RELATION_TYPES = [
    "典型症状",      # Disease -> Symptom
    "可能病因",      # Symptom -> Disease
    "推荐药物",      # Disease -> Drug
    "禁用药物",      # Disease -> Drug
    "并发症",        # Disease -> Disease
    "推荐治疗",      # Disease -> Treatment
    "推荐检查",      # Disease -> Examination
    "所属科室",      # Disease -> Department
    "影响部位",      # Disease -> BodyPart
    "缓解药物",      # Symptom -> Drug
    "适应症",        # Drug -> Disease
    "副作用",        # Drug -> Symptom
    "禁忌症",        # Drug -> Disease
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", required=True, help="Huatuo-26M-Lite 数据文件")
    p.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-password", required=True)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true", help="不写入 Neo4j，仅打印统计")
    return p.parse_args()


# ============================================================
# 实体抽取
# ============================================================

def extract_entities_from_qa(qa_pair: Dict) -> List[Dict]:
    """
    从一条 (question, answer) 中抽取实体
    
    生产环境用 BERT-BiLSTM-CRF NER 模型；这里给个基于词表的简单版本作示意。
    """
    question = qa_pair.get("question", "")
    answer = qa_pair.get("answer", "")
    text = question + " " + answer
    
    # TODO: 实际实现需要：
    # 1. 加载 NER 模型 (BERT-BiLSTM-CRF) 
    # 2. 对 text 做实体抽取
    # 3. 对每个实体做实体链接（链到医学知识库标准词）
    # 4. 返回 [{"name": "脑膜炎", "type": "Disease"}, ...]
    
    raise NotImplementedError(
        "请在 GPU 服务器实现：\n"
        "1. 加载预训练 BERT-BiLSTM-CRF NER 模型\n"
        "2. 抽取后做实体链接（链接到 CN-DBpedia / ICD-10 标准词）"
    )


def extract_relations_from_qa(qa_pair: Dict, entities: List[Dict]) -> List[Dict]:
    """
    从一条 QA 中抽取关系
    
    生产环境：
    - 方案 A: 关系抽取模型（如 CasRel、TPLinker）
    - 方案 B: 让 LLM 按 schema 输出三元组
    
    返回: [{"src": "脑膜炎", "rel": "典型症状", "dst": "头痛"}, ...]
    """
    raise NotImplementedError(
        "请在 GPU 服务器实现：\n"
        "1. 加载关系抽取模型 或 调用 LLM 做 schema-guided 抽取\n"
        "2. 对抽出的三元组做去重和置信度过滤"
    )


# ============================================================
# Neo4j 写入
# ============================================================

def create_indexes(driver):
    """创建 Neo4j 索引（加速查询）"""
    queries = [
        "CREATE INDEX disease_name IF NOT EXISTS FOR (n:Disease) ON (n.name)",
        "CREATE INDEX symptom_name IF NOT EXISTS FOR (n:Symptom) ON (n.name)",
        "CREATE INDEX drug_name IF NOT EXISTS FOR (n:Drug) ON (n.name)",
    ]
    with driver.session() as session:
        for q in queries:
            session.run(q)


def batch_insert_entities(driver, entities: List[Dict]):
    """批量插入实体"""
    cypher = """
    UNWIND $entities AS e
    MERGE (n {name: e.name})
    SET n:`Entity`, n.type = e.type
    """
    with driver.session() as session:
        session.run(cypher, entities=entities)


def batch_insert_relations(driver, relations: List[Dict]):
    """批量插入关系"""
    # 注意：Cypher 不支持动态关系类型，需要按 rel_type 分组
    by_type = {}
    for r in relations:
        by_type.setdefault(r["rel"], []).append(r)
    
    for rel_type, rels in by_type.items():
        cypher = f"""
        UNWIND $rels AS r
        MATCH (a {{name: r.src}}), (b {{name: r.dst}})
        MERGE (a)-[rel:`{rel_type}`]->(b)
        SET rel.weight = COALESCE(r.weight, 1.0),
            rel.confidence = COALESCE(r.confidence, 1.0)
        """
        with driver.session() as session:
            session.run(cypher, rels=rels)


# ============================================================
# 主流程
# ============================================================

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    # 加载数据
    logger.info(f"Loading Huatuo data from {args.input_path}")
    with open(args.input_path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} QA pairs")
    
    # 抽取实体和关系
    all_entities = {}  # name -> entity dict
    all_relations = []
    
    for i, qa in enumerate(data):
        if i % 1000 == 0:
            logger.info(f"Processing {i}/{len(data)}, entities so far: {len(all_entities)}")
        
        try:
            entities = extract_entities_from_qa(qa)
            for e in entities:
                if e["name"] not in all_entities:
                    all_entities[e["name"]] = e
            
            relations = extract_relations_from_qa(qa, entities)
            all_relations.extend(relations)
        except Exception as ex:
            logger.warning(f"Failed on QA {i}: {ex}")
            continue
    
    # 去重关系
    rel_set = set()
    deduped_relations = []
    for r in all_relations:
        key = (r["src"], r["rel"], r["dst"])
        if key not in rel_set:
            rel_set.add(key)
            deduped_relations.append(r)
    
    logger.info(f"Total entities: {len(all_entities)}")
    logger.info(f"Total relations: {len(deduped_relations)}")
    
    if args.dry_run:
        logger.info("Dry run, not writing to Neo4j")
        return
    
    # 写入 Neo4j
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    
    logger.info("Creating indexes...")
    create_indexes(driver)
    
    logger.info("Inserting entities...")
    entities_list = list(all_entities.values())
    for i in range(0, len(entities_list), args.batch_size):
        batch = entities_list[i:i + args.batch_size]
        batch_insert_entities(driver, batch)
        logger.info(f"  inserted {i + len(batch)}/{len(entities_list)}")
    
    logger.info("Inserting relations...")
    for i in range(0, len(deduped_relations), args.batch_size):
        batch = deduped_relations[i:i + args.batch_size]
        batch_insert_relations(driver, batch)
        logger.info(f"  inserted {i + len(batch)}/{len(deduped_relations)}")
    
    driver.close()
    logger.info("✅ KG construction complete!")


if __name__ == "__main__":
    main()
