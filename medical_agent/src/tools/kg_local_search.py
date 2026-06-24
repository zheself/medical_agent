"""
tools/kg_local_search.py — GraphRAG Local Search 工具

实体级检索，适合具体事实查询。
工作流程：
1. 接收实体（来自 NER 或直接输入）
2. 在 Neo4j 上做 1-2 跳邻居展开
3. 用 PageRank 重排，保留重要节点
4. 返回结构化的事实集合

实现状态: ✅ 接口完整 / 🟡 Neo4j 调用部分提供 Mock
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseTool


# ============================================================
# Neo4j 接口抽象
# ============================================================

class KGBackend:
    """KG 后端抽象，便于切换 Neo4j / Mock"""
    
    async def query_neighbors(self, entity: str, depth: int = 1, limit: int = 20) -> List[Dict]:
        raise NotImplementedError
    
    async def get_pagerank(self, node_id: str) -> float:
        raise NotImplementedError


class MockKGBackend(KGBackend):
    """本地测试用的 Mock KG"""
    
    # 一些预置的医学事实（足以让 demo 跑通）
    MOCK_FACTS = {
        "头痛": [
            {"rel": "可能病因", "target": "偏头痛", "target_type": "disease", "weight": 0.6},
            {"rel": "可能病因", "target": "脑膜炎", "target_type": "disease", "weight": 0.3},
            {"rel": "可能病因", "target": "蛛网膜下腔出血", "target_type": "disease", "weight": 0.1},
            {"rel": "缓解药物", "target": "布洛芬", "target_type": "drug", "weight": 0.7},
        ],
        "发烧": [
            {"rel": "可能病因", "target": "流感", "target_type": "disease", "weight": 0.5},
            {"rel": "可能病因", "target": "肺炎", "target_type": "disease", "weight": 0.3},
            {"rel": "可能病因", "target": "脑膜炎", "target_type": "disease", "weight": 0.2},
            {"rel": "缓解药物", "target": "对乙酰氨基酚", "target_type": "drug", "weight": 0.9},
        ],
        "颈强直": [
            {"rel": "可能病因", "target": "脑膜炎", "target_type": "disease", "weight": 0.85},
            {"rel": "可能病因", "target": "蛛网膜下腔出血", "target_type": "disease", "weight": 0.10},
        ],
        "糖尿病": [
            {"rel": "并发症", "target": "视网膜病变", "target_type": "disease", "weight": 0.7},
            {"rel": "并发症", "target": "糖尿病肾病", "target_type": "disease", "weight": 0.6},
            {"rel": "推荐药物", "target": "二甲双胍", "target_type": "drug", "weight": 0.9},
            {"rel": "推荐药物", "target": "胰岛素", "target_type": "drug", "weight": 0.8},
        ],
        "脑膜炎": [
            {"rel": "典型症状", "target": "头痛", "target_type": "symptom", "weight": 0.9},
            {"rel": "典型症状", "target": "发烧", "target_type": "symptom", "weight": 0.9},
            {"rel": "典型症状", "target": "颈强直", "target_type": "symptom", "weight": 0.95},
            {"rel": "推荐治疗", "target": "抗生素", "target_type": "treatment", "weight": 0.9},
        ],
        # —— 药物作为 source 的事实（支持"X 是什么药"类事实查询）——
        "二甲双胍": [
            {"rel": "药物类别", "target": "双胍类降糖药", "target_type": "drug_class", "weight": 0.95},
            {"rel": "适应症", "target": "糖尿病", "target_type": "disease", "weight": 0.95},
            {"rel": "常见副作用", "target": "胃肠道不适", "target_type": "symptom", "weight": 0.6},
        ],
        "胰岛素": [
            {"rel": "药物类别", "target": "降糖激素", "target_type": "drug_class", "weight": 0.95},
            {"rel": "适应症", "target": "糖尿病", "target_type": "disease", "weight": 0.95},
        ],
        "布洛芬": [
            {"rel": "药物类别", "target": "非甾体抗炎药", "target_type": "drug_class", "weight": 0.95},
            {"rel": "适应症", "target": "头痛", "target_type": "symptom", "weight": 0.8},
            {"rel": "适应症", "target": "发烧", "target_type": "symptom", "weight": 0.8},
        ],
    }
    
    MOCK_PAGERANK = {
        "双胍类降糖药": 0.02, "降糖激素": 0.02, "非甾体抗炎药": 0.02,
        "胃肠道不适": 0.015,
        "脑膜炎": 0.085, "糖尿病": 0.092, "高血压": 0.088, "肺炎": 0.075,
        "头痛": 0.062, "发烧": 0.065, "颈强直": 0.041,
        "布洛芬": 0.038, "二甲双胍": 0.044, "胰岛素": 0.048,
    }
    
    async def query_neighbors(self, entity: str, depth: int = 1, limit: int = 20) -> List[Dict]:
        facts = self.MOCK_FACTS.get(entity, [])[:limit]
        # 附加 source entity 信息
        for f in facts:
            f["source"] = entity
        return facts
    
    async def get_pagerank(self, node_id: str) -> float:
        return self.MOCK_PAGERANK.get(node_id, 0.01)


# ============================================================
# 生产环境 Neo4j Backend（骨架）
# ============================================================

class Neo4jBackend(KGBackend):
    """
    Neo4j 真实后端
    
    依赖: pip install neo4j
    使用前请确保 Neo4j 服务在运行（默认 bolt://localhost:7687）
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        # 真实实现时启用以下代码：
        # from neo4j import GraphDatabase
        # self._driver = GraphDatabase.driver(uri, auth=(user, password))
    
    async def query_neighbors(self, entity: str, depth: int = 1, limit: int = 20) -> List[Dict]:
        cypher = f"""
        MATCH (e {{name: $entity}})-[r1*1..{depth}]->(n)
        RETURN n.name AS target, n.type AS target_type, 
               last(r1).type AS rel, last(r1).weight AS weight
        ORDER BY weight DESC
        LIMIT $limit
        """
        # 真实实现：
        # async with self._driver.session() as session:
        #     result = await session.run(cypher, entity=entity, limit=limit)
        #     return [dict(r) for r in await result.data()]
        raise NotImplementedError("启用 Neo4j 前请安装 neo4j 驱动并取消注释")
    
    async def get_pagerank(self, node_id: str) -> float:
        # 通常 PageRank 离线计算后存到 Redis / KV store
        raise NotImplementedError


# ============================================================
# SQLite KG Backend（从真实 .db 读取，介于 Mock 与 Neo4j 之间）
# ============================================================

class SQLiteKGBackend(KGBackend):
    """
    从 SQLite .db 文件读取知识图谱。

    用 scripts/seed_database.py 生成的 data/medical_agent.db，
    读取 kg_relations / kg_pagerank 表。相比 MockKGBackend 的硬编码 dict，
    这是"连了真库"的版本；相比 Neo4jBackend，无需起 Neo4j 服务。

    线程安全说明: 每次查询用独立连接（check_same_thread=False + 短连接），
    适配 orchestrator 的 asyncio.to_thread / 并发调用。

    用法:
        kg = SQLiteKGBackend("data/medical_agent.db")
    """

    def __init__(self, db_path: str):
        import os
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"找不到数据库 {db_path}。请先运行: python scripts/seed_database.py"
            )
        self.db_path = db_path
        # 简单的 pagerank 缓存（启动时一次性载入，KG 静态）
        self._pr_cache = None

    def _connect(self):
        import sqlite3
        return sqlite3.connect(self.db_path, check_same_thread=False)

    async def query_neighbors(self, entity: str, depth: int = 1, limit: int = 20) -> List[Dict]:
        """查询实体的出边邻居（与 MockKGBackend 返回结构一致）"""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT r.rel_type, r.dst, r.weight, e.type
                FROM kg_relations r
                LEFT JOIN kg_entities e ON e.name = r.dst
                WHERE r.src = ?
                ORDER BY r.weight DESC
                LIMIT ?
                """,
                (entity, limit),
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "rel": rel_type,
                "target": dst,
                "target_type": dst_type or "unknown",
                "weight": weight,
                "source": entity,
            }
            for (rel_type, dst, weight, dst_type) in rows
        ]

    async def get_pagerank(self, node_id: str) -> float:
        if self._pr_cache is None:
            conn = self._connect()
            try:
                self._pr_cache = {
                    name: score
                    for name, score in conn.execute("SELECT name, score FROM kg_pagerank")
                }
            finally:
                conn.close()
        return self._pr_cache.get(node_id, 0.01)


# ============================================================
# Local Search Tool
# ============================================================

class KGLocalSearchTool(BaseTool):
    name = "kg_local_search"
    description = (
        "GraphRAG 实体级检索。适合具体事实查询（如'X 的症状'、'Y 的用法'）。"
        "输入: {query: str, entities: Optional[List[str]], max_results: int}。"
        "返回: 实体邻居关系列表，按 PageRank 重排。"
    )
    
    def __init__(self, kg_backend: KGBackend, default_depth: int = 2):
        self.kg = kg_backend
        self.default_depth = default_depth
    
    async def _execute(self, input_data: Any) -> Dict[str, Any]:
        # 输入解析
        if isinstance(input_data, str):
            query = input_data
            entities = []
        else:
            query = input_data.get("query", "")
            entities = input_data.get("entities", [])
        
        max_results = input_data.get("max_results", 15) if isinstance(input_data, dict) else 15
        depth = input_data.get("depth", self.default_depth) if isinstance(input_data, dict) else self.default_depth
        
        # 如果没传 entities，从 query 中粗略提取（生产环境应该让 Planner 传 NER 结果）
        if not entities:
            entities = self._naive_entity_extract(query)
        
        # 1-hop / 2-hop 邻居展开
        all_facts = []
        for entity in entities:
            neighbors = await self.kg.query_neighbors(entity, depth=depth, limit=max_results)
            all_facts.extend(neighbors)
        
        # PageRank 重排
        for fact in all_facts:
            fact["pagerank"] = await self.kg.get_pagerank(fact["target"])
        all_facts.sort(key=lambda x: (x.get("pagerank", 0) * x.get("weight", 0.5)), reverse=True)
        
        return {
            "mode": "local",
            "entities_queried": entities,
            "facts": all_facts[:max_results],
            "fact_count": len(all_facts[:max_results]),
        }
    
    def _extract_citations(self, data: Any) -> List[Dict]:
        """把检索到的事实作为 citations 返回"""
        if not isinstance(data, dict):
            return []
        return [
            {"type": "kg_fact", "source": f["source"], "rel": f["rel"], "target": f["target"]}
            for f in data.get("facts", [])
        ]
    
    def _naive_entity_extract(self, query: str) -> List[str]:
        """简单的实体提取兜底（生产环境应由 NER Tool 提供）"""
        from .ner_tool import MOCK_ENTITY_LEXICON
        found = []
        for terms in MOCK_ENTITY_LEXICON.values():
            for term in terms:
                if term in query:
                    found.append(term)
        return found
