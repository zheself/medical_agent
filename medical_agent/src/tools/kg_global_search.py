"""
tools/kg_global_search.py — GraphRAG Global Search 工具

社区级检索，适合鉴别诊断、多跳推理等需要全局视角的查询。
工作流程：
1. Query 向量化 → 在社区摘要向量库中召回 top-K 社区
2. 从命中社区中提取候选实体作为 PPR 的 seed
3. 跑 Personalized PageRank 找出"扩散"出的相关概念
4. 返回 社区摘要 + 关键事实

实现状态: ✅ 接口完整 / 🟡 向量库部分提供 Mock
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import CommunitySummary
from .base import BaseTool


# ============================================================
# 向量库抽象
# ============================================================

class VectorStore:
    """向量库抽象，便于切换 Zilliz / FAISS / Mock"""
    
    async def search(self, embedding: List[float], top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict]:
        raise NotImplementedError


class MockCommunityVectorStore(VectorStore):
    """Mock 实现：预置几个社区摘要，按关键词匹配"""
    
    MOCK_COMMUNITIES = [
        {
            "community_id": "L1_C001",
            "level": 1,
            "theme": "中枢神经系统感染",
            "summary": (
                "本社区涵盖脑膜炎、脑炎、脑脓肿等中枢神经系统感染性疾病。"
                "核心鉴别要点包括: 头痛 + 发热 + 颈强直三联征是脑膜炎的典型表现; "
                "蛛网膜下腔出血也可表现为剧烈头痛但通常为突发性。"
                "诊断金标准是腰椎穿刺脑脊液检查。治疗原则是早期经验性抗生素 + 针对性抗病毒/抗结核。"
            ),
            "core_entities": ["脑膜炎", "脑炎", "头痛", "发烧", "颈强直", "蛛网膜下腔出血"],
            "match_keywords": ["头痛", "颈强直", "脑膜", "颈部僵硬", "畏光"],
        },
        {
            "community_id": "L1_C002",
            "level": 1,
            "theme": "糖尿病及其并发症",
            "summary": (
                "糖尿病的慢性并发症主要分为微血管病变（视网膜病变、肾病、神经病变）"
                "和大血管病变（冠心病、脑卒中、外周血管病）。"
                "视力模糊、视物变形是糖尿病视网膜病变的早期信号。"
                "二甲双胍是 2 型糖尿病一线用药，胰岛素用于 1 型及晚期 2 型。"
            ),
            "core_entities": ["糖尿病", "视网膜病变", "糖尿病肾病", "二甲双胍", "胰岛素"],
            "match_keywords": ["糖尿病", "视力", "视网膜", "血糖"],
        },
        {
            "community_id": "L1_C003",
            "level": 1,
            "theme": "上呼吸道感染与流感",
            "summary": (
                "本社区包括普通感冒、流感、急性咽炎、扁桃体炎等。"
                "流感典型表现为高热（39°C 以上）+ 全身肌肉酸痛 + 乏力；"
                "普通感冒多以鼻塞、流涕为主，发热轻。"
                "对症治疗 + 流感时使用奥司他韦。"
            ),
            "core_entities": ["流感", "感冒", "发烧", "咳嗽"],
            "match_keywords": ["发烧", "感冒", "流感", "咳嗽"],
        },
    ]
    
    async def search(self, embedding: List[float], top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict]:
        # Mock: 不真做向量计算，按调用者注入的 query 关键词匹配
        # 真实实现：vector_db.search(embedding=embedding, top_k=top_k)
        return self.MOCK_COMMUNITIES[:top_k]


def naive_text_match_communities(query: str, communities: List[Dict], top_k: int = 3) -> List[Dict]:
    """Mock 版本的"伪向量检索"：按关键词命中数排序"""
    scored = []
    for c in communities:
        score = sum(1 for kw in c["match_keywords"] if kw in query)
        if score > 0:
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k]]


# ============================================================
# Embedder 抽象
# ============================================================

class Embedder:
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class MockEmbedder(Embedder):
    """
    确定性语义 Mock Embedder（委托给 SemanticMockEmbedder）。

    注意: 当前 MockCommunityVectorStore 走的是关键词匹配路径（见
    naive_text_match_communities），不直接用向量。但保留语义向量实现，
    以便切换到真实向量库时检索路径行为可演示、可复现。

    生产环境替换为 BAAI/bge-m3 等真实模型。
    """

    def __init__(self):
        from ..memory.mock_embedder import SemanticMockEmbedder
        self._impl = SemanticMockEmbedder()

    def embed(self, text: str) -> List[float]:
        return self._impl.embed(text)


# ============================================================
# Global Search Tool
# ============================================================

class KGGlobalSearchTool(BaseTool):
    name = "kg_global_search"
    description = (
        "GraphRAG 社区级检索。适合需要全局视角的查询（鉴别诊断、可能病因、跨概念推理）。"
        "输入: {query: str, focus: Optional[str], top_k: int}。"
        "返回: 社区摘要 + 核心实体 + 关键事实。"
    )
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k_communities: int = 3,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k_communities = top_k_communities
    
    async def _execute(self, input_data: Any) -> Dict[str, Any]:
        if isinstance(input_data, str):
            query = input_data
            focus = None
        else:
            query = input_data.get("query", "")
            focus = input_data.get("focus")
        
        top_k = (
            input_data.get("top_k", self.top_k_communities)
            if isinstance(input_data, dict) else self.top_k_communities
        )
        
        # 检索相关社区
        query_emb = self.embedder.embed(query)
        if isinstance(self.vector_store, MockCommunityVectorStore):
            # Mock 路径：用关键词匹配模拟向量检索
            communities = naive_text_match_communities(
                query, self.vector_store.MOCK_COMMUNITIES, top_k=top_k
            )
        else:
            communities = await self.vector_store.search(query_emb, top_k=top_k)
        
        # 从命中社区收集候选实体（作为后续 PPR seed）
        candidate_entities = []
        seen = set()
        for c in communities:
            for ent in c.get("core_entities", []):
                if ent not in seen:
                    candidate_entities.append(ent)
                    seen.add(ent)
        
        # 整理输出
        return {
            "mode": "global",
            "focus": focus,
            "communities": [
                {
                    "id": c["community_id"],
                    "theme": c["theme"],
                    "summary": c["summary"],
                    "core_entities": c["core_entities"],
                }
                for c in communities
            ],
            "candidate_entities": candidate_entities,
            "instruction_to_planner": (
                "可基于上述社区摘要做鉴别诊断推理，"
                "如需进一步多跳推理，请调用 ppr_reasoner 工具。"
            ),
        }
    
    def _extract_citations(self, data: Any) -> List[Dict]:
        if not isinstance(data, dict):
            return []
        return [
            {"type": "community_summary", "id": c["id"], "theme": c["theme"]}
            for c in data.get("communities", [])
        ]
