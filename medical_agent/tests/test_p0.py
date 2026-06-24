"""
tests/test_p0.py — P0 改动的单元测试（安全网）

覆盖本轮 P0 三项改动:
1. DAG 引用解析 (_resolve_input) + Speculative Pre-fetch
2. 语义 Mock Embedder + Episodic 混合检索
3. GraphRAG 社区检测 + 摘要生成 + 回填验证

设计为**无需 pytest** 即可运行（纯标准库 assert）:
    cd medical_agent
    python -m tests.test_p0

回去装了 pytest 后，每个 test_* 函数也能直接被 pytest 采集。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import ToolResult, UserQuery


# ============================================================
# 1. DAG 引用解析
# ============================================================

def test_resolve_input_preserves_list_type():
    """单占位符引用 list 字段时，应返回真正的 list 而非字符串"""
    from demo.demo_full_flow import build_system
    orch, _ = build_system()
    results = {
        1: ToolResult(tool_name="ner", success=True,
                      data={"all_entities": ["头痛", "发烧"], "symptom": ["头痛"]})
    }
    out = orch._resolve_input("${step1.output.all_entities}", results, query="q")
    assert isinstance(out, list), f"应为 list，实际 {type(out)}"
    assert out == ["头痛", "发烧"]


def test_resolve_input_query_injection():
    """${query} 应被替换为真实 query"""
    from demo.demo_full_flow import build_system
    orch, _ = build_system()
    out = orch._resolve_input({"q": "${query}"}, {}, query="头痛三天")
    assert out["q"] == "头痛三天"


def test_resolve_input_embedded_string():
    """内嵌占位符做字符串替换"""
    from demo.demo_full_flow import build_system
    orch, _ = build_system()
    results = {1: ToolResult(tool_name="ner", success=True, data={"symptom": ["头痛"]})}
    out = orch._resolve_input("症状: ${step1.output.symptom}", results, query="q")
    assert isinstance(out, str)
    assert "头痛" in out


def test_resolve_input_missing_ref_kept():
    """无法解析的引用应原样保留，不崩溃"""
    from demo.demo_full_flow import build_system
    orch, _ = build_system()
    out = orch._resolve_input("${step9.output}", {}, query="q")
    assert out == "${step9.output}"


def test_speculative_prefetch_stats():
    """鉴别诊断查询应触发 prefetch 并累计统计"""
    from demo.demo_full_flow import build_system
    orch, _ = build_system()

    async def run():
        q = UserQuery(user_id="ptest",
                      text="我最近三天持续头痛伴发烧 38.5°C，颈部僵硬，可能是什么病？")
        await orch.answer_async(q)
    asyncio.run(run())

    assert orch.prefetch_stats["total_prefetched"] > 0, "应有预取候选"
    assert orch.prefetch_stats["total_queries_with_prefetch"] == 1
    # 命中率应在 [0, 1]
    hr = orch.get_prefetch_hit_rate()
    assert 0.0 <= hr <= 1.0


# ============================================================
# 2. 语义 Mock Embedder + Episodic 检索
# ============================================================

def test_semantic_embedder_similarity_ordering():
    """相关概念的相似度应高于无关概念"""
    from src.memory.mock_embedder import SemanticMockEmbedder
    from src.memory.episodic_memory import cosine_similarity
    emb = SemanticMockEmbedder()

    sim_related = cosine_similarity(emb.embed("糖尿病视力模糊"),
                                    emb.embed("糖尿病视网膜病变眼底"))
    sim_unrelated = cosine_similarity(emb.embed("头痛发烧"),
                                      emb.embed("高血压血压"))
    assert sim_related > 0.5, f"相关应 >0.5，实际 {sim_related:.3f}"
    assert sim_unrelated < 0.2, f"无关应 <0.2，实际 {sim_unrelated:.3f}"
    assert sim_related > sim_unrelated


def test_semantic_embedder_deterministic():
    """同样输入必须得到同样向量（确定性）"""
    from src.memory.mock_embedder import SemanticMockEmbedder
    emb = SemanticMockEmbedder()
    assert emb.embed("头痛发烧") == emb.embed("头痛发烧")


def test_episodic_hybrid_retrieval_recalls_old_chronic():
    """老的慢性病 episode 在相关查询下应被召回（混合检索核心卖点）"""
    import datetime
    from src.memory.episodic_memory import (
        EpisodicMemory, Episode, SQLiteEpisodicBackend, MockEmbedder, ImportanceScorer,
    )
    ep = EpisodicMemory(backend=SQLiteEpisodicBackend(":memory:"),
                        embedder=MockEmbedder(), scorer=ImportanceScorer())
    now = datetime.datetime.now()
    ep.write(Episode(user_id="u1", diagnoses=["糖尿病"], medications=["二甲双胍"],
                     symptoms=["多饮", "多尿"], summary="确诊2型糖尿病服用二甲双胍",
                     timestamp=now - datetime.timedelta(days=300)))
    ep.write(Episode(user_id="u1", diagnoses=["高血压"], medications=["氨氯地平"],
                     symptoms=["头晕"], summary="高血压随访服用氨氯地平",
                     timestamp=now - datetime.timedelta(days=5)))

    res = ep.retrieve("u1", "糖尿病最近视力模糊有风险吗", top_k=2)
    assert res, "应有召回结果"
    assert "糖尿病" in res[0].summary, f"老糖尿病记录应排第一，实际: {res[0].summary}"


# ============================================================
# 3. GraphRAG
# ============================================================

def _build_demo_graph():
    from src.tools.kg_local_search import MockKGBackend
    kg = MockKGBackend()
    edges = []
    for src, facts in kg.MOCK_FACTS.items():
        for f in facts:
            edges.append((src, f["target"], f.get("weight", 0.5)))
    return kg, edges


def test_community_detection_hierarchical():
    """分辨率越高，社区数应不减少（更细碎）"""
    from src.graphrag import CommunityDetector
    _, edges = _build_demo_graph()
    det = CommunityDetector(edges)
    levels = det.detect_hierarchical(resolutions=[0.5, 1.0, 1.5])
    n0 = det.num_communities(levels["l0"])
    n2 = det.num_communities(levels["l2"])
    assert n2 >= n0, f"细粒度社区数 {n2} 应 >= 粗粒度 {n0}"


def test_community_groups_meningitis_together():
    """脑膜炎与其典型症状应聚到同一社区"""
    from src.graphrag import CommunityDetector
    _, edges = _build_demo_graph()
    det = CommunityDetector(edges)
    labels = det.detect(resolution=1.0)
    assert labels["脑膜炎"] == labels["颈强直"], "脑膜炎与颈强直应同社区"


def test_summary_generation_no_hallucination():
    """正常社区摘要不应被判定为幻觉"""
    from src.graphrag import CommunitySummaryGenerator
    gen = CommunitySummaryGenerator()
    res = gen.generate("L1_C001", level=1,
                       entities=["脑膜炎", "头痛", "发烧", "颈强直"],
                       relations=[{"src": "脑膜炎", "rel": "典型症状", "dst": "颈强直"}])
    assert not res.is_hallucinated
    assert res.theme == "中枢神经系统感染"


def test_summary_verification_catches_hallucination():
    """注入社区外实体应被回填验证拦截"""
    from src.graphrag import CommunitySummaryGenerator
    gen = CommunitySummaryGenerator(hallucination_threshold=0.1)
    fake = "本社区核心概念包括脑膜炎、头痛、白血病、肝硬化，需联合诊断。"
    v = gen.verify_summary(fake, community_entities=["脑膜炎", "头痛", "发烧"])
    assert v["is_hallucinated"]
    assert "白血病" in v["hallucinated_entities"]


# ============================================================
# Runner（无 pytest 时用）
# ============================================================

def _run_all():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}\n通过 {passed} / {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    print("Running P0 unit tests...\n")
    ok = _run_all()
    sys.exit(0 if ok else 1)
