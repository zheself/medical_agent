"""
tests/test_core.py — 核心算法 + 消融开关单元测试

覆盖:
- Personalized PageRank（算法正确性 + alpha 行为 + Tool 集成）
- L1 规则反思（过敏/年龄/性别/急症/引用）
- DAG 调度（并行、依赖、死锁兜底、引用解析集成）
- 消融开关（prefetch / memory / verifier 级别真实生效）

无需 pytest:
    cd medical_agent
    python -m tests.test_core
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import (
    ComplexityLevel, MemoryUpdate, PatientProfile, Plan, PlanStep, ToolResult, UserQuery,
)


# ============================================================
# Personalized PageRank
# ============================================================

def test_ppr_basic_convergence():
    """链式图 A-B-C，从 A 出发，A 的分数应最高，所有分数非负且和约为 1"""
    from src.tools.ppr_reasoner import SimpleGraph, personalized_pagerank
    g = SimpleGraph()
    g.add_edge("A", "B", 1.0)
    g.add_edge("B", "C", 1.0)
    pr = personalized_pagerank(g, ["A"], alpha=0.5)
    assert all(v >= 0 for v in pr.values())
    assert pr["A"] >= pr["B"] >= pr["C"], f"距 seed 越远分数应越低: {pr}"
    assert abs(sum(pr.values()) - 1.0) < 0.1, f"分数和应接近 1: {sum(pr.values())}"


def test_ppr_alpha_effect():
    """alpha 越低，重启越频繁，seed 自身保留的质量应越高"""
    from src.tools.ppr_reasoner import SimpleGraph, personalized_pagerank
    g = SimpleGraph()
    for a, b in [("S", "X"), ("X", "Y"), ("Y", "Z")]:
        g.add_edge(a, b, 1.0)
    pr_low = personalized_pagerank(g, ["S"], alpha=0.3)
    pr_high = personalized_pagerank(g, ["S"], alpha=0.85)
    assert pr_low["S"] > pr_high["S"], "alpha 低时 seed 自身分数更高"


def test_ppr_empty_graph():
    """空图不应崩溃"""
    from src.tools.ppr_reasoner import SimpleGraph, personalized_pagerank
    pr = personalized_pagerank(SimpleGraph(), ["A"], alpha=0.5)
    assert pr == {}


def test_ppr_tool_identifies_meningitis():
    """PPR Tool 在 mock KG 上，头痛+发烧+颈强直应扩散出脑膜炎为最相关"""
    from src.tools.ppr_reasoner import PPRReasonerTool
    from src.tools.kg_local_search import MockKGBackend
    tool = PPRReasonerTool(kg_backend=MockKGBackend(), alpha=0.5)
    res = asyncio.run(tool.ainvoke({"seed_entities": ["头痛", "发烧", "颈强直"], "top_k": 5}))
    assert res.success
    top = res.data["relevant_concepts"][0]["entity"]
    assert top == "脑膜炎", f"应识别脑膜炎为最相关，实际 {top}"


# ============================================================
# L1 规则反思
# ============================================================

def _l1():
    from src.verifiers.l1_rule_verifier import L1RuleVerifier
    return L1RuleVerifier()

def test_l1_allergy_conflict():
    """给过敏患者推荐过敏药应被拦截"""
    ans = {"content": "建议青霉素治疗", "recommended_drugs": ["青霉素"],
           "possible_diagnoses": ["细菌感染"], "citations": [{"x": 1}]}
    res = _l1().verify(ans, PatientProfile(age=30, gender="男", allergies=["青霉素"]))
    assert not res.passed
    assert any("ALLERGY" in e for e in res.errors)


def test_l1_emergency_needs_referral():
    """涉及急症但无就医建议应被拦截"""
    ans = {"content": "可能是普通头痛", "recommended_drugs": [],
           "possible_diagnoses": ["脑膜炎"], "citations": [{"x": 1}]}
    res = _l1().verify(ans, PatientProfile(age=30))
    assert not res.passed
    assert any("EMERGENCY" in e for e in res.errors)


def test_l1_emergency_with_referral_passes():
    """急症 + 明确就医建议应通过急症规则"""
    ans = {"content": "高度怀疑脑膜炎，请立即就医急诊", "recommended_drugs": [],
           "possible_diagnoses": ["脑膜炎"], "citations": [{"x": 1}]}
    res = _l1().verify(ans, PatientProfile(age=30))
    assert not any("EMERGENCY" in e for e in res.errors)


def test_l1_missing_citation():
    """长回答无引用应被拦截"""
    ans = {"content": "这是一段需要引用支持的较长医学回答内容用于触发引用完整性检查规则" * 2,
           "recommended_drugs": [], "possible_diagnoses": [], "citations": []}
    res = _l1().verify(ans, PatientProfile())
    assert any("CITATION" in e for e in res.errors)


def test_l1_clean_answer_passes():
    """无违规的回答应通过"""
    ans = {"content": "建议多休息多喝水", "recommended_drugs": ["对乙酰氨基酚"],
           "possible_diagnoses": ["普通感冒"], "citations": [{"x": 1}]}
    res = _l1().verify(ans, PatientProfile(age=30, gender="男"))
    assert res.passed, f"不应有错误: {res.errors}"


def test_l1_is_fast():
    """L1 应该是毫秒级（0 LLM 调用）"""
    ans = {"content": "x", "recommended_drugs": [], "possible_diagnoses": [], "citations": [{"x": 1}]}
    res = _l1().verify(ans, PatientProfile())
    assert res.elapsed_ms < 50, f"L1 应 <50ms，实际 {res.elapsed_ms}"


# ============================================================
# DAG 调度
# ============================================================

def _orch():
    from demo.demo_full_flow import build_system
    orch, _ = build_system()
    return orch

def test_dag_parallel_layer_executes():
    """同层无依赖的步骤都应被执行"""
    orch = _orch()
    plan = Plan(
        thought="t", complexity=ComplexityLevel.MEDIUM,
        steps=[
            PlanStep(step_id=1, tool="ner", input="${query}", depends_on=[]),
            PlanStep(step_id=2, tool="kg_local_search", input={"query": "${query}"}, depends_on=[]),
        ],
    )
    results = asyncio.run(orch._execute_plan(plan, query="头痛发烧"))
    assert set(results.keys()) == {1, 2}
    assert results[1].success and results[2].success


def test_dag_dependency_order():
    """依赖步骤应在被依赖步骤之后执行，且能拿到上游输出"""
    orch = _orch()
    plan = Plan(
        thought="t", complexity=ComplexityLevel.HIGH,
        steps=[
            PlanStep(step_id=1, tool="ner", input="${query}", depends_on=[]),
            PlanStep(step_id=2, tool="ppr_reasoner",
                     input={"seed_entities": "${step1.output.all_entities}"}, depends_on=[1]),
        ],
    )
    results = asyncio.run(orch._execute_plan(plan, query="头痛发烧颈强直"))
    assert results[2].success
    # step2 应真正拿到了 step1 抽取的实体（list 透传）
    assert results[2].data["seed_entities"], "PPR 应拿到非空 seed"


def test_dag_deadlock_handled():
    """循环依赖不应卡死，应标记失败"""
    orch = _orch()
    plan = Plan(
        thought="t", complexity=ComplexityLevel.LOW,
        steps=[
            PlanStep(step_id=1, tool="ner", input="x", depends_on=[2]),
            PlanStep(step_id=2, tool="ner", input="x", depends_on=[1]),
        ],
    )
    results = asyncio.run(orch._execute_plan(plan, query="q"))
    assert all(not r.success for r in results.values())
    assert all("deadlock" in (r.error or "") for r in results.values())


# ============================================================
# 消融开关真实生效
# ============================================================

def test_switch_prefetch():
    """enable_prefetch 开关应真实控制预取"""
    orch = _orch()
    q = UserQuery(user_id="u1", text="头痛发烧颈强直可能是什么病？")

    orch.enable_prefetch = False
    asyncio.run(orch.answer_async(q, session_id="s_off"))
    assert orch.prefetch_stats["total_prefetched"] == 0

    orch2 = _orch()
    orch2.enable_prefetch = True
    asyncio.run(orch2.answer_async(q, session_id="s_on"))
    assert orch2.prefetch_stats["total_prefetched"] > 0


def test_switch_verifier_level():
    """max_verifier_level=L1 应让反思只到 L1"""
    orch = _orch()
    orch.max_verifier_level = "L1"
    orch.verifier.max_level = "L1"
    q = UserQuery(user_id="u1", text="头痛发烧颈强直可能是什么病？")
    ans = asyncio.run(orch.answer_async(q))
    levels = [vr.level.value for vr in ans.verification_results]
    assert "L3" not in levels and "L2" not in levels, f"应只到 L1，实际 {levels}"


def test_switch_memory_injection():
    """关闭 memory injection 后，冷启动不应注入历史过敏史"""
    import datetime
    from src.memory.episodic_memory import (
        Episode, EpisodicMemory, SQLiteEpisodicBackend, MockEmbedder, ImportanceScorer,
    )
    from demo.demo_full_flow import build_system
    orch, episodic = build_system()
    # 写一条带过敏的历史
    episodic.write(Episode(user_id="umem", diagnoses=["偏头痛"], symptoms=["头痛"],
                           summary="既往就诊 过敏 青霉素 记录",
                           timestamp=datetime.datetime.now()))
    orch.enable_memory_injection = False
    q = UserQuery(user_id="umem", text="头痛")
    asyncio.run(orch.answer_async(q, session_id="s_nomem"))
    wm = orch.wm_pool["s_nomem"]
    # 关闭注入后，allergies 不应被冷启动填充
    assert wm.patient_profile.allergies == [], f"不该注入历史: {wm.patient_profile.allergies}"


def test_critical_facts_extracts_clean_allergen():
    """retrieve_critical_facts 应提取干净的过敏原，而非整段摘要"""
    from src.memory.episodic_memory import (
        EpisodicMemory, Episode, SQLiteEpisodicBackend, MockEmbedder, ImportanceScorer,
    )
    ep = EpisodicMemory(SQLiteEpisodicBackend(":memory:"), MockEmbedder(), ImportanceScorer())
    ep.write(Episode(user_id="u", diagnoses=["糖尿病"], medications=["二甲双胍"],
                     summary="确诊2型糖尿病服用二甲双胍青霉素过敏"))
    facts = ep.retrieve_critical_facts("u")
    assert facts["allergies"] == ["青霉素"], f"应只提取过敏原: {facts['allergies']}"
    assert "糖尿病" in facts["chronic_diseases"]


def test_mock_complexity_not_polluted_by_history():
    """简单事实查询不应因历史症状被误判为 high（回归测试）"""
    import asyncio as _aio
    from demo.demo_full_flow import build_system
    orch, _ = build_system()
    # 先问一个复杂症状问题（污染 working memory）
    _aio.run(orch.answer_async(UserQuery(user_id="u", text="头痛发烧颈强直"), session_id="s"))
    # 再问简单事实查询，复杂度应为 low
    ans = _aio.run(orch.answer_async(UserQuery(user_id="u", text="二甲双胍是什么药？"), session_id="s"))
    assert ans.plan.complexity.value == "low", f"应为 low，实际 {ans.plan.complexity.value}"


# ============================================================
# Runner
# ============================================================

def _run_all():
    passed, failed = 0, 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
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
    print("Running core unit tests...\n")
    sys.exit(0 if _run_all() else 1)
