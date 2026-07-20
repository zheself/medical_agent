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
    VerifierLevel, VerifyResult,
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


def test_l1_handles_string_draft_answer():
    """L1 防御：draft_answer 是字符串时不应崩溃，应转为 dict"""
    verifier = _l1()
    res = verifier.verify("plain string answer", PatientProfile())
    assert res.passed or not res.passed  # 不应抛异常


def test_l3_citation_string_normalized():
    """L3 reflexion _parse_reflexion: citations 中的 string 应被转为 dict"""
    from src.verifiers.l3_reflexion import L3Reflexion
    # 模拟 LLM 返回的 JSON 中 citations 是 string list
    response = '{"corrected_answer": {"content": "test", "citations": ["str_cite1", {"type": "kg", "rel": "has", "source": "X", "target": "Y"}]}}'
    parsed = L3Reflexion._parse_reflexion(response)
    cites = parsed["corrected_answer"]["citations"]
    assert isinstance(cites, list)
    assert all(isinstance(c, dict) for c in cites)
    # string 应被包装
    assert cites[0]["type"] == "kg_inferred"
    assert cites[0]["source"] == "str_cite1"
    # 原 dict 应保持
    assert cites[1]["type"] == "kg"


def test_l3_citation_none_normalized():
    """L3 reflexion _parse_reflexion: citations=None 应转为空 list"""
    from src.verifiers.l3_reflexion import L3Reflexion
    response = '{"corrected_answer": {"content": "test"}}'
    parsed = L3Reflexion._parse_reflexion(response)
    assert isinstance(parsed["corrected_answer"]["citations"], list)
    assert parsed["corrected_answer"]["citations"] == []


# ============================================================
# L3 Diagnosis Merge Guard
# ============================================================

from src.verifiers.l3_reflexion import merge_l3_diagnoses


def test_merge_keeps_original_top3_when_l3_wipes():
    """原 ['胆囊结石', ...] + L3 ['腹水'] → 结果仍包含胆囊结石"""
    diag, content = merge_l3_diagnoses(
        original_diagnoses=["胆囊结石", "胆囊息肉", "胆囊腺肌症"],
        l3_diagnoses=["腹水"],
        l3_content="诊断：腹水",
    )
    assert "胆囊结石" in diag, f"应保留原 Top-1，实际: {diag}"
    assert diag[0] == "胆囊结石", f"原 Top-1 应排第一，实际: {diag}"
    assert len(diag) <= 5
    assert "综合候选诊断" in content


def test_merge_uses_l3_when_original_empty():
    """原空 + L3 有候选 → 使用 L3"""
    diag, content = merge_l3_diagnoses(
        original_diagnoses=[],
        l3_diagnoses=["胃食管反流病", "哮喘", "慢性支气管炎"],
        l3_content="诊断：胃食管反流病",
    )
    assert diag == ["胃食管反流病", "哮喘", "慢性支气管炎"]
    assert "综合候选诊断" in content


def test_merge_keeps_original_when_l3_empty():
    """L3 没有 possible_diagnoses → 保留原候选"""
    diag, content = merge_l3_diagnoses(
        original_diagnoses=["胆管炎", "胆囊炎"],
        l3_diagnoses=[],
        l3_content="无修正",
    )
    assert diag == ["胆管炎", "胆囊炎"]
    assert "综合候选诊断" not in content  # 没 merge 不追加摘要


def test_merge_dedup_and_truncate():
    """去重 + 截断到 5"""
    diag, _ = merge_l3_diagnoses(
        original_diagnoses=["A", "B", "C"],
        l3_diagnoses=["B", "D", "E", "F", "G", "H"],
        l3_content="...",
    )
    assert len(diag) == 5
    assert diag == ["A", "B", "C", "D", "E"]  # B 去重


def test_merge_handles_non_list_inputs():
    """防御：original_diagnoses 或 l3_diagnoses 为非法类型"""
    diag, _ = merge_l3_diagnoses(
        original_diagnoses="not_a_list",
        l3_diagnoses=None,
        l3_content="test",
    )
    assert diag == []

    diag2, _ = merge_l3_diagnoses(
        original_diagnoses=["A"],
        l3_diagnoses="not_a_list",
        l3_content="test",
    )
    assert diag2 == ["A"]


# ============================================================
# L3 Trigger Guard
# ============================================================

from src.verifiers.l3_reflexion import (
    GradedVerifierOrchestrator,
    _should_trigger_l3, _numeric_l2_scores, _L1_SAFETY_TAGS,
    TRIGGER_EMPTY_DIAGNOSES, TRIGGER_L1_SAFETY, TRIGGER_L1_CITATION,
    TRIGGER_LOW_L2_SCORE, SKIP_ENOUGH_CANDIDATES,
)


def _make_verify_result(passed=True, level=VerifierLevel.L1, errors=None,
                        scores=None, elapsed_ms=1):
    """构造 VerifyResult，兼容带 scores 的 L2 result"""
    vr = VerifyResult(passed=passed, level=level, errors=errors or [], elapsed_ms=elapsed_ms)
    if scores:
        vr.scores = scores
    return vr


def test_trigger_empty_diagnoses():
    """空预测 → 强制触发"""
    draft = {"possible_diagnoses": [], "content": "..."}
    l1 = _make_verify_result(passed=True)
    should, reason = _should_trigger_l3(draft, l1, None)
    assert should and reason == TRIGGER_EMPTY_DIAGNOSES


def test_trigger_l1_safety_error():
    """L1 安全错误 → 强制触发"""
    draft = {"possible_diagnoses": ["脑膜炎"], "content": "..."}
    l1 = _make_verify_result(passed=False, errors=["L1[EMERGENCY]: 缺少紧急就医建议"])
    should, reason = _should_trigger_l3(draft, l1, None)
    assert should and reason == TRIGGER_L1_SAFETY


def test_trigger_l1_citation_few_candidates():
    """L1 citation-only + 候选 ≤ 2 → 触发"""
    draft = {"possible_diagnoses": ["A"], "content": "..."}
    l1 = _make_verify_result(passed=False, errors=["L1[CITATION]: 缺少引用支持"])
    should, reason = _should_trigger_l3(draft, l1, None)
    assert should and reason == TRIGGER_L1_CITATION


def test_skip_enough_candidates():
    """候选 ≥ 3 且 L2 分数 ≥ 0.6 → 跳过"""
    draft = {"possible_diagnoses": ["A", "B", "C"], "content": "..."}
    l1 = _make_verify_result(passed=True)
    l2 = _make_verify_result(passed=False, level=VerifierLevel.L2,
                             scores={"faithfulness": 0.7, "relevance": 0.8, "factuality": 0.75})
    should, reason = _should_trigger_l3(draft, l1, l2)
    assert not should and reason == SKIP_ENOUGH_CANDIDATES


def test_trigger_low_l2_score():
    """L2 分数 < 0.5 → 触发"""
    draft = {"possible_diagnoses": ["A", "B", "C"], "content": "..."}
    l1 = _make_verify_result(passed=True)
    l2 = _make_verify_result(passed=False, level=VerifierLevel.L2,
                             scores={"faithfulness": 0.3, "relevance": 0.4, "factuality": 0.5})
    should, reason = _should_trigger_l3(draft, l1, l2)
    assert should and reason == TRIGGER_LOW_L2_SCORE


# --- _numeric_l2_scores edge cases ---

def test_numeric_l2_scores_filters_issues():
    """scores 包含 issues: [...] 时不崩溃，只取数值"""
    l2 = _make_verify_result(passed=False, level=VerifierLevel.L2,
                             scores={"faithfulness": 0.85, "relevance": 0.88, "factuality": 0.82, "issues": []})
    vals = _numeric_l2_scores(l2)
    assert vals == [0.85, 0.88, 0.82]


def test_numeric_l2_scores_all_non_numeric_returns_empty():
    """scores 只有非数值字段 → 返回 []"""
    l2 = _make_verify_result(passed=False, level=VerifierLevel.L2,
                             scores={"issues": [], "notes": "bad"})
    vals = _numeric_l2_scores(l2)
    assert vals == []


def test_trigger_with_issues_field_does_not_crash():
    """真实 L2 scores（含 issues:[]）→ 应正常 skip（分数够 + 候选够）"""
    draft = {"possible_diagnoses": ["A", "B", "C"], "content": "..."}
    l1 = _make_verify_result(passed=True)
    l2 = _make_verify_result(passed=False, level=VerifierLevel.L2,
                             scores={"faithfulness": 0.7, "relevance": 0.8, "factuality": 0.75, "issues": []})
    should, reason = _should_trigger_l3(draft, l1, l2)
    assert not should and reason == SKIP_ENOUGH_CANDIDATES


def test_trigger_no_numeric_scores_defaults_to_trigger():
    """无有效数值分数 → 走 default trigger（保守）"""
    draft = {"possible_diagnoses": ["A"], "content": "..."}
    l1 = _make_verify_result(passed=True)
    l2 = _make_verify_result(passed=False, level=VerifierLevel.L2,
                             scores={"issues": []})
    should, reason = _should_trigger_l3(draft, l1, l2)
    assert should  # 保守触发


# ============================================================
# L3 Trigger Guard 集成测试（GradedVerifierOrchestrator.verify）
# ============================================================

class _FakeL1:
    """Fake L1 verifier：可注入 errors 控制 passed/failed"""
    def __init__(self, passed=True, errors=None):
        self._passed = passed
        self._errors = errors or []
    def verify(self, draft_answer, patient_profile=None):
        return VerifyResult(passed=self._passed, level=VerifierLevel.L1,
                            errors=list(self._errors), elapsed_ms=1)


class _FakeL2:
    """Fake L2 verifier：可注入 scores 控制 passed/failed"""
    def __init__(self, passed=True, scores=None):
        self._passed = passed
        self._scores = scores or {}
    def verify(self, query, draft_answer, evidence):
        vr = VerifyResult(passed=self._passed, level=VerifierLevel.L2,
                          errors=[] if self._passed else ["L2 fail"], elapsed_ms=1)
        vr.scores = dict(self._scores)
        return vr


class _FakeL3:
    """Fake L3 reflexion：返回固定的 corrected_answer"""
    def reflect_and_correct(self, query, draft_answer, l1_result=None, l2_result=None):
        return {
            "corrected_answer": {
                "content": "L3 corrected content",
                "possible_diagnoses": ["L3诊断A", "L3诊断B"],
                "citations": [],
            },
            "verify_result": VerifyResult(passed=True, level=VerifierLevel.L3,
                                          errors=[], elapsed_ms=10),
        }


def test_verify_l1_safety_forces_l3():
    """L1 EMERGENCY fail + high complexity → verify() 必须返回 L3"""
    orch = GradedVerifierOrchestrator(
        l1_verifier=_FakeL1(passed=False, errors=["L1[EMERGENCY]: 缺少紧急就医建议"]),
        l2_verifier=_FakeL2(passed=True, scores={}),
        l3_reflexion=_FakeL3(),
        max_level="L3",
    )
    draft = {"possible_diagnoses": ["脑膜炎"], "content": "可能是脑膜炎"}
    result = orch.verify(
        query="头痛发烧脖子硬",
        draft_answer=draft,
        evidence=[],
        complexity="high",
    )
    assert result["level_reached"] == "L3", f"安全错误必须进 L3，实际: {result['level_reached']}"
    assert result.get("trigger_reason") == TRIGGER_L1_SAFETY


def test_verify_l2_fail_skip_l3_when_candidates_ok():
    """L2 fail + 候选 ≥ 3 + scores ≥ 0.6 → verify() 跳过 L3 返回 L2"""
    orch = GradedVerifierOrchestrator(
        l1_verifier=_FakeL1(passed=True, errors=[]),
        l2_verifier=_FakeL2(passed=False, scores={"faithfulness": 0.7, "relevance": 0.8, "factuality": 0.75}),
        l3_reflexion=_FakeL3(),
        max_level="L3",
    )
    draft = {"possible_diagnoses": ["A", "B", "C"], "content": "可能是A、B或C"}
    result = orch.verify(
        query="症状A、B、C",
        draft_answer=draft,
        evidence=[],
        complexity="high",
    )
    assert result["level_reached"] == "L2", f"候选够应跳过 L3，实际: {result['level_reached']}"
    assert result.get("l3_skip_reason") == SKIP_ENOUGH_CANDIDATES
    assert "needs_replan" not in result, "skip L3 不应带 needs_replan"


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


def test_ppr_off_sibling_dep_cleaned():
    """enable_ppr=False 时，依赖被删 PPR step 的后续 step 依赖应被清洗，不 deadlock"""
    orch = _orch()
    orch.enable_ppr = False
    plan = Plan(
        thought="t", complexity=ComplexityLevel.HIGH,
        steps=[
            PlanStep(step_id=1, tool="ner", input="${query}", depends_on=[]),
            PlanStep(step_id=2, tool="ppr_reasoner", input={"seed_entities": ["发热"]}, depends_on=[1]),
            PlanStep(step_id=3, tool="kg_global_search", input={"query": "${step2.output}"}, depends_on=[2]),
            PlanStep(step_id=4, tool="kg_local_search", input={"query": "${query}"}, depends_on=[]),
        ],
    )
    results = asyncio.run(orch._execute_plan(plan, query="头痛发烧"))
    # step 2 (PPR) 应被移除，不应出现在结果中
    assert 2 not in results, f"PPR step should be removed, got results keys: {results.keys()}"
    # step 3 的 depends_on=[2] 应被清洗为 []，正常执行
    assert 3 in results and results[3].success, f"step 3 should execute, got: {results.get(3)}"
    # step 1 和 4 不受影响
    assert 1 in results and results[1].success
    assert 4 in results and results[4].success


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
# 执行元数据（V9 parallel tools）
# ============================================================

class _FakeSleepTool:
    """Fake tool: 固定 sleep 后返回结果"""
    def __init__(self, sleep_s=0.05):
        self.sleep_s = sleep_s
        self.name = "sleep_tool"

    async def ainvoke(self, resolved_input, **kwargs):
        import asyncio
        s = self.sleep_s
        if isinstance(resolved_input, dict) and "sleep_s" in resolved_input:
            s = resolved_input["sleep_s"]
        await asyncio.sleep(s)
        return ToolResult(tool_name=self.name, success=True, elapsed_ms=s * 1000)


class _FakeToolRegistry:
    """只有 sleep tool 的 registry"""
    def __init__(self):
        self._tools = {}
    def get(self, name):
        if name not in self._tools:
            self._tools[name] = _FakeSleepTool()
        return self._tools[name]


def test_execution_meta_parallel_reduces_wall_time():
    """两个无依赖 sleep step → wall time 明显小于 sum elapsed"""
    orch = _orch()
    orch.tools = _FakeToolRegistry()
    plan = Plan(
        thought="t", complexity=ComplexityLevel.MEDIUM,
        steps=[
            PlanStep(step_id=1, tool="sleep_tool", input={"sleep_s": 0.1}, depends_on=[]),
            PlanStep(step_id=2, tool="sleep_tool", input={"sleep_s": 0.1}, depends_on=[]),
        ],
    )
    results, meta = asyncio.run(orch._execute_plan_with_meta(plan, query="test"))
    assert results[1].success and results[2].success
    assert meta["tool_sum_elapsed_ms"] >= 180, f"sum should be ~200ms, got {meta['tool_sum_elapsed_ms']}"
    assert meta["tool_wall_ms"] < 170, f"wall should be <170ms with parallel, got {meta['tool_wall_ms']}"
    assert meta["parallelism_ratio"] > 1.3, f"ratio should >1.3, got {meta['parallelism_ratio']}"
    assert meta["max_layer_width"] == 2, f"max width should be 2, got {meta['max_layer_width']}"
    assert meta["layer_widths"] == [2], f"should be single layer of 2, got {meta['layer_widths']}"


def test_execution_meta_dependency_serial():
    """两个 sleep step 有依赖 → wall time 接近 sum elapsed（串行）"""
    orch = _orch()
    orch.tools = _FakeToolRegistry()
    plan = Plan(
        thought="t", complexity=ComplexityLevel.MEDIUM,
        steps=[
            PlanStep(step_id=1, tool="sleep_tool", input={"sleep_s": 0.05}, depends_on=[]),
            PlanStep(step_id=2, tool="sleep_tool", input={"sleep_s": 0.05}, depends_on=[1]),
        ],
    )
    results, meta = asyncio.run(orch._execute_plan_with_meta(plan, query="test"))
    assert meta["layer_widths"] == [1, 1], f"should be [1, 1], got {meta['layer_widths']}"
    assert meta["max_layer_width"] == 1
    assert 0.8 < meta["parallelism_ratio"] < 1.3, f"ratio should be ~1.0, got {meta['parallelism_ratio']}"


def test_execution_meta_mixed_dag():
    """混合 DAG: 1 → [2, 3] → 4 → layer_widths 应为 [1, 2, 1]"""
    orch = _orch()
    orch.tools = _FakeToolRegistry()
    plan = Plan(
        thought="t", complexity=ComplexityLevel.MEDIUM,
        steps=[
            PlanStep(step_id=1, tool="sleep_tool", input={"sleep_s": 0.03}, depends_on=[]),
            PlanStep(step_id=2, tool="sleep_tool", input={"sleep_s": 0.03}, depends_on=[1]),
            PlanStep(step_id=3, tool="sleep_tool", input={"sleep_s": 0.03}, depends_on=[1]),
            PlanStep(step_id=4, tool="sleep_tool", input={"sleep_s": 0.03}, depends_on=[2, 3]),
        ],
    )
    results, meta = asyncio.run(orch._execute_plan_with_meta(plan, query="test"))
    assert meta["layer_widths"] == [1, 2, 1], f"got {meta['layer_widths']}"
    assert meta["max_layer_width"] == 2
    assert meta["layer_count"] == 3
    # Mixed DAG: layer 2 has 2 parallel tasks → ratio should be > 1.1
    assert meta["parallelism_ratio"] > 1.1, f"ratio should >1.1, got {meta['parallelism_ratio']}"


def test_execution_meta_exported_in_final_answer():
    """answer_async 的 FinalAnswer 应包含 execution_meta"""
    orch = _orch()
    q = UserQuery(user_id="u1", text="头痛发烧可能是什么病？")
    ans = asyncio.run(orch.answer_async(q))
    meta = ans.execution_meta
    assert isinstance(meta, dict), f"execution_meta should be dict, got {type(meta)}"
    assert "tool_wall_ms" in meta, f"missing tool_wall_ms in {meta.keys()}"
    assert "parallelism_ratio" in meta
    assert "layer_count" in meta
    assert "max_layer_width" in meta
    assert "layer_widths" in meta
    assert "executed_step_count" in meta


# ============================================================
# Memory Observability
# ============================================================

def test_memory_meta_exported_in_final_answer():
    """answer_async 的 FinalAnswer 应包含 memory_meta，字段完整"""
    orch = _orch()
    q = UserQuery(user_id="u_memory", text="头痛发烧可能是什么病？")
    ans = asyncio.run(orch.answer_async(q, session_id="s_memory"))
    meta = ans.memory_meta
    assert isinstance(meta, dict), f"memory_meta should be dict, got {type(meta)}"
    assert "scope" in meta
    assert "working" in meta
    assert "critical" in meta
    assert "episodic" in meta
    assert "injection" in meta
    assert meta["scope"]["user_id"] == "u_memory"
    assert meta["scope"]["session_id"] == "s_memory"
    assert "long_term_enabled" in meta["injection"]
    assert "long_term_context_chars" in meta["injection"]
    assert "working_context_chars" in meta["injection"]
    assert "total_context_chars_with_working" in meta["injection"]


def test_memory_meta_disabled_keeps_working_but_disables_long_term():
    """关闭 memory injection 后 working 仍运行，episodic 关闭，long_term=0"""
    orch = _orch()
    orch.enable_memory_injection = False
    q = UserQuery(user_id="u_nomem", text="头痛发烧可能是什么病？")
    ans = asyncio.run(orch.answer_async(q, session_id="s_nomem"))
    meta = ans.memory_meta
    assert meta["working"]["enabled"] is True
    assert meta["episodic"]["enabled"] is False
    assert meta["episodic"]["retrieved_count"] == 0
    assert meta["injection"]["long_term_enabled"] is False
    assert meta["injection"]["long_term_context_chars"] == 0


def test_memory_meta_scope_is_session_specific():
    """不同 session 的 scope 应不同"""
    orch = _orch()
    q1 = UserQuery(user_id="user_a", text="我青霉素过敏")
    a1 = asyncio.run(orch.answer_async(q1, session_id="session_a"))
    q2 = UserQuery(user_id="user_b", text="我能吃阿莫西林吗？")
    a2 = asyncio.run(orch.answer_async(q2, session_id="session_b"))
    assert a1.memory_meta["scope"]["session_id"] == "session_a"
    assert a2.memory_meta["scope"]["session_id"] == "session_b"
    assert a1.memory_meta["scope"]["session_id"] != a2.memory_meta["scope"]["session_id"]


# ============================================================
# Memory Gating (V10b)
# ============================================================

from src.memory.memory_gate import score_memory_relevance, gate_episodic_hints, MemoryGateDecision


def test_memory_gate_keeps_relevant_episode():
    """相关记忆应保持"""
    decision = score_memory_relevance(
        "头痛发烧颈强直可能是什么病？",
        {"summary": "患者有头痛、发烧、颈强直", "diagnoses": ["脑膜炎"]},
        threshold=0.2,
    )
    assert decision.keep, f"should keep, got {decision}"
    assert decision.score >= 0.2


def test_memory_gate_filters_irrelevant_episode():
    """不相关记忆应过滤"""
    decision = score_memory_relevance(
        "右上腹痛发热可能是什么病？",
        {"summary": "既往咨询过偏头痛", "diagnoses": ["偏头痛"]},
        threshold=0.2,
    )
    assert not decision.keep, f"should filter, got {decision}"


def test_memory_gate_disabled_keeps_all():
    """gating disabled 时全保留"""
    hints = [
        {"summary": "既往偏头痛", "diagnoses": ["偏头痛"]},
        {"summary": "既往腹痛", "diagnoses": ["胆囊炎"]},
    ]
    kept, records = gate_episodic_hints("头痛发烧", hints, enabled=False)
    assert len(kept) == 2, f"disabled should keep all, got {len(kept)}"
    assert all(r["reason"] == "gating_disabled" for r in records)


def test_memory_gate_empty_query():
    """空 query 应返回不保留"""
    decision = score_memory_relevance("", {"summary": "something"})
    assert not decision.keep
    assert decision.reason == "empty_query_or_memory"


def test_gate_records_in_memory_meta():
    """默认 enable_memory_gating=False，meta 应有 gating_enabled=False"""
    orch = _orch()
    assert orch.enable_memory_gating is False
    q = UserQuery(user_id="u_gate", text="头痛发烧可能是什么病？")
    ans = asyncio.run(orch.answer_async(q, session_id="s_gate"))
    meta = ans.memory_meta
    assert meta["episodic"]["gating_enabled"] is False
    assert meta["episodic"]["retrieved_count"] == meta["episodic"]["injected_count"]


# ============================================================
# Memory Gating 集成测试（fake episodic + orchestrator）
# ============================================================

class _FakeEpisodicMemory:
    """返回两条固定 episodic records：一条相关，一条不相关"""
    def retrieve_critical_facts(self, user_id):
        return {"allergies": [], "chronic_diseases": []}

    def retrieve(self, user_id, query, top_k=5):
        import datetime
        from src.memory.episodic_memory import Episode
        return [
            Episode(user_id=user_id, diagnoses=["脑膜炎"], symptoms=["头痛", "发烧"],
                    summary="患者曾因头痛发烧就诊，诊断为脑膜炎",
                    timestamp=datetime.datetime.now()),
            Episode(user_id=user_id, diagnoses=["糖尿病"], symptoms=["多饮", "多尿"],
                    summary="患者有糖尿病史，长期服用二甲双胍",
                    timestamp=datetime.datetime.now()),
        ]

    def write(self, episode, l3_triggered=False):
        return True


class _FakeEpisodicEmpty:
    """空 episodic"""
    def retrieve_critical_facts(self, user_id):
        return {"allergies": [], "chronic_diseases": []}
    def retrieve(self, user_id, query, top_k=5):
        return []
    def write(self, episode, l3_triggered=False):
        return True


def test_memory_gating_orchestrator_filters_irrelevant():
    """gating=True 时，orchestrator 应过滤不相关 episodic hint"""
    orch = _orch()
    orch.episodic = _FakeEpisodicMemory()
    orch.enable_memory_injection = True
    orch.enable_memory_gating = True
    orch.memory_gate_threshold = 0.2
    q = UserQuery(user_id="u_filter", text="头痛发烧颈强直可能是什么病？")
    ans = asyncio.run(orch.answer_async(q, session_id="s_filter"))
    meta = ans.memory_meta
    ep = meta["episodic"]
    assert ep["gating_enabled"] is True
    assert ep["retrieved_count"] == 2, f"should retrieve 2, got {ep['retrieved_count']}"
    assert ep["filtered_count"] == 1, f"should filter 1 (糖尿病), got {ep['filtered_count']}"
    assert ep["injected_count"] == 1, f"should inject 1 (脑膜炎), got {ep['injected_count']}"
    assert len(ep["gate_records"]) == 2, f"should have 2 gate records, got {len(ep['gate_records'])}"
    # injection totals
    inj = meta["injection"]
    assert inj["total_retrieved_count"] == 2
    assert inj["total_injected_count"] == 1
    assert inj["total_filtered_count"] == 1


def test_memory_gating_disabled_injects_all():
    """gating=False 时全部注入"""
    orch = _orch()
    orch.episodic = _FakeEpisodicMemory()
    orch.enable_memory_injection = True
    orch.enable_memory_gating = False
    q = UserQuery(user_id="u_nogate", text="头痛发烧颈强直可能是什么病？")
    ans = asyncio.run(orch.answer_async(q, session_id="s_nogate"))
    meta = ans.memory_meta
    ep = meta["episodic"]
    assert ep["gating_enabled"] is False
    assert ep["retrieved_count"] == ep["injected_count"]
    assert ep["filtered_count"] == 0


def test_memory_meta_scope_includes_user_id():
    """eval 隔离：每个 item 使用独立 user_id"""
    orch = _orch()
    q = UserQuery(user_id="eval_cmb_1_2", text="病史...")
    ans = asyncio.run(orch.answer_async(q, session_id="eval_cmb_1_2"))
    assert ans.memory_meta["scope"]["user_id"] == "eval_cmb_1_2"
    assert ans.memory_meta["scope"]["memory_key"] == "eval_cmb_1_2"


# ============================================================
# Memory Benchmark / Lifecycle (V10c)
# ============================================================

def test_memory_write_uses_final_diagnoses_and_can_be_flushed():
    """后台写入应可等待，且 Episode.diagnoses 来自最终回答而不是预取候选。"""
    from src.memory.episodic_memory import EpisodicMemory, MockEmbedder, SQLiteEpisodicBackend

    async def run():
        orch = _orch()
        memory = EpisodicMemory(SQLiteEpisodicBackend(":memory:"), MockEmbedder())
        orch.episodic = memory
        answer = await orch.answer_async(
            UserQuery(user_id="v10c_writer", text="头痛发烧颈强直可能是什么病？"),
            session_id="v10c_write_session",
        )
        await orch.flush_memory_writes()
        episodes = memory.backend.list_by_user("v10c_writer")
        assert len(episodes) == 1
        assert episodes[0].diagnoses == answer.diagnoses
        assert not orch._background_tasks
        assert not orch._background_errors

    asyncio.run(run())


def test_memory_written_in_one_session_is_visible_in_next_session():
    """同一用户的 Episodic Memory 应跨 session 可见。"""
    from src.memory.episodic_memory import EpisodicMemory, MockEmbedder, SQLiteEpisodicBackend

    async def run():
        orch = _orch()
        memory = EpisodicMemory(SQLiteEpisodicBackend(":memory:"), MockEmbedder())
        orch.episodic = memory
        await orch.answer_async(
            UserQuery(user_id="v10c_same_user", text="头痛发烧颈强直可能是什么病？"),
            session_id="v10c_setup_session",
        )
        await orch.flush_memory_writes()
        answer = await orch.answer_async(
            UserQuery(user_id="v10c_same_user", text="头痛再次发作，既往情况重要吗？"),
            session_id="v10c_test_session",
        )
        assert answer.memory_meta["episodic"]["retrieved_count"] >= 1
        assert answer.memory_meta["episodic"]["retrieved_episode_ids"]
        await orch.flush_memory_writes()

    asyncio.run(run())


def test_memory_flush_surfaces_background_write_error():
    """后台写入失败不能被 flush 静默吞掉。"""
    from src.memory.episodic_memory import EpisodicMemory, MockEmbedder, SQLiteEpisodicBackend

    async def run():
        orch = _orch()
        memory = EpisodicMemory(SQLiteEpisodicBackend(":memory:"), MockEmbedder())
        orch.episodic = memory

        def fail_write(*args, **kwargs):
            raise ValueError("injected write failure")

        memory.write = fail_write
        await orch.answer_async(
            UserQuery(user_id="v10c_fail_write", text="头痛发烧颈强直可能是什么病？"),
            session_id="v10c_fail_write_session",
        )
        try:
            await orch.flush_memory_writes()
            assert False, "flush should raise when a memory write fails"
        except RuntimeError as exc:
            assert "injected write failure" in str(exc)

    asyncio.run(run())


def test_memory_does_not_leak_across_users():
    """另一个用户不能检索到前一用户写入的 Episode。"""
    from src.memory.episodic_memory import EpisodicMemory, MockEmbedder, SQLiteEpisodicBackend

    async def run():
        orch = _orch()
        memory = EpisodicMemory(SQLiteEpisodicBackend(":memory:"), MockEmbedder())
        orch.episodic = memory
        await orch.answer_async(
            UserQuery(user_id="v10c_user_a", text="头痛发烧颈强直可能是什么病？"),
            session_id="v10c_user_a_session",
        )
        await orch.flush_memory_writes()
        answer = await orch.answer_async(
            UserQuery(user_id="v10c_user_b", text="头痛再次发作，既往情况重要吗？"),
            session_id="v10c_user_b_session",
        )
        assert answer.memory_meta["episodic"]["retrieved_count"] == 0
        await orch.flush_memory_writes()

    asyncio.run(run())


def test_memory_gate_records_preserve_episode_id():
    hints = [{"episode_id": "ep-1", "summary": "既往偏头痛", "diagnoses": ["偏头痛"]}]
    _, records = gate_episodic_hints("头痛", hints, enabled=True)
    assert records[0]["episode_id"] == "ep-1"


def test_critical_memory_bypasses_rule_gate_and_is_observable():
    """慢病关键事实应进入 PatientProfile，并与 episodic gate 路径分开观测。"""
    from src.memory.episodic_memory import Episode, EpisodicMemory, MockEmbedder, SQLiteEpisodicBackend

    async def run():
        orch = _orch()
        memory = EpisodicMemory(SQLiteEpisodicBackend(":memory:"), MockEmbedder())
        memory.write(Episode(
            user_id="v10c_critical", diagnoses=["糖尿病"], medications=["二甲双胍"],
            summary="患者长期患有糖尿病，服用二甲双胍",
        ))
        orch.episodic = memory
        orch.enable_memory_gating = True
        answer = await orch.answer_async(
            UserQuery(user_id="v10c_critical", text="最近肩膀酸痛怎么办？"),
            session_id="v10c_critical_session",
        )
        critical = answer.memory_meta["critical"]
        assert critical["bypasses_gate"] is True
        assert "糖尿病" in critical["chronic_diseases"]
        assert critical["injected_count"] == 1
        await orch.flush_memory_writes()

    asyncio.run(run())


def test_memory_metrics_recall_rank_and_leakage():
    from eval.memory_metrics import score_memory_result
    scores = score_memory_result({
        "expected_memory_ids": ["rel-1"],
        "forbidden_memory_ids": ["old-1"],
        "retrieved_episode_ids": ["noise-1", "rel-1"],
        "injected_episode_ids": ["rel-1", "old-1"],
        "answer": "记得青霉素过敏",
        "must_include": ["青霉素", "过敏"],
        "must_not_include": [],
    })
    assert scores["retrieval_recall"] == 1.0
    assert scores["injection_recall"] == 1.0
    assert scores["mrr"] == 0.5
    assert scores["forbidden_injected"] == 1.0
    assert scores["answer_required_recall"] == 1.0


def test_memory_eval_dataset_has_60_unique_scenarios():
    import json
    path = ROOT / "data" / "eval_memory_scenarios.jsonl"
    scenarios = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [scenario["scenario_id"] for scenario in scenarios]
    assert len(scenarios) == 60
    assert len(ids) == len(set(ids))
    assert {scenario["category"] for scenario in scenarios} == {
        "allergy", "chronic", "medication", "history", "temporal", "irrelevant",
    }


# ============================================================
# Planner 输出规范化 / 去重 guard
# ============================================================

from src.agents.planner import PlannerAgent, _coerce_int

def test_parse_depends_on_string_coerce():
    """LLM 输出 depends_on: ["step1", "2"] 应被规范化为 [1, 2]"""

    # _coerce_int 直接测试
    assert _coerce_int(1) == 1
    assert _coerce_int("1") == 1
    assert _coerce_int("step1") == 1
    assert _coerce_int("Step1") == 1
    assert _coerce_int(1.0) == 1
    assert _coerce_int("step_1") == 1
    assert _coerce_int("abc") is None
    assert _coerce_int(None) is None

    # PlannerAgent._sanitize_plan 测试
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)
    plan = Plan(
        thought="t", complexity=ComplexityLevel.MEDIUM,
        steps=[
            PlanStep(step_id=1, tool="ner", input="q", depends_on=[]),
            PlanStep(step_id=2, tool="kg_local_search", input={"query": "q"}, depends_on=["step1", "2"]),
        ],
    )
    sanitized = planner._sanitize_plan(plan)
    assert sanitized.steps[1].depends_on == [1]  # "step1"→1, "2"→2但自引用step2所以去掉2


def test_parse_depends_on_invalid_removed():
    """引用不存在 step_id、自依赖应被清除"""
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)
    plan = Plan(
        thought="t", complexity=ComplexityLevel.LOW,
        steps=[
            PlanStep(step_id=1, tool="ner", input="q", depends_on=[1]),   # 自依赖 → 清除
            PlanStep(step_id=2, tool="kg_local_search", input={"query": "q"}, depends_on=[99]),  # 不存在 → 清除
        ],
    )
    sanitized = planner._sanitize_plan(plan)
    assert sanitized.steps[0].depends_on == []  # 自依赖被清除
    assert sanitized.steps[1].depends_on == []  # 不存在引用被清除


def test_sanitize_duplicate_tools_removed():
    """5 个 kg_global_search + 2 个 ppr_reasoner 应被压到 1 + 1"""
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)
    plan = Plan(
        thought="t", complexity=ComplexityLevel.HIGH,
        steps=[
            PlanStep(step_id=1, tool="ner", input="q", depends_on=[]),
            PlanStep(step_id=2, tool="kg_global_search", input={"query": "q"}, depends_on=[1]),
            PlanStep(step_id=3, tool="kg_global_search", input={"query": "q2"}, depends_on=[1]),
            PlanStep(step_id=4, tool="kg_global_search", input={"query": "q3"}, depends_on=[1]),
            PlanStep(step_id=5, tool="kg_global_search", input={"query": "q4"}, depends_on=[1]),
            PlanStep(step_id=6, tool="ppr_reasoner", input={"seed_entities": ["x"]}, depends_on=[1]),
            PlanStep(step_id=7, tool="ppr_reasoner", input={"seed_entities": ["y"]}, depends_on=[1]),
            PlanStep(step_id=8, tool="kg_local_search", input={"query": "z"}, depends_on=[2]),
        ],
    )
    sanitized = planner._sanitize_plan(plan)

    # 只保留 1 个 kg_global_search + 1 个 ppr_reasoner + 1 个 ner + 1 个 kg_local_search
    assert len(sanitized.steps) == 4
    tool_names = [s.tool for s in sanitized.steps]
    assert tool_names.count("kg_global_search") == 1
    assert tool_names.count("ppr_reasoner") == 1
    assert tool_names.count("ner") == 1
    # 删除的 step 的依赖引用也应被清洗
    for s in sanitized.steps:
        for dep in s.depends_on:
            assert dep in {s2.step_id for s2 in sanitized.steps}


def test_sanitize_local_search_multiple_allowed():
    """2 个 kg_local_search（不同 input）应保留，超过 3 个截断"""
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)

    # 2 个 local_search → 全部保留
    plan2 = Plan(
        thought="t", complexity=ComplexityLevel.HIGH,
        steps=[
            PlanStep(step_id=1, tool="ner", input="q", depends_on=[]),
            PlanStep(step_id=2, tool="kg_local_search", input={"query": "entity_a"}, depends_on=[1]),
            PlanStep(step_id=3, tool="kg_local_search", input={"query": "entity_b"}, depends_on=[1]),
            PlanStep(step_id=4, tool="ppr_reasoner", input={"seed_entities": ["x"]}, depends_on=[1]),
        ],
    )
    s2 = planner._sanitize_plan(plan2)
    assert len(s2.steps) == 4
    assert [s.tool for s in s2.steps].count("kg_local_search") == 2

    # 4 个 local_search → 截断到 3
    plan4 = Plan(
        thought="t", complexity=ComplexityLevel.HIGH,
        steps=[
            PlanStep(step_id=1, tool="ner", input="q", depends_on=[]),
            PlanStep(step_id=2, tool="kg_local_search", input={"query": "a"}, depends_on=[1]),
            PlanStep(step_id=3, tool="kg_local_search", input={"query": "b"}, depends_on=[1]),
            PlanStep(step_id=4, tool="kg_local_search", input={"query": "c"}, depends_on=[1]),
            PlanStep(step_id=5, tool="kg_local_search", input={"query": "d"}, depends_on=[1]),
        ],
    )
    s4 = planner._sanitize_plan(plan4)
    assert [s.tool for s in s4.steps].count("kg_local_search") == 3


# ============================================================
# 规则层路由 guard
# ============================================================

def test_route_complexity_clinical_case_keeps_high():
    """临床病例文本（有现病史+体格检查等标记）应保持 high"""
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)
    query = "现病史：病人男，49岁。主诉：腹痛发热。体格检查：体温39℃。辅助检查：CT显示..."
    plan = Plan(
        thought="t", complexity=ComplexityLevel.HIGH,
        steps=[PlanStep(step_id=1, tool="ppr_reasoner", input={"seed_entities": ["腹痛"]}, depends_on=[])],
    )
    routed = planner._route_complexity(query, plan)
    assert routed.complexity == ComplexityLevel.HIGH
    # PPR step 应保留（临床病例需要 PPR）
    assert any(s.tool == "ppr_reasoner" for s in routed.steps)


def test_route_complexity_clinical_case_upgrades_medium():
    """LLM 误判 medium 的临床病例应被升级到 high"""
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)
    query = "现病史：病人发烧3天。体格检查：扁桃体肿大。辅助检查：血常规WBC升高。"
    plan = Plan(
        thought="t", complexity=ComplexityLevel.MEDIUM,
        steps=[PlanStep(step_id=1, tool="kg_global_search", input="发烧", depends_on=[])],
    )
    routed = planner._route_complexity(query, plan)
    assert routed.complexity == ComplexityLevel.HIGH  # 升级


def test_route_complexity_short_drug_query_downgrades():
    """短药品查询（<200 chars）应从 high 降级到 medium 并移除 PPR"""
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)
    query = "二甲双胍是什么药？有什么副作用？"  # ~20 chars, drug query
    plan = Plan(
        thought="t", complexity=ComplexityLevel.HIGH,
        steps=[
            PlanStep(step_id=1, tool="kg_global_search", input="二甲双胍", depends_on=[]),
            PlanStep(step_id=2, tool="ppr_reasoner", input={"seed_entities": ["二甲双胍"]}, depends_on=[]),
        ],
    )
    routed = planner._route_complexity(query, plan)
    assert routed.complexity == ComplexityLevel.MEDIUM  # 降级
    # PPR step 应被移除
    assert not any(s.tool == "ppr_reasoner" for s in routed.steps)
    # kg_global_search 应保留
    assert any(s.tool == "kg_global_search" for s in routed.steps)


def test_route_complexity_treatment_query_downgrades():
    """短治疗查询应从 high 降级"""
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)
    query = "高血压怎么治疗？吃什么药？"  # treatment query
    plan = Plan(
        thought="t", complexity=ComplexityLevel.HIGH,
        steps=[
            PlanStep(step_id=1, tool="kg_global_search", input="高血压治疗", depends_on=[]),
            PlanStep(step_id=2, tool="ppr_reasoner", input={"seed_entities": ["高血压"]}, depends_on=[]),
        ],
    )
    routed = planner._route_complexity(query, plan)
    assert routed.complexity == ComplexityLevel.MEDIUM
    assert not any(s.tool == "ppr_reasoner" for s in routed.steps)


def test_route_complexity_short_symptom_query_keeps_medium():
    """短症状查询（如'X的症状'）LLM 已判 medium 则保持"""
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)
    query = "偏头痛的典型症状有哪些？"
    plan = Plan(
        thought="t", complexity=ComplexityLevel.MEDIUM,
        steps=[PlanStep(step_id=1, tool="kg_global_search", input="偏头痛 症状", depends_on=[])],
    )
    routed = planner._route_complexity(query, plan)
    # 应保持 medium（不升级，因为不满足临床病例标记；不降级，因为已经是 medium）
    assert routed.complexity == ComplexityLevel.MEDIUM


def test_route_complexity_preserves_non_high():
    """LLM 判 low/medium 的非病例查询应保持原判"""
    planner = PlannerAgent(llm=None, tool_descriptions=[], enable_self_critique=False)
    # Low complexity: very short, non-medical or simple lookup
    query = "什么是感冒？"
    plan = Plan(
        thought="t", complexity=ComplexityLevel.LOW,
        steps=[PlanStep(step_id=1, tool="kg_local_search", input={"query": "感冒"}, depends_on=[])],
    )
    routed = planner._route_complexity(query, plan)
    assert routed.complexity == ComplexityLevel.LOW  # 不变


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
# V11 Memory dense/hybrid retrieval
# ============================================================

class _CountingEmbedder:
    model_id = "counting-v1"
    dim = 2

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [1.0, 0.0]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


def test_v11_embedding_metadata_roundtrip():
    from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
    from src.schemas import Episode

    memory = EpisodicMemory(SQLiteEpisodicBackend(":memory:"), _CountingEmbedder())
    episode = Episode(user_id="u", diagnoses=["糖尿病"], summary="慢性糖尿病病史")
    assert memory.write(episode)
    stored = memory.backend.list_by_user("u")[0]
    assert stored.embedding_model == "counting-v1"
    assert stored.embedding_dim == 2


def test_v11_incompatible_embedding_is_reindexed():
    from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
    from src.schemas import Episode

    embedder = _CountingEmbedder()
    backend = SQLiteEpisodicBackend(":memory:")
    backend.insert(Episode(
        episode_id="old", user_id="u", summary="旧向量", embedding=[0.2],
        embedding_model="legacy", embedding_dim=1, importance_score=0.8,
    ))
    memory = EpisodicMemory(backend, embedder, retrieval_mode="dense")
    result = memory.retrieve("u", "当前问题", top_k=1)
    stored = backend.list_by_user("u")[0]
    assert result[0].episode_id == "old"
    assert stored.embedding_model == "counting-v1" and stored.embedding_dim == 2
    assert "旧向量" in embedder.calls


def test_v11_dense_and_hybrid_have_explainable_ranking():
    from datetime import datetime, timedelta
    from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
    from src.schemas import Episode

    def build(mode):
        backend = SQLiteEpisodicBackend(":memory:")
        memory = EpisodicMemory(backend, _CountingEmbedder(), retrieval_mode=mode)
        for episode_id, days in (("old", 120), ("new", 0)):
            episode = Episode(
                episode_id=episode_id, user_id="u", summary="相同语义",
                timestamp=datetime.now() - timedelta(days=days), importance_score=0.8,
            )
            memory._set_embedding(episode, [1.0, 0.0])
            backend.insert(episode)
        return memory

    dense = build("dense").retrieve("u", "query", top_k=2)
    hybrid = build("hybrid").retrieve("u", "query", top_k=2)
    assert [ep.episode_id for ep in dense] == ["old", "new"]
    assert hybrid[0].episode_id == "new"
    assert set(hybrid[0].retrieval_components) == {
        "similarity", "importance", "time_decay", "frequency",
    }


def test_v11_bge_embedder_is_lazy():
    from src.memory.embedders import BGEEmbedder

    embedder = BGEEmbedder(device="cuda")
    assert embedder._model is None
    assert embedder.model_id == "BAAI/bge-m3"


def test_v11_factory_accepts_memory_retrieval_config():
    from src.factory import build_system

    agent, memory = build_system(
        backend="mock", memory_embedder="mock", memory_retrieval_mode="dense",
    )
    assert agent.episodic is memory
    assert memory.embedding_model_id == "semantic-mock-v1"
    assert memory.retrieval_mode == "dense"


def test_v11b_dataset_split_and_candidate_depth():
    from scripts.build_memory_reranking_dataset import build_reranking_scenarios

    rows = build_reranking_scenarios()
    assert len(rows) == 60
    assert sum(row["split"] == "dev" for row in rows) == 24
    assert sum(row["split"] == "test" for row in rows) == 36
    assert min(row["candidate_count"] for row in rows) >= 10
    for row in rows:
        target = [memory for memory in row["memories"] if memory["scope"] == "target"]
        assert len({memory["summary"] for memory in target}) == len(target)
        assert len(row.get("expected_memory_ids") or []) <= 1


class _ReverseReranker:
    model_id = "reverse-v1"

    def score(self, query, documents):
        docs = list(documents)
        return [float(index) for index in range(len(docs))]


def test_v11b_reranker_reorders_and_exports_scores():
    from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
    from src.schemas import Episode

    backend = SQLiteEpisodicBackend(":memory:")
    memory = EpisodicMemory(
        backend, _CountingEmbedder(), retrieval_mode="dense",
        reranker=_ReverseReranker(), reranker_candidate_k=3,
    )
    for episode_id in ("first", "second", "third"):
        episode = Episode(episode_id=episode_id, user_id="u", summary=episode_id, importance_score=0.8)
        memory._set_embedding(episode, [1.0, 0.0])
        backend.insert(episode)
    result = memory.retrieve("u", "query", top_k=2)
    assert [episode.episode_id for episode in result] == ["third", "second"]
    assert result[0].retrieval_components["reranker_score"] == 2.0
    assert result[0].retrieval_components["base_score"] == 1.0


def test_v11b_reranker_threshold_can_abstain():
    from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
    from src.schemas import Episode

    backend = SQLiteEpisodicBackend(":memory:")
    memory = EpisodicMemory(
        backend, _CountingEmbedder(), retrieval_mode="dense",
        reranker=_ReverseReranker(), reranker_candidate_k=2, reranker_threshold=5.0,
    )
    episode = Episode(episode_id="only", user_id="u", summary="only", importance_score=0.8)
    memory._set_embedding(episode, [1.0, 0.0]); backend.insert(episode)
    assert memory.retrieve("u", "query", top_k=1) == []


def test_v11b_rejects_incomplete_reranker_scores():
    from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
    from src.schemas import Episode

    class ShortReranker:
        model_id = "short-v1"

        def score(self, query, documents):
            return []

    backend = SQLiteEpisodicBackend(":memory:")
    memory = EpisodicMemory(backend, _CountingEmbedder(), reranker=ShortReranker())
    episode = Episode(episode_id="only", user_id="u", summary="only", importance_score=0.8)
    memory._set_embedding(episode, [1.0, 0.0]); backend.insert(episode)
    try:
        memory.retrieve("u", "query", top_k=1)
        assert False, "incomplete reranker scores must fail"
    except ValueError as exc:
        assert "returned 0 scores for 1 candidates" in str(exc)


def test_v11b_factory_reranker_is_optional():
    from src.factory import build_system

    _, plain = build_system(backend="mock")
    _, reranked = build_system(backend="mock", memory_reranker="identity")
    assert plain.reranker is None
    assert reranked.reranker_model_id == "identity-v1"


def test_v11b_factory_exposes_calibrated_hybrid_weights():
    from src.factory import build_system

    weights = (0.8, 0.1, 0.2, 0.0)
    agent, memory = build_system(backend="mock", memory_retrieval_weights=weights)
    assert memory.retrieval_weights == weights
    assert agent.episodic.retrieval_weights == weights


def test_v11b_eval_uses_actual_top5_injection_cutoff():
    from eval.run_memory_reranking_eval import rank_row

    candidates = []
    for index, episode_id in enumerate(("a", "b", "c", "old", "target")):
        candidates.append({
            "episode_id": episode_id,
            "summary": episode_id,
            "similarity": 1.0 - index * 0.1,
            "importance": 0.8,
            "time_decay": 1.0,
            "frequency": 0.0,
            "reranker_score": 1.0 - index * 0.1,
        })
    ranked = rank_row({
        "scenario_id": "temporal_test", "category": "temporal", "split": "test",
        "expected_memory_ids": ["target"], "forbidden_memory_ids": ["old"],
        "candidates": candidates, "dense_ms": 2.0, "reranker_ms": 9.0,
    }, "dense", (1.0, 0.0, 0.0, 0.0), candidate_k=5)
    assert ranked["forbidden_at_1"] == 0.0
    assert ranked["forbidden_at_5"] == 1.0
    assert ranked["reranker_ms"] == 0.0


# ============================================================
# V11c Temporal Memory lifecycle
# ============================================================

def test_v11c_legacy_sqlite_schema_migrates():
    import sqlite3
    import tempfile
    from pathlib import Path
    from src.memory.episodic_memory import SQLiteEpisodicBackend

    with tempfile.NamedTemporaryFile(suffix=".db") as handle:
        conn = sqlite3.connect(handle.name)
        conn.execute("""
            CREATE TABLE episodes (
                episode_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, timestamp TEXT,
                episode_type TEXT, diagnoses TEXT, medications TEXT, symptoms TEXT,
                summary TEXT, importance_score REAL, access_count INTEGER DEFAULT 0,
                last_accessed TEXT, embedding TEXT, embedding_model TEXT DEFAULT '',
                embedding_dim INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
        backend = SQLiteEpisodicBackend(handle.name)
        columns = {row[1] for row in backend.conn.execute("PRAGMA table_info(episodes)")}
        assert {"status", "provenance", "superseded_by"} <= columns
        indexes = {row[1] for row in backend.conn.execute("PRAGMA index_list(episodes)")}
        assert "idx_user_status" in indexes


def test_v11c_supersede_is_atomic_and_active_only():
    from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
    from src.schemas import Episode

    backend = SQLiteEpisodicBackend(":memory:")
    memory = EpisodicMemory(backend, _CountingEmbedder())
    old = Episode(
        episode_id="old", user_id="u", summary="正在服用二甲双胍",
        medications=["二甲双胍"], importance_score=0.8,
    )
    memory._set_embedding(old, [1.0, 0.0])
    backend.insert(old)
    new = Episode(
        episode_id="new", user_id="u", summary="医生已停用二甲双胍",
        medications=["二甲双胍"], provenance={"source_type": "clinician_note"},
    )
    assert memory.write_superseding(new, ["old"])
    audit = {episode.episode_id: episode for episode in backend.list_by_user("u")}
    assert audit["old"].status == "superseded"
    assert audit["old"].superseded_by == "new"
    assert audit["new"].status == "active"
    assert [episode.episode_id for episode in memory.retrieve("u", "二甲双胍", 5)] == ["new"]
    assert memory.lifecycle_counts("u") == {"active": 1, "superseded": 1, "retracted": 0}


def test_v11c_cross_user_supersede_rolls_back():
    from src.memory.episodic_memory import SQLiteEpisodicBackend
    from src.schemas import Episode

    backend = SQLiteEpisodicBackend(":memory:")
    backend.insert(Episode(episode_id="old", user_id="other", summary="old"))
    try:
        backend.insert_superseding(
            Episode(episode_id="new", user_id="target", summary="new"), ["old"]
        )
        assert False, "cross-user supersede must fail"
    except ValueError as exc:
        assert "another user" in str(exc)
    assert backend.list_by_user("target") == []
    assert backend.list_by_user("other")[0].status == "active"


def test_v11c_retracted_fact_is_auditable_but_not_retrieved():
    from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
    from src.schemas import Episode

    backend = SQLiteEpisodicBackend(":memory:")
    memory = EpisodicMemory(backend, _CountingEmbedder())
    episode = Episode(episode_id="bad", user_id="u", summary="录入错误", importance_score=0.8)
    memory._set_embedding(episode, [1.0, 0.0])
    backend.insert(episode)
    memory.retract("u", "bad")
    assert memory.retrieve("u", "录入错误", 5) == []
    assert backend.list_by_user("u")[0].status == "retracted"


def test_v11c_critical_facts_ignore_superseded_and_negated_allergy():
    from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
    from src.schemas import Episode

    backend = SQLiteEpisodicBackend(":memory:")
    memory = EpisodicMemory(backend, _CountingEmbedder())
    old = Episode(episode_id="old", user_id="u", summary="青霉素过敏")
    memory._set_embedding(old, [1.0, 0.0])
    backend.insert(old)
    new = Episode(
        episode_id="new", user_id="u", summary="复查确认青霉素不过敏",
        medications=["青霉素"],
    )
    assert memory.write_superseding(new, ["old"])
    assert memory.retrieve_critical_facts("u")["allergies"] == []


def test_v11c_orchestrator_exports_active_lifecycle_meta():
    from src.factory import build_system
    from src.schemas import Episode, UserQuery

    agent, memory = build_system(backend="mock")
    agent.enable_prefetch = False
    old = Episode(
        episode_id="old", user_id="u", summary="正在服用二甲双胍",
        status="superseded", superseded_by="new", importance_score=0.8,
    )
    new = Episode(
        episode_id="new", user_id="u", summary="医生已停用二甲双胍",
        provenance={"source_type": "clinician_note"}, importance_score=0.8,
    )
    memory._set_embedding(old, memory.embedder.embed(old.summary))
    memory._set_embedding(new, memory.embedder.embed(new.summary))
    memory.backend.insert(old)
    memory.backend.insert(new)

    async def run():
        answer = await agent.answer_async(
            UserQuery(user_id="u", text="我还在服用二甲双胍吗？"),
            session_id="v11c-meta",
        )
        await agent.flush_memory_writes()
        return answer

    answer = asyncio.run(run())
    episodic = answer.memory_meta["episodic"]
    assert episodic["active_only"] is True
    assert "old" not in episodic["retrieved_episode_ids"]
    assert "new" in episodic["retrieved_episode_ids"]
    assert episodic["lifecycle_counts"]["superseded"] == 1
    new_record = next(record for record in episodic["retrieval_records"] if record["episode_id"] == "new")
    assert new_record["status"] == "active"
    assert new_record["provenance"]["source_type"] == "clinician_note"


def test_v11c_memory_eval_temporal_config_supersedes_old_fact():
    from eval.run_memory_eval import seed_scenario
    from src.memory.episodic_memory import EpisodicMemory, MockEmbedder, SQLiteEpisodicBackend

    scenario = {
        "scenario_id": "temporal_unit", "category": "temporal",
        "expected_memory_ids": ["new"], "forbidden_memory_ids": ["old"],
        "memories": [
            {"episode_id": "old", "summary": "正在服药", "scope": "target"},
            {"episode_id": "new", "summary": "已经停药", "scope": "target"},
        ],
    }
    memory = EpisodicMemory(SQLiteEpisodicBackend(":memory:"), MockEmbedder())
    user_id, _ = seed_scenario(memory, scenario, temporal_lifecycle=True)
    audit = {episode.episode_id: episode for episode in memory.backend.list_by_user(user_id)}
    assert audit["old"].status == "superseded"
    assert audit["old"].superseded_by == "new"
    assert audit["new"].status == "active"


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
