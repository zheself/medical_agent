"""
verifiers/l3_reflexion.py — L3 完整反思

工作流程:
1. 接收 L1/L2 的失败信号
2. 从 Semantic Memory 检索相似失败案例作 few-shot
3. 让主 LLM (7B) 重新推理并修正
4. 提取 root cause（哪类错误？）
5. 写入 Semantic Memory，供后续 self-improvement

仅在 L1/L2 失败时触发，全量流量中触发率 ~5%。

实现状态: ✅ 接口完整 / 🟡 LLM 调用部分需替换为真实模型
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import ReflexionRecord, VerifierLevel, VerifyResult


L3_REFLEXION_PROMPT = """你之前给出的回答存在问题，请反思并修正。

# 用户问题
{query}

# 你的初版回答
{draft_answer}

# L1 规则校验错误（如有）
{l1_errors}

# L2 语义校验分数（如有）
{l2_scores}

# 历史类似错误案例（参考，避免重复犯错）
{similar_failures}

# 修正规则（必须遵守）
1. **保留优先**：如果原回答的 possible_diagnoses 中已有正确或合理的诊断，必须保留，不要删除或替换。
2. **修正而非重写**：只修改有问题的部分（错误关系、遗漏诊断、错误事实）。原回答中正确的部分必须保留。
3. **诊断候选合并输出**：你的 possible_diagnoses 应该是"原候选中的正确部分 + 新补充的鉴别诊断"的合并结果。
4. **按置信度排序**：possible_diagnoses 输出 3-5 个候选，按置信度从高到低排列。第一位是最可能的主要诊断。
5. **证据不足时保留原诊断**：如果无法确定原回答的诊断是否错误，保留原候选并说明不确定性，不要强行替换。

# 任务
请严格按以下结构输出 JSON:
{{
  "analysis": "错误原因深度分析",
  "corrected_answer": {{
    "content": "修正后的完整回答",
    "recommended_drugs": [],
    "possible_diagnoses": ["保留正确的原候选 + 新补充的鉴别诊断（3-5个，按置信度排序）"],
    "citations": []
  }},
  "root_cause": {{
    "type": "missing_relation | missing_constraint | factual_error | other",
    "detail": {{...具体描述，便于沉淀到 KG/规则}}
  }}
}}
"""


class L3Reflexion:
    """
    完整反思与修正
    
    与 L1/L2 的关键区别:
    - 调用完整 7B 模型（最贵）
    - 但仅在前两级失败时触发
    - 会写入 Semantic Memory，形成自我进化
    """
    
    def __init__(
        self,
        llm_backend,                            # 主 LLM（同 Planner）
        semantic_memory=None,                    # Semantic Memory，可选
    ):
        self.llm = llm_backend
        self.semantic_memory = semantic_memory
    
    def reflect_and_correct(
        self,
        query: str,
        draft_answer: Dict[str, Any],
        l1_result: Optional[VerifyResult] = None,
        l2_result: Optional[VerifyResult] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "corrected_answer": {...},
                "verify_result": VerifyResult,
                "reflexion_record": ReflexionRecord
            }
        """
        start = time.time()
        
        # 1. 从 Semantic Memory 检索相似失败案例
        similar_failures = []
        if self.semantic_memory:
            similar_failures = self.semantic_memory.get_few_shot_examples(query, top_k=3)
        
        # 2. 构造 prompt
        prompt = L3_REFLEXION_PROMPT.format(
            query=query,
            draft_answer=json.dumps(draft_answer, ensure_ascii=False)[:1500],
            l1_errors=l1_result.errors if l1_result else "无",
            l2_scores=l2_result.scores if l2_result else "无",
            similar_failures=self._format_failures(similar_failures),
        )
        
        # 3. 调用主模型生成修正
        response = self.llm.generate(prompt, max_tokens=2048)
        parsed = self._parse_reflexion(response)
        
        # 4. 构造记录，准备写入 Semantic Memory
        record = ReflexionRecord(
            query=query,
            wrong_answer=draft_answer.get("content", ""),
            errors=(l1_result.errors if l1_result else []) + (l2_result.errors if l2_result else []),
            correction=parsed["corrected_answer"].get("content", ""),
            root_cause_type=parsed["root_cause"].get("type", "other"),
            root_cause_detail=parsed["root_cause"].get("detail", {}),
            timestamp=datetime.now(),
        )
        
        # 5. 写入 Semantic Memory（自我进化）
        if self.semantic_memory:
            sm_report = self.semantic_memory.write_from_reflexion(record)
        else:
            sm_report = {}
        
        elapsed_ms = (time.time() - start) * 1000
        
        verify_result = VerifyResult(
            passed=True,  # L3 总是产出"修正后版本"
            level=VerifierLevel.L3,
            errors=[],
            elapsed_ms=elapsed_ms,
        )
        
        return {
            "corrected_answer": parsed["corrected_answer"],
            "verify_result": verify_result,
            "reflexion_record": record,
            "semantic_memory_report": sm_report,
        }
    
    @staticmethod
    def _format_failures(failures: List[Dict]) -> str:
        if not failures:
            return "（无相似历史案例）"
        lines = []
        for i, f in enumerate(failures, 1):
            lines.append(
                f"案例 {i}:\n"
                f"  错误回答: {f.get('wrong_answer', '')[:100]}\n"
                f"  修正: {f.get('correction', '')[:200]}\n"
            )
        return "\n".join(lines)
    
    @staticmethod
    def _parse_reflexion(response: str) -> Dict:
        """从 LLM 输出中提取 JSON，并填充缺失字段"""
        import re
        
        default = {
            "analysis": "fallback",
            "corrected_answer": {"content": response[:500] if response else "", "citations": []},
            "root_cause": {"type": "other", "detail": {}},
        }
        
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if not match:
            return default
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return default
        
        # 填充缺失字段
        if "corrected_answer" not in parsed or not isinstance(parsed["corrected_answer"], dict):
            parsed["corrected_answer"] = default["corrected_answer"]
        if "root_cause" not in parsed or not isinstance(parsed["root_cause"], dict):
            parsed["root_cause"] = default["root_cause"]
        # 确保 corrected_answer 字段类型安全
        parsed["corrected_answer"].setdefault("content", "")
        if not isinstance(parsed["corrected_answer"].get("content"), str):
            parsed["corrected_answer"]["content"] = str(parsed["corrected_answer"]["content"])
        # Guard: citations 必须是 list of dict，LLM 可能输出 string list 或空
        raw_cites = parsed["corrected_answer"].setdefault("citations", [])
        if not isinstance(raw_cites, list):
            raw_cites = []
        parsed["corrected_answer"]["citations"] = [
            c if isinstance(c, dict) else {"type": "kg_inferred", "source": str(c)[:200], "rel": "", "target": ""}
            for c in raw_cites
        ]
        parsed["root_cause"].setdefault("type", "other")
        parsed["root_cause"].setdefault("detail", {})
        return parsed


# ============================================================
# L3 Diagnosis Merge Guard
# ============================================================

def merge_l3_diagnoses(
    original_diagnoses: List[str],
    l3_diagnoses: List[str],
    l3_content: str,
    max_candidates: int = 5,
) -> Tuple[List[str], str]:
    """
    合并 L2 原始诊断与 L3 反思诊断。

    防止 L3 "重写而非修正"——当 L2 已有正确诊断时，
    L3 不应将其覆盖。策略：
    1. 原诊断为无 → 使用 L3 诊断
    2. 原诊断非空 → 保留原 Top-3，追加 L3 新候选，去重截断
    3. L3 无诊断 → 保留原诊断

    Returns:
        (merged_diagnoses, updated_content)
    """
    # Guard: 类型安全
    if not isinstance(original_diagnoses, list):
        original_diagnoses = []
    if not isinstance(l3_diagnoses, list):
        l3_diagnoses = []

    # Case 1: L3 无候选 → 保留原诊断
    if not l3_diagnoses:
        return original_diagnoses[:max_candidates], l3_content

    # Case 2: 原诊断为无 → 使用 L3 候选
    if not original_diagnoses:
        merged = l3_diagnoses[:max_candidates]
        source_label = "反思修正"
    else:
        # Case 3: 两者都有 → merge（保留原 Top-3，补充 L3 新候选）
        merged = list(original_diagnoses[:3])
        for d in l3_diagnoses:
            if d not in merged:
                merged.append(d)
            if len(merged) >= max_candidates:
                break
        source_label = "原始候选 + 反思修正"

    # 追加合并候选摘要到 content（不重写正文）
    merged_str = "、".join(merged)
    summary_line = f"\n\n**综合候选诊断**（{source_label}）：{merged_str}"
    return merged, l3_content + summary_line


# ============================================================
# L3 Trigger Guard
# ============================================================

# L1 安全标签：这些错误绝不能静默放行
_L1_SAFETY_TAGS = {"ALLERGY", "AGE", "GENDER", "EMERGENCY", "DYNAMIC"}

# trigger reason 常量
TRIGGER_EMPTY_DIAGNOSES = "empty_diagnoses"
TRIGGER_L1_SAFETY = "l1_safety_error"
TRIGGER_L1_CITATION = "l1_citation_failure"
TRIGGER_LOW_L2_SCORE = "low_l2_score"
SKIP_ENOUGH_CANDIDATES = "enough_candidates_skip"

_L2_SCORE_KEYS = ("faithfulness", "relevance", "factuality")


def _numeric_l2_scores(l2_result: Optional[VerifyResult]) -> List[float]:
    """从 L2 VerifyResult 中提取纯数值分数。

    L2 scores 形如 {"faithfulness": 0.85, "relevance": 0.88, "factuality": 0.82, "issues": [...]}，
    必须过滤非数值字段（如 issues list），否则 min() 会触发 TypeError。
    """
    if l2_result is None:
        return []
    scores = getattr(l2_result, "scores", None) or {}
    if not isinstance(scores, dict):
        return []
    return [
        v for k in _L2_SCORE_KEYS
        if (v := scores.get(k)) is not None and isinstance(v, (int, float))
    ]


def _should_trigger_l3(
    draft_answer: Dict[str, Any],
    l1_result: Optional[VerifyResult],
    l2_result: Optional[VerifyResult],
) -> Tuple[bool, str]:
    """
    判断是否应触发 L3 reflexion。

    决策顺序：
    1. 空预测 → 强制触发（empty rescue）
    2. L1 安全错误 → 强制触发（安全不能放行）
    3. L1 citation-only + 候选 ≤ 2 → 触发
    4. L2 分数 < 0.5 → 触发
    5. 候选 ≥ 3 且 L2 分数 ≥ 0.6 → 跳过
    6. 默认 → 触发（保守）

    Returns:
        (should_trigger, reason)
    """
    diagnoses = draft_answer.get("possible_diagnoses", [])
    if not isinstance(diagnoses, list):
        diagnoses = []
    n_candidates = len(diagnoses)

    # 1. 空预测 → 强制 rescue
    if n_candidates == 0:
        return True, TRIGGER_EMPTY_DIAGNOSES

    # 2. L1 安全错误 → 强制触发
    if l1_result is not None and not l1_result.passed:
        for err in l1_result.errors:
            for tag in _L1_SAFETY_TAGS:
                if tag in err:
                    return True, TRIGGER_L1_SAFETY

    # 3. L1 citation-only + 候选少 → 触发
    if l1_result is not None and not l1_result.passed:
        has_safety = any(
            any(tag in err for tag in _L1_SAFETY_TAGS)
            for err in l1_result.errors
        )
        if not has_safety and n_candidates <= 2:
            return True, TRIGGER_L1_CITATION

    # 4. L2 分数极低 → 触发（仅用数值分数，避开 issues 等非数值字段）
    numeric_scores = _numeric_l2_scores(l2_result)
    if numeric_scores and min(numeric_scores) < 0.5:
        return True, TRIGGER_LOW_L2_SCORE

    # 5. 候选充足 + L2 不差 → 跳过
    if n_candidates >= 3:
        if numeric_scores and min(numeric_scores) >= 0.6:
            return False, SKIP_ENOUGH_CANDIDATES
        # L2 不存在但候选够 → 也跳过
        if not numeric_scores and l2_result is None:
            return False, SKIP_ENOUGH_CANDIDATES

    # 6. 默认 → 触发（保守）
    return True, "default_trigger"


# ============================================================
# 三级反思的统一调度器
# ============================================================

class GradedVerifierOrchestrator:
    """
    分级反思的总调度器
    
    根据 query 复杂度决定走多深的反思:
    - LOW: 仅 L1
    - MEDIUM: L1 + L2，L2 失败转 replan（不走 L3）
    - HIGH: L1 + L2 + L3
    
    用法:
        orchestrator = GradedVerifierOrchestrator(l1=..., l2=..., l3=...)
        result = orchestrator.verify(query, draft, evidence, complexity, patient_profile)
    """
    
    def __init__(self, l1_verifier, l2_verifier, l3_reflexion, max_level: str = "L3"):
        self.l1 = l1_verifier
        self.l2 = l2_verifier
        self.l3 = l3_reflexion
        # 限制最高反思级别（消融用）: "L1" | "L2" | "L3"
        self.max_level = max_level
    
    def verify(
        self,
        query: str,
        draft_answer: Dict[str, Any],
        evidence: List[Dict],
        complexity: str = "medium",
        patient_profile=None,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "final_answer": dict,
                "verify_chain": [VerifyResult, ...],  # 每级的结果
                "level_reached": "L1" | "L2" | "L3",
                "total_elapsed_ms": float
            }
        """
        chain = []
        start = time.time()

        # 按 max_level 裁剪复杂度：
        #   max_level="L1" -> 强制 low（仅 L1）
        #   max_level="L2" -> high 降为 medium（最高到 L2/replan，不进 L3）
        if self.max_level == "L1":
            complexity = "low"
        elif self.max_level == "L2" and complexity == "high":
            complexity = "medium"

        # L1 永远跑
        l1_result = self.l1.verify(draft_answer, patient_profile)
        chain.append(l1_result)
        
        if not l1_result.passed:
            # L1 失败 → 高复杂度走 trigger guard，否则触发 replan
            if complexity == "high":
                should_trigger, trigger_reason = _should_trigger_l3(
                    draft_answer, l1_result, None
                )
                if should_trigger:
                    l3_output = self.l3.reflect_and_correct(
                        query, draft_answer, l1_result=l1_result
                    )
                    chain.append(l3_output["verify_result"])
                    # Merge guard: 防止 L3 覆盖 L2 正确诊断
                    original_diag = draft_answer.get("possible_diagnoses", [])
                    l3_answer = l3_output["corrected_answer"]
                    if isinstance(l3_answer, dict):
                        merged_diag, merged_content = merge_l3_diagnoses(
                            original_diag,
                            l3_answer.get("possible_diagnoses", []),
                            l3_answer.get("content", ""),
                        )
                        l3_answer["possible_diagnoses"] = merged_diag
                        l3_answer["content"] = merged_content
                    return {
                        "final_answer": l3_answer,
                        "verify_chain": chain,
                        "level_reached": "L3",
                        "total_elapsed_ms": (time.time() - start) * 1000,
                        "reflexion_record": l3_output.get("reflexion_record"),
                        "trigger_reason": trigger_reason,
                    }
                else:
                    # L1 fail 但不触发 L3 → replan（不能静默返回 draft）
                    return {
                        "final_answer": draft_answer,
                        "verify_chain": chain,
                        "level_reached": "L1",
                        "needs_replan": True,
                        "l3_skip_reason": trigger_reason,
                        "total_elapsed_ms": (time.time() - start) * 1000,
                    }
            else:
                # 中低复杂度：返回失败，让上层 replan
                return {
                    "final_answer": draft_answer,
                    "verify_chain": chain,
                    "level_reached": "L1",
                    "needs_replan": True,
                    "total_elapsed_ms": (time.time() - start) * 1000,
                }
        
        # 低复杂度且 L1 通过 → 不再校验
        if complexity == "low":
            return {
                "final_answer": draft_answer,
                "verify_chain": chain,
                "level_reached": "L1",
                "total_elapsed_ms": (time.time() - start) * 1000,
            }
        
        # 中高复杂度：跑 L2
        l2_result = self.l2.verify(query, draft_answer, evidence)
        chain.append(l2_result)
        
        if l2_result.passed:
            return {
                "final_answer": draft_answer,
                "verify_chain": chain,
                "level_reached": "L2",
                "total_elapsed_ms": (time.time() - start) * 1000,
            }
        
        # L2 失败 → 高复杂度走 trigger guard，中等复杂度触发 replan
        if complexity == "high":
            should_trigger, trigger_reason = _should_trigger_l3(
                draft_answer, l1_result, l2_result
            )
            if should_trigger:
                l3_output = self.l3.reflect_and_correct(
                    query, draft_answer, l1_result=l1_result, l2_result=l2_result
                )
                chain.append(l3_output["verify_result"])
                # Merge guard: 防止 L3 覆盖 L2 正确诊断
                original_diag = draft_answer.get("possible_diagnoses", [])
                l3_answer = l3_output["corrected_answer"]
                if isinstance(l3_answer, dict):
                    merged_diag, merged_content = merge_l3_diagnoses(
                        original_diag,
                        l3_answer.get("possible_diagnoses", []),
                        l3_answer.get("content", ""),
                    )
                    l3_answer["possible_diagnoses"] = merged_diag
                    l3_answer["content"] = merged_content
                return {
                    "final_answer": l3_answer,
                    "verify_chain": chain,
                    "level_reached": "L3",
                    "total_elapsed_ms": (time.time() - start) * 1000,
                    "reflexion_record": l3_output.get("reflexion_record"),
                    "trigger_reason": trigger_reason,
                }
            else:
                # L2 fail 但跳过 L3 → 返回 draft（候选够 + 分数不差）
                return {
                    "final_answer": draft_answer,
                    "verify_chain": chain,
                    "level_reached": "L2",
                    "l3_skip_reason": trigger_reason,
                    "total_elapsed_ms": (time.time() - start) * 1000,
                }
        else:
            return {
                "final_answer": draft_answer,
                "verify_chain": chain,
                "level_reached": "L2",
                "needs_replan": True,
                "total_elapsed_ms": (time.time() - start) * 1000,
            }
