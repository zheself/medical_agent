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
from typing import Any, Dict, List, Optional

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

# 任务
请严格按以下结构输出 JSON:
{{
  "analysis": "错误原因深度分析",
  "corrected_answer": {{
    "content": "修正后的完整回答",
    "recommended_drugs": [],
    "possible_diagnoses": [],
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
        # 确保 corrected_answer 至少有 content
        parsed["corrected_answer"].setdefault("content", "")
        parsed["corrected_answer"].setdefault("citations", [])
        parsed["root_cause"].setdefault("type", "other")
        parsed["root_cause"].setdefault("detail", {})
        return parsed


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
            # L1 失败 → 高复杂度直接 L3，否则触发 replan
            if complexity == "high":
                l3_output = self.l3.reflect_and_correct(
                    query, draft_answer, l1_result=l1_result
                )
                chain.append(l3_output["verify_result"])
                return {
                    "final_answer": l3_output["corrected_answer"],
                    "verify_chain": chain,
                    "level_reached": "L3",
                    "total_elapsed_ms": (time.time() - start) * 1000,
                    "reflexion_record": l3_output.get("reflexion_record"),
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
        
        # L2 失败 → 高复杂度上 L3，中等复杂度触发 replan
        if complexity == "high":
            l3_output = self.l3.reflect_and_correct(
                query, draft_answer, l1_result=l1_result, l2_result=l2_result
            )
            chain.append(l3_output["verify_result"])
            return {
                "final_answer": l3_output["corrected_answer"],
                "verify_chain": chain,
                "level_reached": "L3",
                "total_elapsed_ms": (time.time() - start) * 1000,
                "reflexion_record": l3_output.get("reflexion_record"),
            }
        else:
            return {
                "final_answer": draft_answer,
                "verify_chain": chain,
                "level_reached": "L2",
                "needs_replan": True,
                "total_elapsed_ms": (time.time() - start) * 1000,
            }
