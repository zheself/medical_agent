"""
verifiers/l2_model_verifier.py — L2 小模型校验器
verifiers/l3_reflexion.py — L3 完整反思（同文件，便于查看）

L2: 用蒸馏的 1.5B 小模型做语义级校验
    - 维度: faithfulness / relevance / factuality
    - 任一低于阈值则升级到 L3

L3: 完整反思
    - 检索历史相似失败案例作 few-shot
    - 让大模型修正
    - 修正结果写入 Semantic Memory

实现状态: 🟡 接口完整，LLM 调用部分为 Mock
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from ..schemas import ReflexionRecord, VerifierLevel, VerifyResult


# ============================================================
# L2 Small Model Verifier
# ============================================================

L2_VERIFIER_PROMPT = """你是医疗回答校验员。判断回答的质量。

用户问题: {query}
检索到的证据 (top-3):
{evidence}

回答内容: {answer}

请按以下维度评分（0-1）:
1. faithfulness: 回答是否忠于证据，没有编造证据外的内容?
2. relevance: 回答是否切题，回应了用户问题?
3. factuality: 回答中的医学事实是否准确?

输出严格的 JSON:
{{"faithfulness": <0-1>, "relevance": <0-1>, "factuality": <0-1>, "issues": ["问题描述"]}}
"""


class SmallModelLLM:
    """L2 用的小模型接口（1.5B），蒸馏自 7B 或 GPT-4"""
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        raise NotImplementedError


class MockSmallModel(SmallModelLLM):
    """Mock：根据简单规则返回评分"""
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        # 这是 mock 实现：实际场景这里调用蒸馏后的 1.5B 模型
        # 简单根据 prompt 长度估算 quality
        answer_part = prompt.lower()
        # 检查是否有明显问题信号
        issues = []
        scores = {"faithfulness": 0.85, "relevance": 0.88, "factuality": 0.82}
        if "未知" in answer_part or "不确定" in answer_part:
            scores["faithfulness"] = 0.6
            issues.append("回答中存在大量不确定表述")
        return json.dumps({**scores, "issues": issues}, ensure_ascii=False)


class L2ModelVerifier:
    """L2 校验器"""
    
    def __init__(
        self,
        small_model: SmallModelLLM,
        threshold: float = 0.7,
    ):
        self.model = small_model
        self.threshold = threshold
    
    def verify(
        self,
        query: str,
        draft_answer: Dict[str, Any],
        evidence: List[Dict],
    ) -> VerifyResult:
        start = time.time()
        
        # 只取 top-3 证据，节省 token
        evidence_text = "\n".join(
            f"- {json.dumps(e, ensure_ascii=False)[:200]}"
            for e in evidence[:3]
        )
        
        prompt = L2_VERIFIER_PROMPT.format(
            query=query,
            evidence=evidence_text,
            answer=draft_answer.get("content", "")[:1000],
        )
        
        try:
            response = self.model.generate(prompt, max_tokens=200)
            scores = self._parse_scores(response)
        except Exception as e:
            return VerifyResult(
                passed=False,
                level=VerifierLevel.L2,
                errors=[f"L2 model error: {e}"],
                escalate=True,
                elapsed_ms=(time.time() - start) * 1000,
            )
        
        min_score = min(scores.get(k, 0) for k in ["faithfulness", "relevance", "factuality"])
        passed = min_score >= self.threshold
        
        return VerifyResult(
            passed=passed,
            level=VerifierLevel.L2,
            errors=scores.get("issues", []),
            scores=scores,
            escalate=not passed,
            elapsed_ms=(time.time() - start) * 1000,
        )
    
    @staticmethod
    def _parse_scores(response: str) -> Dict[str, Any]:
        import re
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return {"faithfulness": 0, "relevance": 0, "factuality": 0, "issues": ["parse_failed"]}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"faithfulness": 0, "relevance": 0, "factuality": 0, "issues": ["json_error"]}
