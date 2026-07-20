"""
memory/memory_gate.py — Long-term Memory Relevance Gating

纯规则 relevance scoring，不依赖模型、不依赖 GPU。
只对 episodic/semantic long-term memory 做 gating，不动 working memory。

用法:
    from src.memory.memory_gate import score_memory_relevance, gate_episodic_hints

    decision = score_memory_relevance("头痛发烧", {"summary": "...", "diagnoses": [...]})
    if decision.keep:
        ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


# ============================================================
# Scoring
# ============================================================

@dataclass
class MemoryGateDecision:
    keep: bool
    score: float
    reason: str
    matched_terms: List[str] = field(default_factory=list)


def _extract_memory_terms(text: str) -> Set[str]:
    """从文本中提取医学实体词（基于 MOCK_ENTITY_LEXICON）。"""
    # lazy import 避免循环依赖
    from src.tools.ner_tool import MOCK_ENTITY_LEXICON
    terms: Set[str] = set()
    for terms_in_type in MOCK_ENTITY_LEXICON.values():
        for term in terms_in_type:
            if term in text:
                terms.add(term)
    return terms


def _extract_query_terms(query: str) -> Set[str]:
    """从 query 中提取医学实体词，追加常见 symptom/body-parts 子串匹配。"""
    terms = _extract_memory_terms(query)
    # 追加常见模式：中文 2-4 字词
    common_patterns = {"头痛", "发烧", "发热", "咳嗽", "腹痛", "胸痛", "呕吐", "恶心",
                       "头晕", "乏力", "失眠", "心悸", "腹泻", "便秘", "皮疹", "黄疸",
                       "出血", "肿胀", "疼痛", "痉挛", "呼吸困难"}
    for pat in common_patterns:
        if pat in query:
            terms.add(pat)
    return terms


def score_memory_relevance(
    query: str,
    memory: Dict[str, Any],
    threshold: float = 0.2,
) -> MemoryGateDecision:
    """
    计算 query 与 episodic memory 的相关性分数。

    规则（无 embedding）:
    1. 提取 query 和 memory 中的医学实体词
    2. overlap / max(1, len(query_terms))
    3. diagnoses 直接匹配加权

    返回 MemoryGateDecision.
    """
    query = query or ""
    if not query or not memory:
        return MemoryGateDecision(keep=False, score=0.0, reason="empty_query_or_memory")

    query_terms = _extract_query_terms(query)
    if not query_terms:
        return MemoryGateDecision(keep=False, score=0.0, reason="empty_query_or_memory",
                                  matched_terms=[])

    # memory text: summary + diagnoses
    summary = memory.get("summary", "") or ""
    diagnoses = memory.get("diagnoses") or []
    if isinstance(diagnoses, list):
        diag_text = " ".join(str(d) for d in diagnoses)
    else:
        diag_text = str(diagnoses) if diagnoses else ""
    memory_text = f"{summary} {diag_text}"

    memory_terms = _extract_memory_terms(memory_text)
    if not memory_terms:
        return MemoryGateDecision(keep=False, score=0.0, reason="empty_query_or_memory",
                                  matched_terms=[])

    # 计算 overlap terms
    overlap = query_terms & memory_terms

    # base score: overlap ratio
    base_score = min(1.0, len(overlap) / max(1, len(query_terms)))

    # bonus: diagnosis overlap
    diag_bonus = 0.0
    diag_matched = []
    for diag in diagnoses if isinstance(diagnoses, list) else [diagnoses]:
        diag_str = str(diag)
        diag_terms = _extract_memory_terms(diag_str)
        diag_overlap = query_terms & diag_terms
        if diag_overlap:
            diag_bonus = 0.4
            diag_matched = list(diag_overlap)

    # bonus: summary overlap
    summary_terms = _extract_memory_terms(summary)
    summary_overlap = query_terms & summary_terms
    summary_bonus = 0.2 if summary_overlap else 0.0

    score = base_score + diag_bonus + summary_bonus
    score = min(1.0, score)

    matched = list(overlap | set(diag_matched))

    # reason
    if score >= threshold:
        if diag_bonus > 0:
            reason = "diagnosis_overlap"
        else:
            reason = "term_overlap"
    else:
        reason = "below_threshold"

    return MemoryGateDecision(
        keep=score >= threshold,
        score=round(score, 2),
        reason=reason,
        matched_terms=matched,
    )


# ============================================================
# Batch gating
# ============================================================

def gate_episodic_hints(
    query: str,
    hints: List[Dict[str, Any]],
    enabled: bool = True,
    threshold: float = 0.2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    对 episodic_hints 做 batch gating。

    Returns:
        (kept_hints, gate_records)
    """
    records = []
    for hint in hints:
        decision = score_memory_relevance(query, hint, threshold=threshold)
        records.append({
            "episode_id": str(hint.get("episode_id", "")),
            "keep": decision.keep,
            "score": decision.score,
            "reason": decision.reason,
            "matched_terms": decision.matched_terms[:10],
        })

    if enabled:
        kept = [h for h, r in zip(hints, records) if r["keep"]]
    else:
        # disabled: 全保留
        kept = list(hints)
        for r in records:
            r["keep"] = True
            r["reason"] = "gating_disabled"

    return kept, records
