"""Metrics for the multi-session episodic-memory benchmark."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List


def _reciprocal_rank(ranked_ids: List[str], relevant_ids: set[str]) -> float:
    for rank, episode_id in enumerate(ranked_ids, start=1):
        if episode_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranked_ids: List[str], relevant_ids: set[str], k: int = 5) -> float:
    if not relevant_ids:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, episode_id in enumerate(ranked_ids[:k], start=1)
        if episode_id in relevant_ids
    )
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def score_memory_result(result: Dict[str, Any]) -> Dict[str, float]:
    """Score one scenario from exported memory metadata and answer text."""
    relevant = set(result.get("expected_memory_ids") or [])
    forbidden = set(result.get("forbidden_memory_ids") or [])
    retrieved = list(result.get("retrieved_episode_ids") or [])
    injected = list(result.get("injected_episode_ids") or [])
    answer = result.get("answer", "") or ""
    must_include = result.get("must_include") or []
    must_not_include = result.get("must_not_include") or []
    expected_critical = set(result.get("expected_critical_facts") or [])
    injected_critical = set(result.get("injected_critical_facts") or [])

    retrieval_hits = relevant.intersection(retrieved)
    injection_hits = relevant.intersection(injected)
    irrelevant_injected = [eid for eid in injected if eid not in relevant]
    cross_user = [eid for eid in retrieved if eid.startswith("other::")]

    required_recall = (
        sum(1 for term in must_include if term in answer) / len(must_include)
        if must_include else 1.0
    )
    forbidden_hits = float(sum(1 for term in must_not_include if term in answer))
    return {
        "retrieval_recall": len(retrieval_hits) / len(relevant) if relevant else 1.0,
        "injection_recall": len(injection_hits) / len(relevant) if relevant else 1.0,
        "mrr": _reciprocal_rank(retrieved, relevant) if relevant else 1.0,
        "ndcg_at_5": _ndcg_at_k(retrieved, relevant, k=5),
        "irrelevant_injected": float(len(irrelevant_injected)),
        "forbidden_injected": float(len(forbidden.intersection(injected))),
        "cross_user_leakage": float(len(cross_user)),
        "answer_required_recall": required_recall,
        "answer_forbidden_hits": forbidden_hits,
        "answer_constraint_pass": float(required_recall == 1.0 and forbidden_hits == 0),
        "critical_profile_recall": (
            len(expected_critical.intersection(injected_critical)) / len(expected_critical)
            if expected_critical else 1.0
        ),
    }


def aggregate_memory_results(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    all_rows = list(results)
    rows = [row for row in all_rows if row.get("success")]
    if not rows:
        return {"success_count": 0, "error_count": len(all_rows)}

    scored = [score_memory_result(row) for row in rows]
    total = len(scored)
    report = {
        "success_count": total,
        "error_count": sum(1 for row in all_rows if not row.get("success")),
    }
    relevant_pairs = [
        (row, score_memory_result(row)) for row in rows
        if row.get("expected_memory_ids")
    ]
    for key in ("retrieval_recall", "injection_recall", "mrr", "ndcg_at_5"):
        report[key] = (
            sum(item[key] for _, item in relevant_pairs) / len(relevant_pairs)
            if relevant_pairs else 0.0
        )
    report["irrelevant_injected"] = sum(item["irrelevant_injected"] for item in scored) / total
    answer_required_pairs = [
        (row, score_memory_result(row)) for row in rows if row.get("must_include")
    ]
    report["answer_required_recall"] = (
        sum(item["answer_required_recall"] for _, item in answer_required_pairs) / len(answer_required_pairs)
        if answer_required_pairs else 0.0
    )
    report["forbidden_injection_rate"] = (
        sum(1 for item in scored if item["forbidden_injected"] > 0) / total
    )
    report["cross_user_leakage_rate"] = (
        sum(1 for item in scored if item["cross_user_leakage"] > 0) / total
    )
    answer_forbidden_pairs = [
        (row, score_memory_result(row)) for row in rows if row.get("must_not_include")
    ]
    report["answer_forbidden_rate"] = (
        sum(1 for _, item in answer_forbidden_pairs if item["answer_forbidden_hits"] > 0) / len(answer_forbidden_pairs)
        if answer_forbidden_pairs else 0.0
    )
    constrained_pairs = [
        (row, score_memory_result(row)) for row in rows
        if row.get("must_include") or row.get("must_not_include")
    ]
    report["answer_constraint_pass_rate"] = (
        sum(item["answer_constraint_pass"] for _, item in constrained_pairs) / len(constrained_pairs)
        if constrained_pairs else 0.0
    )
    report["avg_retrieved"] = sum(len(row.get("retrieved_episode_ids") or []) for row in rows) / total
    report["avg_injected"] = sum(len(row.get("injected_episode_ids") or []) for row in rows) / total
    report["avg_filtered"] = sum(row.get("filtered_count", 0) for row in rows) / total
    report["avg_long_term_chars"] = sum(row.get("long_term_context_chars", 0) for row in rows) / total
    report["avg_elapsed_ms"] = sum(row.get("elapsed_ms", 0) for row in rows) / total

    critical_rows = [row for row in rows if row.get("expected_critical_facts")]
    if critical_rows:
        report["critical_episodic_injection_recall"] = sum(
            score_memory_result(row)["injection_recall"] for row in critical_rows
        ) / len(critical_rows)
        report["critical_profile_recall"] = sum(
            score_memory_result(row)["critical_profile_recall"] for row in critical_rows
        ) / len(critical_rows)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("category", "unknown")].append(row)
    report["by_category"] = {}
    for category, category_rows in sorted(grouped.items()):
        category_relevant = [row for row in category_rows if row.get("expected_memory_ids")]
        category_answer_rows = [row for row in category_rows if row.get("must_include")]
        report["by_category"][category] = {
            "count": len(category_rows),
            "retrieval_recall": (
                sum(score_memory_result(row)["retrieval_recall"] for row in category_relevant) / len(category_relevant)
                if category_relevant else None
            ),
            "injection_recall": (
                sum(score_memory_result(row)["injection_recall"] for row in category_relevant) / len(category_relevant)
                if category_relevant else None
            ),
            "irrelevant_injected": sum(score_memory_result(row)["irrelevant_injected"] for row in category_rows) / len(category_rows),
            "forbidden_injection_rate": sum(
                1 for row in category_rows if score_memory_result(row)["forbidden_injected"] > 0
            ) / len(category_rows),
            "answer_required_recall": (
                sum(score_memory_result(row)["answer_required_recall"] for row in category_answer_rows) / len(category_answer_rows)
                if category_answer_rows else None
            ),
            "answer_constraint_pass_rate": (
                sum(score_memory_result(row)["answer_constraint_pass"] for row in category_answer_rows) / len(category_answer_rows)
                if category_answer_rows else None
            ),
        }
    return report
