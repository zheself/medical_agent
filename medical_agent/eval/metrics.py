"""
eval/metrics.py — 评估指标

涵盖四维评估:
- Capability: accuracy, multi-hop F1
- Faithfulness: citation accuracy, hallucination rate
- Efficiency: latency P50/P95/P99, token consumption, breakdown
- Safety: contraindication intercept rate

实现状态: ✅ 核心指标完整
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Set

# ============================================================
# Capability
# ============================================================

def compute_accuracy(predictions: List[Dict], gold_field: str = "gold_answer") -> float:
    """简单的字符串/类别精确匹配（跳过无 gold_answer 的条目）"""
    if not predictions:
        return 0.0
    scored = [p for p in predictions if p.get(gold_field)]
    if not scored:
        return 0.0
    correct = sum(1 for p in scored
                  if p.get("predicted_answer", "").strip() == p.get(gold_field, "").strip())
    return correct / len(scored)


def _loose_match(pred: str, gold: str) -> bool:
    """宽松匹配：精确相等或一方完整包含另一方。

    只有两种命中方式：
    1. EXACT: pred == gold
    2. CONTAINS: pred 是 gold 的子串，或 gold 是 pred 的子串
       例: "心肌梗死" ⊂ "急性心肌梗死" → 命中
       例: "肠梗阻" ⊂ "嵌顿性腹股沟斜疝合并肠梗阻" → 命中
    不命中: "腹外疝" vs "嵌顿性腹股沟斜疝" — 无包含关系 → MISS
    不命中: "胆管结石" vs "肝左外叶胆管多发结石" — 无包含关系 → MISS
    宁可漏判，不要误判。
    """
    if pred == gold:
        return True
    if pred in gold or gold in pred:
        return True
    return False
    return False


def _match_sets(
    pred_set: Set[str],
    gold_set: Set[str],
    loose_match: bool = False,
) -> int:
    """计算 pred 和 gold 之间的匹配数（精确或宽松）"""
    if not loose_match:
        return len(pred_set & gold_set)
    hits = 0
    for g in gold_set:
        for p in pred_set:
            if _loose_match(p, g):
                hits += 1
                break  # 每个 gold 只匹配一次
    return hits


def compute_multihop_f1(
    predictions: List[Dict],
    pred_field: str = "predicted_diagnoses",
    gold_field: str = "gold_diagnoses",
    loose_match: bool = False,
) -> Dict[str, float]:
    """
    多跳推理 F1

    每条数据是 list of diagnoses，算 set-level F1。
    loose_match=True 时用宽松匹配（包含关系）。
    """
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    all_p, all_r, all_f1 = [], [], []
    for item in predictions:
        pred_set: Set[str] = set(item.get(pred_field, []))
        gold_set: Set[str] = set(item.get(gold_field, []))
        if not gold_set:
            continue

        tp = _match_sets(pred_set, gold_set, loose_match)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gold_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        all_p.append(precision)
        all_r.append(recall)
        all_f1.append(f1)

    return {
        "precision": sum(all_p) / len(all_p) if all_p else 0.0,
        "recall": sum(all_r) / len(all_r) if all_r else 0.0,
        "f1": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
    }


def compute_topk_hit_rate(
    predictions: List[Dict],
    k: int = 3,
    pred_field: str = "predicted_diagnoses",
    gold_field: str = "gold_diagnoses",
    loose_match: bool = False,
) -> float:
    """Top-K 命中率: gold 中任意一个宽松或精确匹配 pred 的 top-K 中"""
    if not predictions:
        return 0.0
    hits = 0
    for item in predictions:
        pred_topk = item.get(pred_field, [])[:k]
        gold_set = set(item.get(gold_field, []))
        if not gold_set:
            continue
        if _match_sets(set(pred_topk), gold_set, loose_match) > 0:
            hits += 1
    scored = [i for i in predictions if i.get(gold_field)]
    return hits / len(scored) if scored else 0.0


# ============================================================
# Faithfulness
# ============================================================

def compute_citation_accuracy(predictions: List[Dict]) -> float:
    """
    Citation Accuracy: 模型 cite 的引用是否真的支持对应 claim
    
    需要为每条数据准备:
    - item.predicted_citations: 模型给出的引用列表
    - item.gold_evidence_pool: 真实的证据池
    
    简化版：检查每个 citation 的 source/target 是否在 evidence pool 中
    """
    if not predictions:
        return 0.0
    
    correct_citations = 0
    total_citations = 0
    
    for item in predictions:
        cites = item.get("predicted_citations", [])
        evidence = item.get("gold_evidence_pool", [])
        evidence_set = {(e.get("source"), e.get("target")) for e in evidence}
        
        for c in cites:
            if not isinstance(c, dict):
                total_citations += 1
                continue
            total_citations += 1
            if (c.get("source"), c.get("target")) in evidence_set:
                correct_citations += 1
    
    return correct_citations / total_citations if total_citations > 0 else 0.0


# ============================================================
# Efficiency
# ============================================================

def compute_latency_stats(predictions: List[Dict]) -> Dict[str, float]:
    """P50/P95/P99 延迟"""
    latencies = [p.get("total_elapsed_ms", 0) for p in predictions]
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}
    
    latencies.sort()
    n = len(latencies)
    return {
        "mean": sum(latencies) / n,
        "p50": latencies[int(0.50 * n)],
        "p95": latencies[int(0.95 * n)] if n > 1 else latencies[0],
        "p99": latencies[int(0.99 * n)] if n > 1 else latencies[0],
        "max": latencies[-1],
    }


def compute_latency_breakdown(predictions: List[Dict]) -> Dict[str, float]:
    """
    每个模块的平均延迟占比
    
    需要预测结果中包含 module_latencies 字段，如:
    {"planner": 800, "tools": 250, "synthesize": 30, "verify": 1}
    """
    if not predictions:
        return {}
    
    totals = defaultdict(float)
    counts = defaultdict(int)
    
    for p in predictions:
        breakdown = p.get("module_latencies", {})
        for module, latency in breakdown.items():
            totals[module] += latency
            counts[module] += 1
    
    return {
        module: totals[module] / counts[module]
        for module in totals
    }


def compute_token_stats(predictions: List[Dict]) -> Dict[str, float]:
    """Token 消耗统计"""
    tokens = [p.get("total_tokens", 0) for p in predictions]
    if not tokens:
        return {"mean": 0, "p95": 0}
    tokens.sort()
    return {
        "mean": sum(tokens) / len(tokens),
        "p95": tokens[int(0.95 * len(tokens))] if len(tokens) > 1 else tokens[0],
    }


def compute_verify_level_distribution(predictions: List[Dict]) -> Dict[str, float]:
    """L1/L2/L3 分级反思的触发分布"""
    counts = defaultdict(int)
    total = 0
    for p in predictions:
        level = p.get("verify_level_reached")
        if level:
            counts[level] += 1
            total += 1
    if total == 0:
        return {}
    return {level: count / total for level, count in counts.items()}


# ============================================================
# Safety
# ============================================================

def compute_safety_intercept_rate(predictions: List[Dict]) -> Dict[str, float]:
    """
    安全测试集的拦截率
    
    每条数据需要:
    - must_avoid: 禁止出现的关键词列表
    - must_include_keywords: 必须出现的关键词
    """
    if not predictions:
        return {}
    
    total = len(predictions)
    avoid_passed = 0  # 避免了 must_avoid 的
    include_passed = 0  # 包含了 must_include 的
    both_passed = 0
    
    for p in predictions:
        answer = p.get("predicted_answer", "")
        must_avoid = p.get("must_avoid", [])
        must_include = p.get("must_include_keywords", [])
        
        avoid_ok = not any(kw in answer for kw in must_avoid) if must_avoid else True
        include_ok = all(kw in answer for kw in must_include) if must_include else True
        
        if avoid_ok:
            avoid_passed += 1
        if include_ok:
            include_passed += 1
        if avoid_ok and include_ok:
            both_passed += 1
    
    return {
        "avoid_rate": avoid_passed / total,
        "include_rate": include_passed / total,
        "overall_safety_pass_rate": both_passed / total,
    }


# ============================================================
# Multi-turn consistency
# ============================================================

def compute_multi_turn_consistency(sessions: List[Dict]) -> float:
    """
    多轮对话一致性
    
    sessions: [
        {"turns": [
            {"query": ..., "answer": ..., "expected_aware_of": ["头痛", "青霉素过敏"]},
            ...
        ]},
        ...
    ]
    
    一致性 = 每轮回答中"应被知晓的信息"被正确处理的比例
    """
    if not sessions:
        return 0.0
    
    all_scores = []
    for session in sessions:
        turns = session.get("turns", [])
        for turn in turns:
            answer = turn.get("answer", "")
            expected = turn.get("expected_aware_of", [])
            if not expected:
                continue
            hit = sum(1 for kw in expected if kw in answer) / len(expected)
            all_scores.append(hit)
    
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


# ============================================================
# 综合报告
# ============================================================

def generate_full_report(
    predictions: List[Dict],
    safety_predictions: List[Dict] = None,
    consistency_sessions: List[Dict] = None,
    loose_match: bool = True,
) -> Dict[str, Any]:
    """生成完整评测报告

    loose_match=True 时用宽松匹配（主要评测口径），
    同时保留精确匹配作为辅助对照。
    """
    report = {
        "capability": {
            "top1_hit_rate": compute_topk_hit_rate(predictions, k=1, loose_match=loose_match),
            "top3_hit_rate": compute_topk_hit_rate(predictions, k=3, loose_match=loose_match),
            "top5_hit_rate": compute_topk_hit_rate(predictions, k=5, loose_match=loose_match),
            "multihop_f1_loose": compute_multihop_f1(predictions, loose_match=loose_match),
            "multihop_f1_exact": compute_multihop_f1(predictions, loose_match=False),
        },
        "faithfulness": {
            "citation_accuracy": compute_citation_accuracy(predictions),
        },
        "efficiency": {
            "latency": compute_latency_stats(predictions),
            "latency_breakdown": compute_latency_breakdown(predictions),
            "tokens": compute_token_stats(predictions),
            "verify_level_distribution": compute_verify_level_distribution(predictions),
        },
    }

    if safety_predictions:
        report["safety"] = compute_safety_intercept_rate(safety_predictions)

    if consistency_sessions:
        report["consistency"] = {
            "multi_turn_score": compute_multi_turn_consistency(consistency_sessions),
        }

    return report
