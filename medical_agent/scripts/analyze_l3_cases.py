"""
scripts/analyze_l3_cases.py — L3 reflexion case analysis

从 V6 评测结果中提取 L3 行为模式，不做 prompt 修改。

分析维度：
1. L3 触发统计：哪些 item 触发了 L3，频率、难度分布
2. L3 修正效果：L3 allowed vs L2-only 的差异（hit/miss 翻转）
3. L3 候选模式：L3 是否只是扩充候选，还是重排/纠错
4. 识别 L3 明显缺陷：把正确诊断挤出 Top-3、过度扩充、无效反思

用法:
    python scripts/analyze_l3_cases.py \
        --v6-l3allowed eval_results/cmb_clin_vllm_v6/raw_predictions.jsonl \
        --v6-l2only eval_results/cmb_clin_vllm_v6_L2only/raw_predictions.jsonl \
        --output-dir eval_results/l3_analysis
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _loose_match(pred: str, gold: str) -> bool:
    if pred == gold:
        return True
    if pred in gold or gold in pred:
        return True
    return False


def is_hit(pred_diag: List[str], gold_diag: List[str], top_k: int) -> bool:
    if not pred_diag or not gold_diag:
        return False
    for g in gold_diag:
        for p in pred_diag[:top_k]:
            if _loose_match(p, g):
                return True
    return False


def hit_rank(pred_diag: List[str], gold_diag: List[str]) -> int:
    if not pred_diag or not gold_diag:
        return 0
    for i, p in enumerate(pred_diag):
        for g in gold_diag:
            if _loose_match(p, g):
                return i + 1
    return 0


def load(path: str) -> Dict[str, Dict]:
    items = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "error" not in d:
                items[d["item_id"]] = d
    return items


def analyze(v6_l3: Dict[str, Dict], v6_l2: Dict[str, Dict]) -> Dict:
    """分析 L3 行为模式"""
    common = sorted(set(v6_l3.keys()) & set(v6_l2.keys()))
    print(f"Common items: {len(common)} (V6 L3={len(v6_l3)}, V6 L2={len(v6_l2)})")

    cases = []
    l3_triggered = []
    l3_helped = []      # L3 hit, L2 miss
    l3_hurt = []        # L2 hit, L3 miss
    l3_no_effect_hit = []  # both hit
    l3_no_effect_miss = []  # both miss

    for iid in common:
        l3_item = v6_l3[iid]
        l2_item = v6_l2[iid]
        gold = l3_item.get("gold_diagnoses") or []
        pred_l3 = l3_item.get("predicted_diagnoses") or []
        pred_l2 = l2_item.get("predicted_diagnoses") or []
        level = l3_item.get("verify_level_reached", "?")
        difficulty = l3_item.get("difficulty", "?")

        l3_hit_top3 = is_hit(pred_l3, gold, 3)
        l2_hit_top3 = is_hit(pred_l2, gold, 3)
        l3_rank = hit_rank(pred_l3, gold)
        l2_rank = hit_rank(pred_l2, gold)

        case = {
            "item_id": iid,
            "difficulty": difficulty,
            "verify_level": level,
            "gold": gold,
            "pred_l3": pred_l3[:5],
            "pred_l2": pred_l2[:5],
            "l3_hit_top3": l3_hit_top3,
            "l2_hit_top3": l2_hit_top3,
            "l3_rank": l3_rank,
            "l2_rank": l2_rank,
            "n_pred_l3": len(pred_l3),
            "n_pred_l2": len(pred_l2),
            "latency_l3": l3_item.get("total_elapsed_ms", 0),
            "latency_l2": l2_item.get("total_elapsed_ms", 0),
            "tokens_l3": l3_item.get("total_tokens", 0),
            "tokens_l2": l2_item.get("total_tokens", 0),
        }
        cases.append(case)

        if level == "L3":
            l3_triggered.append(case)

        # L3 修正方向
        if l3_hit_top3 and not l2_hit_top3:
            l3_helped.append(case)
        elif not l3_hit_top3 and l2_hit_top3:
            l3_hurt.append(case)
        elif l3_hit_top3 and l2_hit_top3:
            l3_no_effect_hit.append(case)
        else:
            l3_no_effect_miss.append(case)

    # ---- 汇总统计 ----
    summary = {
        "total_common": len(common),
        "l3_triggered_count": len(l3_triggered),
        "l3_triggered_pct": len(l3_triggered) / len(common) if common else 0,
        "l3_helped": len(l3_helped),
        "l3_hurt": len(l3_hurt),
        "l3_no_effect_hit": len(l3_no_effect_hit),
        "l3_no_effect_miss": len(l3_no_effect_miss),
    }

    # L3 触发率 by difficulty
    l3_by_diff = Counter(c["difficulty"] for c in l3_triggered)
    total_by_diff = Counter(c["difficulty"] for c in cases)
    summary["l3_by_difficulty"] = {
        diff: f"{l3_by_diff[diff]}/{total_by_diff[diff]} ({l3_by_diff[diff]/total_by_diff[diff]:.1%})"
        for diff in sorted(total_by_diff)
    }

    # 候选数对比
    summary["avg_preds_l3_triggered"] = (
        sum(c["n_pred_l3"] for c in l3_triggered) / len(l3_triggered)
        if l3_triggered else 0
    )
    summary["avg_preds_l2_for_l3_triggered"] = (
        sum(c["n_pred_l2"] for c in l3_triggered) / len(l3_triggered)
        if l3_triggered else 0
    )

    # L3 修正方向细分
    summary["l3_helped_items"] = [c["item_id"] for c in l3_helped]
    summary["l3_hurt_items"] = [c["item_id"] for c in l3_hurt]

    # L3 成本
    if l3_triggered:
        summary["l3_avg_latency_ms"] = sum(c["latency_l3"] for c in l3_triggered) / len(l3_triggered)
        summary["l3_avg_tokens"] = sum(c["tokens_l3"] for c in l3_triggered) / len(l3_triggered)
        non_l3 = [c for c in cases if c["verify_level"] != "L3"]
        if non_l3:
            summary["non_l3_avg_latency_ms"] = sum(c["latency_l3"] for c in non_l3) / len(non_l3)
            summary["non_l3_avg_tokens"] = sum(c["tokens_l3"] for c in non_l3) / len(non_l3)

    return {
        "summary": summary,
        "l3_triggered": l3_triggered,
        "l3_helped": l3_helped,
        "l3_hurt": l3_hurt,
        "all_cases": cases,
    }


def print_report(results: Dict):
    s = results["summary"]

    print("=" * 80)
    print("  L3 Reflexion Case Analysis")
    print("=" * 80)

    print(f"\nCommon items: {s['total_common']}")
    print(f"L3 triggered: {s['l3_triggered_count']} ({s['l3_triggered_pct']:.1%})")
    print(f"L3 by difficulty: {s['l3_by_difficulty']}")

    print(f"\n{'─' * 60}")
    print("  L3 修正方向")
    print(f"{'─' * 60}")
    print(f"  L3 helped  (L3 hit, L2 miss):  {s['l3_helped']}")
    if s['l3_helped_items']:
        for iid in s['l3_helped_items']:
            print(f"    {iid}")
    print(f"  L3 hurt    (L2 hit, L3 miss):  {s['l3_hurt']}")
    if s['l3_hurt_items']:
        for iid in s['l3_hurt_items']:
            print(f"    {iid}")
    print(f"  Both hit:                     {s['l3_no_effect_hit']}")
    print(f"  Both miss:                    {s['l3_no_effect_miss']}")

    print(f"\n{'─' * 60}")
    print("  L3 候选数影响")
    print(f"{'─' * 60}")
    print(f"  L3 triggered items: avg preds = {s['avg_preds_l3_triggered']:.1f}")
    print(f"  Same items @ L2-only: avg preds = {s['avg_preds_l2_for_l3_triggered']:.1f}")

    if "l3_avg_latency_ms" in s:
        print(f"\n{'─' * 60}")
        print("  L3 成本")
        print(f"{'─' * 60}")
        print(f"  L3 items avg latency: {s['l3_avg_latency_ms']:.0f}ms")
        print(f"  L3 items avg tokens:  {s['l3_avg_tokens']:.0f}")
        if "non_l3_avg_latency_ms" in s:
            print(f"  Non-L3 avg latency:   {s['non_l3_avg_latency_ms']:.0f}ms")
            print(f"  Non-L3 avg tokens:    {s['non_l3_avg_tokens']:.0f}")

    # ---- L3 helped detail ----
    if results["l3_helped"]:
        print(f"\n{'─' * 60}")
        print("  ★ L3 HELPED cases (L3 hit, L2 miss)")
        print(f"{'─' * 60}")
        for c in results["l3_helped"]:
            print(f"  {c['item_id']} ({c['difficulty']}):")
            print(f"    Gold:    {c['gold']}")
            print(f"    L3 pred: {c['pred_l3']}")
            print(f"    L2 pred: {c['pred_l2']}")
            print(f"    L3 rank={c['l3_rank']}  L2 rank={c['l2_rank']}")

    # ---- L3 hurt detail ----
    if results["l3_hurt"]:
        print(f"\n{'─' * 60}")
        print("  ✗ L3 HURT cases (L2 hit, L3 miss)")
        print(f"{'─' * 60}")
        for c in results["l3_hurt"]:
            print(f"  {c['item_id']} ({c['difficulty']}):")
            print(f"    Gold:    {c['gold']}")
            print(f"    L3 pred: {c['pred_l3']}")
            print(f"    L2 pred: {c['pred_l2']}")
            print(f"    L3 rank={c['l3_rank']}  L2 rank={c['l2_rank']}")

    # ---- L3 triggered but both miss (无效 L3) ----
    l3_both_miss = [c for c in results["l3_triggered"] if not c["l3_hit_top3"] and not c["l2_hit_top3"]]
    if l3_both_miss:
        print(f"\n{'─' * 60}")
        print(f"  ⚠ L3 triggered but BOTH miss ({len(l3_both_miss)} cases)")
        print(f"{'─' * 60}")
        for c in l3_both_miss[:10]:
            print(f"  {c['item_id']} ({c['difficulty']}):")
            print(f"    Gold:    {c['gold']}")
            print(f"    L3 pred: {c['pred_l3'][:3]}")
            print(f"    L2 pred: {c['pred_l2'][:3]}")

    print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="L3 Reflexion Case Analysis")
    parser.add_argument("--v6-l3allowed", required=True, help="V6 PPR ON + L3 allowed raw_predictions.jsonl")
    parser.add_argument("--v6-l2only", required=True, help="V6 PPR ON + L2 only raw_predictions.jsonl")
    parser.add_argument("--output-dir", default="eval_results/l3_analysis")
    args = parser.parse_args()

    v6_l3 = load(args.v6_l3allowed)
    v6_l2 = load(args.v6_l2only)
    print(f"Loaded: L3={len(v6_l3)}, L2={len(v6_l2)}")

    results = analyze(v6_l3, v6_l2)
    print_report(results)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "l3_case_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_dir / 'l3_case_analysis.json'}")


if __name__ == "__main__":
    main()
