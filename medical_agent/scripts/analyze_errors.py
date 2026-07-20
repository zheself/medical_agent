"""
scripts/analyze_errors.py — V4 误差分析：按复杂度 × PPR 使用交叉分析

回答关键问题：
1. PPR 在 hard 题上是否提升了命中率？
2. PPR 在 medium 题上是否引入了噪声（降低命中率）？
3. 哪些 item 的 PPR 使用是"正确"的，哪些是"多余"的？
4. 选择性路由的潜在收益有多大？

用法:
    conda run -n cjz_opd python scripts/analyze_errors.py \
        --predictions eval_results/cmb_clin_vllm_v4/raw_predictions.jsonl \
        --tool-diagnosis eval_results/tool_diagnosis_v5_vllm/item_tool_details.json \
        --output-dir eval_results/error_analysis_v4
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ============================================================
# 匹配逻辑（与 eval/metrics.py 一致）
# ============================================================

def _loose_match(pred: str, gold: str) -> bool:
    """宽松匹配：精确相等或一方完整包含另一方。"""
    if pred == gold:
        return True
    if pred in gold or gold in pred:
        return True
    return False


def _is_hit(pred_diagnoses: List[str], gold_diagnoses: List[str], top_k: int) -> bool:
    """Top-K 宽松命中：pred 前 K 个诊断中是否有任一命中 gold 中的任一诊断。"""
    if not pred_diagnoses or not gold_diagnoses:
        return False
    top = pred_diagnoses[:top_k]
    for g in gold_diagnoses:
        for p in top:
            if _loose_match(p, g):
                return True
    return False


def _hit_rank(pred_diagnoses: List[str], gold_diagnoses: List[str]) -> int:
    """返回首次命中的排名（1-based），0 表示未命中。"""
    if not pred_diagnoses or not gold_diagnoses:
        return 0
    for i, p in enumerate(pred_diagnoses):
        for g in gold_diagnoses:
            if _loose_match(p, g):
                return i + 1
    return 0


def _count_hits(pred_diagnoses: List[str], gold_diagnoses: List[str]) -> int:
    """计算 pred 中命中 gold 的诊断数（loose match）。"""
    if not pred_diagnoses or not gold_diagnoses:
        return 0
    hits = 0
    for g in gold_diagnoses:
        for p in pred_diagnoses:
            if _loose_match(p, g):
                hits += 1
                break
    return hits


# ============================================================
# 主分析
# ============================================================

def load_predictions(path: str) -> List[Dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_tool_diagnosis(path: str) -> Dict[str, Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["item_id"]: item for item in data}


def analyze(predictions: List[Dict], tool_diag: Dict[str, Dict]) -> Dict:
    """执行交叉分析，返回结构化结果。"""

    # ---- 逐条分析 ----
    items = []
    for pred in predictions:
        item_id = pred["item_id"]
        gold = pred.get("gold_diagnoses") or []
        pred_diag = pred.get("predicted_diagnoses") or []
        difficulty = pred.get("difficulty", "?")

        td = tool_diag.get(item_id, {})
        has_ppr = td.get("has_ppr", False)
        planner_complexity = td.get("complexity", "?")
        tools_used = td.get("tools_unique", [])
        tools_raw = td.get("tools_raw", [])
        has_duplicates = td.get("has_duplicate_tools", False)

        hit_rank = _hit_rank(pred_diag, gold)
        top1_hit = hit_rank == 1
        top3_hit = 1 <= hit_rank <= 3
        top5_hit = 1 <= hit_rank <= 5
        any_hit = hit_rank > 0
        n_hits = _count_hits(pred_diag, gold)

        items.append({
            "item_id": item_id,
            "difficulty": difficulty,
            "planner_complexity": planner_complexity,
            "has_ppr": has_ppr,
            "tools_used": tools_used,
            "tools_raw": tools_raw,
            "has_duplicates": has_duplicates,
            "gold_diagnoses": gold,
            "predicted_diagnoses": pred_diag,
            "hit_rank": hit_rank,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "top5_hit": top5_hit,
            "any_hit": any_hit,
            "n_hits": n_hits,
            "n_pred": len(pred_diag),
            "n_gold": len(gold),
            "latency_ms": pred.get("total_elapsed_ms", 0),
            "tokens": pred.get("total_tokens", 0),
        })

    n = len(items)

    # ---- 分层统计 ----
    def subgroup_stats(subset: List[Dict], label: str) -> Dict:
        m = len(subset)
        if m == 0:
            return {"label": label, "count": 0}
        top1 = sum(1 for x in subset if x["top1_hit"]) / m
        top3 = sum(1 for x in subset if x["top3_hit"]) / m
        top5 = sum(1 for x in subset if x["top5_hit"]) / m
        any_hit = sum(1 for x in subset if x["any_hit"]) / m
        avg_hits = sum(x["n_hits"] for x in subset) / m
        avg_pred = sum(x["n_pred"] for x in subset) / m
        avg_latency = sum(x["latency_ms"] for x in subset) / m
        avg_tokens = sum(x["tokens"] for x in subset) / m
        return {
            "label": label,
            "count": m,
            "top1_hit_rate": round(top1, 4),
            "top3_hit_rate": round(top3, 4),
            "top5_hit_rate": round(top5, 4),
            "any_hit_rate": round(any_hit, 4),
            "avg_hits_per_item": round(avg_hits, 2),
            "avg_pred_count": round(avg_pred, 2),
            "avg_latency_ms": round(avg_latency, 0),
            "avg_tokens": round(avg_tokens, 0),
        }

    # 全局
    overall = subgroup_stats(items, "overall")

    # 按 difficulty
    by_difficulty = {}
    for diff in ["hard", "medium"]:
        subset = [x for x in items if x["difficulty"] == diff]
        by_difficulty[diff] = subgroup_stats(subset, f"difficulty={diff}")

    # 按 PPR 使用
    by_ppr = {}
    for ppr_flag, label in [(True, "PPR=yes"), (False, "PPR=no")]:
        subset = [x for x in items if x["has_ppr"] == ppr_flag]
        by_ppr[label] = subgroup_stats(subset, label)

    # 交叉：difficulty × PPR —— 这是核心分析
    cross_tab = {}
    for diff in ["hard", "medium"]:
        for ppr_flag, ppr_label in [(True, "PPR"), (False, "noPPR")]:
            label = f"{diff}+{ppr_label}"
            subset = [x for x in items if x["difficulty"] == diff and x["has_ppr"] == ppr_flag]
            cross_tab[label] = subgroup_stats(subset, label)

    # 按 planner_complexity
    by_complexity = {}
    for comp in ["high", "medium", "low"]:
        subset = [x for x in items if x["planner_complexity"] == comp]
        if subset:
            by_complexity[comp] = subgroup_stats(subset, f"planner_complexity={comp}")

    # ---- 分类 item 列表 ----
    # 四种关键类别
    hard_ppr_hit = [x for x in items if x["difficulty"] == "hard" and x["has_ppr"] and x["any_hit"]]
    hard_ppr_miss = [x for x in items if x["difficulty"] == "hard" and x["has_ppr"] and not x["any_hit"]]
    hard_noppr_miss = [x for x in items if x["difficulty"] == "hard" and not x["has_ppr"] and not x["any_hit"]]
    hard_noppr_hit = [x for x in items if x["difficulty"] == "hard" and not x["has_ppr"] and x["any_hit"]]
    medium_ppr_hit = [x for x in items if x["difficulty"] == "medium" and x["has_ppr"] and x["any_hit"]]
    medium_ppr_miss = [x for x in items if x["difficulty"] == "medium" and x["has_ppr"] and not x["any_hit"]]
    medium_noppr_hit = [x for x in items if x["difficulty"] == "medium" and not x["has_ppr"] and x["any_hit"]]
    medium_noppr_miss = [x for x in items if x["difficulty"] == "medium" and not x["has_ppr"] and not x["any_hit"]]

    # ---- What-if 分析 ----
    # 1. 选择性路由（仅 hard 用 PPR，medium 不用 PPR）
    #    对 hard 题保持实际结果；对 medium 题估算无 PPR 时的表现
    #    估算方式：medium+PPR item 的命中率 = medium+noPPR 命中率（保守）
    medium_noppr_hit_rate = (
        len(medium_noppr_hit) / (len(medium_noppr_hit) + len(medium_noppr_miss))
        if (len(medium_noppr_hit) + len(medium_noppr_miss)) > 0 else 0
    )
    selective_scenario = {
        "hard_items": len(hard_ppr_hit) + len(hard_ppr_miss) + len(hard_noppr_hit) + len(hard_noppr_miss),
        "hard_hits": len(hard_ppr_hit) + len(hard_noppr_hit),
        "medium_items_current_ppr": len(medium_ppr_hit) + len(medium_ppr_miss),
        "medium_estimated_hits_if_noppr": round((len(medium_ppr_hit) + len(medium_ppr_miss)) * medium_noppr_hit_rate),
        "medium_actual_hits_with_ppr": len(medium_ppr_hit),
        "medium_noppr_items": len(medium_noppr_hit) + len(medium_noppr_miss),
        "medium_noppr_hits": len(medium_noppr_hit),
        "medium_noppr_hit_rate": round(medium_noppr_hit_rate, 4),
        "note": "对 medium 题当前用 PPR 的 item，用 medium+noPPR 命中率估算去除 PPR 后的命中数"
    }
    # 计算选择性路由后的总体指标
    selective_total = selective_scenario["hard_items"] + selective_scenario["medium_items_current_ppr"] + selective_scenario["medium_noppr_items"]
    selective_hits = selective_scenario["hard_hits"] + selective_scenario["medium_estimated_hits_if_noppr"] + selective_scenario["medium_noppr_hits"]
    selective_scenario["total_items"] = selective_total
    selective_scenario["estimated_hits"] = selective_hits
    selective_scenario["estimated_hit_rate"] = round(selective_hits / selective_total, 4) if selective_total > 0 else 0

    # 2. 最优选择性路由（假设 hard+noPPR item 加上 PPR 后能达到 hard+PPR 的命中率）
    hard_ppr_hit_rate = (
        len(hard_ppr_hit) / (len(hard_ppr_hit) + len(hard_ppr_miss))
        if (len(hard_ppr_hit) + len(hard_ppr_miss)) > 0 else 0
    )
    optimal_scenario = {
        "hard_noppr_items": len(hard_noppr_hit) + len(hard_noppr_miss),
        "hard_noppr_actual_hits": len(hard_noppr_hit),
        "hard_noppr_potential_hits_if_ppr": round((len(hard_noppr_hit) + len(hard_noppr_miss)) * hard_ppr_hit_rate),
        "hard_ppr_hit_rate": round(hard_ppr_hit_rate, 4),
        "note": "对 hard 题当前未用 PPR 的 item，用 hard+PPR 命中率估算加上 PPR 后的命中数"
    }

    return {
        "total_items": n,
        "overall": overall,
        "by_difficulty": by_difficulty,
        "by_ppr": by_ppr,
        "cross_tab": cross_tab,
        "by_planner_complexity": by_complexity,
        "categories": {
            "hard_ppr_hit": {"count": len(hard_ppr_hit), "items": [x["item_id"] for x in hard_ppr_hit]},
            "hard_ppr_miss": {"count": len(hard_ppr_miss), "items": [x["item_id"] for x in hard_ppr_miss]},
            "hard_noppr_miss": {"count": len(hard_noppr_miss), "items": [x["item_id"] for x in hard_noppr_miss]},
            "hard_noppr_hit": {"count": len(hard_noppr_hit), "items": [x["item_id"] for x in hard_noppr_hit]},
            "medium_ppr_hit": {"count": len(medium_ppr_hit), "items": [x["item_id"] for x in medium_ppr_hit]},
            "medium_ppr_miss": {"count": len(medium_ppr_miss), "items": [x["item_id"] for x in medium_ppr_miss]},
            "medium_noppr_hit": {"count": len(medium_noppr_hit), "items": [x["item_id"] for x in medium_noppr_hit]},
            "medium_noppr_miss": {"count": len(medium_noppr_miss), "items": [x["item_id"] for x in medium_noppr_miss]},
        },
        "selective_routing_scenario": selective_scenario,
        "optimal_routing_scenario": optimal_scenario,
        # 详细 item 列表（用于人工检查）
        "per_item": items,
    }


def print_report(results: Dict):
    """打印人类可读的分析报告。"""
    print("=" * 90)
    print("  V4 误差分析：PPR 使用 × 难度交叉分析")
    print("=" * 90)

    print(f"\n总 item 数: {results['total_items']}")

    # ---- 1. 全局 ----
    o = results["overall"]
    print(f"\n{'─' * 70}")
    print("  1. 全局指标")
    print(f"{'─' * 70}")
    print(f"  Top-1: {o['top1_hit_rate']:.1%}  Top-3: {o['top3_hit_rate']:.1%}  Top-5: {o['top5_hit_rate']:.1%}")
    print(f"  Any Hit: {o['any_hit_rate']:.1%}  Avg Hits/Item: {o['avg_hits_per_item']}  Avg Preds: {o['avg_pred_count']}")
    print(f"  Avg Latency: {o['avg_latency_ms']:.0f}ms  Avg Tokens: {o['avg_tokens']:.0f}")

    # ---- 2. 按难度 ----
    print(f"\n{'─' * 70}")
    print("  2. 按难度分层")
    print(f"{'─' * 70}")
    for diff in ["hard", "medium"]:
        d = results["by_difficulty"].get(diff)
        if d:
            print(f"  {diff:8s} (n={d['count']:2d}): Top-1={d['top1_hit_rate']:.1%}  Top-3={d['top3_hit_rate']:.1%}  Top-5={d['top5_hit_rate']:.1%}  AnyHit={d['any_hit_rate']:.1%}  Lat={d['avg_latency_ms']:.0f}ms")

    # ---- 3. 按 PPR 使用 ----
    print(f"\n{'─' * 70}")
    print("  3. 按 PPR 使用分层")
    print(f"{'─' * 70}")
    for label in ["PPR=yes", "PPR=no"]:
        d = results["by_ppr"].get(label)
        if d:
            print(f"  {label:8s} (n={d['count']:2d}): Top-1={d['top1_hit_rate']:.1%}  Top-3={d['top3_hit_rate']:.1%}  Top-5={d['top5_hit_rate']:.1%}  AnyHit={d['any_hit_rate']:.1%}  Lat={d['avg_latency_ms']:.0f}ms")

    # ---- 4. 交叉分析（核心） ----
    print(f"\n{'─' * 70}")
    print("  4. 交叉分析：难度 × PPR（核心）")
    print(f"{'─' * 70}")
    header = f"  {'类别':<20s} {'n':>3s}  {'Top-1':>6s}  {'Top-3':>6s}  {'Top-5':>6s}  {'AnyHit':>7s}  {'Lat(ms)':>8s}  {'Tokens':>7s}"
    print(header)
    print(f"  {'─' * 85}")
    for label in ["hard+PPR", "hard+noPPR", "medium+PPR", "medium+noPPR"]:
        d = results["cross_tab"].get(label)
        if d and d["count"] > 0:
            print(f"  {label:<20s} {d['count']:3d}  {d['top1_hit_rate']:6.1%}  {d['top3_hit_rate']:6.1%}  {d['top5_hit_rate']:6.1%}  {d['any_hit_rate']:7.1%}  {d['avg_latency_ms']:8.0f}  {d['avg_tokens']:7.0f}")
        elif d and d["count"] == 0:
            print(f"  {label:<20s}   0  (无数据)")

    # ---- 5. 关键发现 ----
    print(f"\n{'─' * 70}")
    print("  5. 关键发现")
    print(f"{'─' * 70}")

    cat = results["categories"]

    # PPR 在 hard 题上的贡献
    hard_ppr = cat["hard_ppr_hit"]["count"] + cat["hard_ppr_miss"]["count"]
    hard_noppr = cat["hard_noppr_hit"]["count"] + cat["hard_noppr_miss"]["count"]
    if hard_ppr > 0:
        ppr_hard_hit_rate = cat["hard_ppr_hit"]["count"] / hard_ppr
        print(f"\n  Hard 题 PPR 使用情况:")
        print(f"    PPR 调用: {hard_ppr} 条, 命中 {cat['hard_ppr_hit']['count']} 条 ({ppr_hard_hit_rate:.1%})")
        print(f"    PPR 未调用: {hard_noppr} 条, 命中 {cat['hard_noppr_hit']['count']} 条 ({cat['hard_noppr_hit']['count']/hard_noppr:.1%})" if hard_noppr else "")
        if cat["hard_noppr_miss"]["count"] > 0:
            print(f"    ⚠️  Hard 题未调 PPR 且未命中: {cat['hard_noppr_miss']['count']} 条 → 可能需要 PPR")
            for iid in cat["hard_noppr_miss"]["items"][:5]:
                print(f"       {iid}")

    # PPR 在 medium 题上的潜在噪声
    medium_ppr = cat["medium_ppr_hit"]["count"] + cat["medium_ppr_miss"]["count"]
    medium_noppr = cat["medium_noppr_hit"]["count"] + cat["medium_noppr_miss"]["count"]
    if medium_ppr > 0:
        ppr_med_hit_rate = cat["medium_ppr_hit"]["count"] / medium_ppr
        noppr_med_hit_rate = cat["medium_noppr_hit"]["count"] / medium_noppr if medium_noppr else 0
        print(f"\n  Medium 题 PPR 使用情况:")
        print(f"    PPR 调用: {medium_ppr} 条, 命中 {cat['medium_ppr_hit']['count']} 条 ({ppr_med_hit_rate:.1%})")
        print(f"    PPR 未调用: {medium_noppr} 条, 命中 {cat['medium_noppr_hit']['count']} 条 ({noppr_med_hit_rate:.1%})")
        delta = noppr_med_hit_rate - ppr_med_hit_rate
        direction = "PPR noise" if delta > 0 else "PPR helps"
        print(f"    Δ (noPPR − PPR): {delta:+.1%} → {direction}")
        if cat["medium_ppr_miss"]["count"] > 0:
            print(f"    ⚠️  Medium 题调了 PPR 但未命中: {cat['medium_ppr_miss']['count']} 条 → 可能是 PPR 噪声")
            for iid in cat["medium_ppr_miss"]["items"][:5]:
                print(f"       {iid}")

    # ---- 6. What-if 选择性路由 ----
    print(f"\n{'─' * 70}")
    print("  6. What-if 分析：选择性路由")
    print(f"{'─' * 70}")

    sel = results["selective_routing_scenario"]
    print(f"\n  场景 A: 仅 hard 用 PPR，medium 不用 PPR")
    print(f"    Hard 题 (保持现状): {sel['hard_items']} 条, 命中 {sel['hard_hits']} 条")
    print(f"    Medium 题 当前用 PPR: {sel['medium_items_current_ppr']} 条, 实际命中 {sel['medium_actual_hits_with_ppr']} 条")
    print(f"    Medium 题 去除 PPR 估算命中: {sel['medium_estimated_hits_if_noppr']} 条 (基于 noPPR 命中率 {sel['medium_noppr_hit_rate']:.1%})")
    print(f"    Medium 题 本就不调 PPR: {sel['medium_noppr_items']} 条, 命中 {sel['medium_noppr_hits']} 条")
    print(f"    总命中率估算: {sel['estimated_hits']}/{sel['total_items']} = {sel['estimated_hit_rate']:.1%}")
    print(f"    ⚡ Latency 收益: 省去 {sel['medium_items_current_ppr']} 条 medium item 的 PPR 调用")

    opt = results["optimal_routing_scenario"]
    print(f"\n  场景 B (上限): Hard 当前未用 PPR 的加上 PPR")
    print(f"    Hard 题未用 PPR: {opt['hard_noppr_items']} 条, 实际命中 {opt['hard_noppr_actual_hits']} 条")
    print(f"    若加上 PPR 估算命中: {opt['hard_noppr_potential_hits_if_ppr']} 条 (基于 hard+PPR 命中率 {opt['hard_ppr_hit_rate']:.1%})")
    print(f"    潜在收益: +{opt['hard_noppr_potential_hits_if_ppr'] - opt['hard_noppr_actual_hits']} 条命中")

    # ---- 7. Planner 复杂度与实际难度对照 ----
    print(f"\n{'─' * 70}")
    print("  7. Planner 复杂度判断 vs 实际难度")
    print(f"{'─' * 70}")
    # 统计 planner 对 hard/medium 题的判断分布
    for diff in ["hard", "medium"]:
        items_in_diff = [x for x in results["per_item"] if x["difficulty"] == diff]
        comp_dist = Counter(x["planner_complexity"] for x in items_in_diff)
        comp_str = ", ".join(f"{k}:{v}" for k, v in comp_dist.most_common())
        print(f"  实际{diff:6s} (n={len(items_in_diff)}): Planner 判断 → {comp_str}")

    # ---- 8. 工具组合 vs 命中率 ----
    print(f"\n{'─' * 70}")
    print("  8. PPR 工具组合 vs 命中率")
    print(f"{'─' * 70}")
    # 按 PPR + global_search 组合分析
    combo_stats = defaultdict(lambda: {"count": 0, "hits": 0, "top3_hits": 0})
    for x in results["per_item"]:
        has_global = "kg_global_search" in x["tools_used"]
        has_local = "kg_local_search" in x["tools_used"]
        has_ppr = x["has_ppr"]
        combo = []
        if has_ppr: combo.append("PPR")
        if has_global: combo.append("Global")
        if has_local: combo.append("Local")
        key = "+".join(combo) if combo else "none"
        combo_stats[key]["count"] += 1
        if x["any_hit"]:
            combo_stats[key]["hits"] += 1
        if x["top3_hit"]:
            combo_stats[key]["top3_hits"] += 1

    print(f"  {'Combo':<20s} {'n':>4s}  {'AnyHit':>7s}  {'Top3Hit':>8s}")
    print(f"  {'─' * 45}")
    for combo in sorted(combo_stats.keys()):
        s = combo_stats[combo]
        c = s["count"]
        print(f"  {combo:<20s} {c:4d}  {s['hits']/c:7.1%}  {s['top3_hits']/c:8.1%}")

    print(f"\n{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(description="V4 误差分析：PPR × 难度交叉分析")
    parser.add_argument("--predictions", required=True, help="raw_predictions.jsonl 路径")
    parser.add_argument("--tool-diagnosis", required=True, help="item_tool_details.json 路径")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    tool_path = Path(args.tool_diagnosis)

    if not pred_path.exists():
        print(f"错误: predictions 文件不存在: {pred_path}")
        sys.exit(1)
    if not tool_path.exists():
        print(f"错误: tool diagnosis 文件不存在: {tool_path}")
        sys.exit(1)

    print(f"加载 predictions: {pred_path}")
    predictions = load_predictions(str(pred_path))
    print(f"加载 tool diagnosis: {tool_path}")
    tool_diag = load_tool_diagnosis(str(tool_path))

    print(f"\npredictions: {len(predictions)} 条, tool diagnosis: {len(tool_diag)} 条")

    # 检查 item_id 覆盖
    pred_ids = {p["item_id"] for p in predictions}
    tool_ids = set(tool_diag.keys())
    missing_in_tool = pred_ids - tool_ids
    missing_in_pred = tool_ids - pred_ids
    if missing_in_tool:
        print(f"⚠️  predictions 中有 {len(missing_in_tool)} 条不在 tool diagnosis 中: {sorted(missing_in_tool)[:5]}...")
    if missing_in_pred:
        print(f"⚠️  tool diagnosis 中有 {len(missing_in_pred)} 条不在 predictions 中: {sorted(missing_in_pred)[:5]}...")

    results = analyze(predictions, tool_diag)

    # 输出
    print_report(results)

    # 保存
    out_dir = Path(args.output_dir) if args.output_dir else pred_path.parent.parent / "error_analysis_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整 JSON
    json_path = out_dir / "error_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n完整分析数据: {json_path}")

    # 保存精简 CSV（方便 Excel 查看）
    csv_path = out_dir / "per_item_analysis.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id", "difficulty", "planner_complexity", "has_ppr",
            "tools", "hit_rank", "top3_hit", "any_hit",
            "n_pred", "n_gold", "n_hits",
            "gold_diagnoses", "pred_diagnoses_top3",
            "latency_ms", "tokens"
        ])
        for item in results["per_item"]:
            writer.writerow([
                item["item_id"],
                item["difficulty"],
                item["planner_complexity"],
                item["has_ppr"],
                "|".join(item["tools_used"]),
                item["hit_rank"],
                item["top3_hit"],
                item["any_hit"],
                item["n_pred"],
                item["n_gold"],
                item["n_hits"],
                "; ".join(item["gold_diagnoses"]),
                "; ".join(item["predicted_diagnoses"][:3]),
                item["latency_ms"],
                item["tokens"],
            ])
    print(f"逐条 CSV: {csv_path}")


if __name__ == "__main__":
    main()
