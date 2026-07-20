"""
scripts/diagnose_tool_usage.py — 诊断 Planner 工具选择

直接调用 Planner LLM，生成每条评测数据的计划，
统计工具调用分布（含重复步骤统计）。不执行工具，不跑完整 pipeline。

用法:
    # db 模式（sanity check，MockLLM 不受 prompt 变化影响）
    conda run -n cjz_opd python scripts/diagnose_tool_usage.py --backend db

    # vllm 模式（真实验证 prompt 路由效果）
    conda run -n cjz_opd python scripts/diagnose_tool_usage.py --backend vllm --output-dir eval_results/tool_diagnosis_v4_vllm
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.factory import build_system


def main():
    parser = argparse.ArgumentParser(description="诊断 Planner 工具选择分布")
    parser.add_argument("--backend", default="db", help="后端模式: mock | db | vllm（vllm 用于真实验证 prompt 效果）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认 eval_results/tool_diagnosis）")
    args = parser.parse_args()

    # 加载评测数据
    eval_path = ROOT / "data" / "eval_cmb_clin.jsonl"
    items = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append(d)

    print(f"加载评测数据: {len(items)} 条 | backend: {args.backend}\n")

    if args.backend != "vllm":
        print("⚠️  非 vllm 后端 — MockLLM 输出硬编码，prompt 变化不影响工具分布。仅用于 sanity check。\n")

    # 构建系统，提取 Planner
    agent, _ = build_system(backend=args.backend)
    planner = agent.planner

    # 统计
    tool_counter = Counter()       # 各工具被调用的 item 数（去重后）
    tool_step_counter = Counter()  # 各工具被调用的总 step 数（含重复）
    item_tools = []                # 每条 item 的详细信息
    ppr_items = []                 # 调用了 PPR 的 item_id 列表
    items_with_duplicates = 0      # 有重复工具调用的 item 数
    duplicate_detail_counter = Counter()  # (tool_name, repeat_count) → item 数

    for i, item in enumerate(items):
        query = item["query"]
        item_id = item.get("item_id", str(i))

        # 调用 Planner 生成计划
        try:
            plan = planner.plan(query, working_memory=None, episodic_hints=None)
        except Exception as e:
            print(f"  [{i+1}/{len(items)}] {item_id}: Planner 失败 — {e}")
            item_tools.append({"item_id": item_id, "steps": [], "tools_unique": [], "error": str(e)})
            continue

        tools_used = [step.tool for step in plan.steps]
        complexity = plan.complexity.value if hasattr(plan.complexity, "value") else str(plan.complexity)

        # 保存完整 step 信息
        steps_detail = []
        for s in plan.steps:
            steps_detail.append({
                "step_id": s.step_id,
                "tool": s.tool,
                "input": str(s.input)[:120],  # 截断避免过长
                "depends_on": s.depends_on,
            })

        # 统计（去重后——工具是否在该 item 中被调用）
        tools_set = set(tools_used)
        for t in tools_set:
            tool_counter[t] += 1
        # 统计（含重复——总共多少次 step 调用）
        for t in tools_used:
            tool_step_counter[t] += 1

        # 重复工具调用检测
        tool_counts = Counter(tools_used)
        duplicates = {t: c for t, c in tool_counts.items() if c > 1}
        has_duplicates = len(duplicates) > 0
        if has_duplicates:
            items_with_duplicates += 1
            for t, c in duplicates.items():
                duplicate_detail_counter[(t, c)] += 1

        has_ppr = "ppr_reasoner" in tools_set
        if has_ppr:
            ppr_items.append(item_id)

        item_tools.append({
            "item_id": item_id,
            "query": query[:60],
            "steps": steps_detail,
            "tools_unique": sorted(tools_set),
            "tools_raw": tools_used,
            "complexity": complexity,
            "thought_preview": plan.thought[:100],
            "has_ppr": has_ppr,
            "has_duplicate_tools": has_duplicates,
            "duplicate_tools": duplicates,
            "n_steps": len(plan.steps),
        })

        # 进度
        if (i + 1) % 10 == 0 or i == 0 or has_ppr or has_duplicates:
            ppr_flag = " ★ PPR" if has_ppr else ""
            dup_flag = " ⚠️dup" if has_duplicates else ""
            print(f"  [{i+1}/{len(items)}] {item_id}: {tools_used} (complexity={complexity}, steps={len(plan.steps)}){ppr_flag}{dup_flag}")

    # ===== 输出统计 =====
    successful = [x for x in item_tools if not x.get("error")]
    n_successful = len(successful)

    print("\n" + "=" * 80)
    print("  工具调用统计")
    print("=" * 80)
    print(f"\n总 item 数: {len(items)}")
    print(f"成功生成计划: {n_successful}")
    print(f"Planner 失败: {len([x for x in item_tools if x.get('error')])}")

    print(f"\n--- 各工具被多少条 item 调用（去重后） ---")
    for tool, count in tool_counter.most_common():
        pct = count / n_successful * 100
        print(f"  {tool}: {count} 条 ({pct:.1f}%)")

    print(f"\n--- 各工具总 step 调用次数（含重复） ---")
    for tool, count in tool_step_counter.most_common():
        print(f"  {tool}: {count} 次")

    print(f"\n--- PPR 调用情况 ---")
    if ppr_items:
        print(f"  调用 PPR 的 item 数: {len(ppr_items)}")
        print(f"  PPR 调用率: {len(ppr_items) / n_successful * 100:.1f}%")
    else:
        print(f"  ⚠️ 没有任何 item 调用 ppr_reasoner")

    # 复杂度分布
    complexity_counter = Counter(x.get("complexity", "?") for x in item_tools)
    print(f"\n--- 复杂度分布 ---")
    for c, count in complexity_counter.most_common():
        print(f"  {c}: {count} 条")

    # 重复工具调用统计
    print(f"\n--- 重复工具调用统计 ---")
    print(f"  有重复工具调用的 item: {items_with_duplicates} / {n_successful} ({items_with_duplicates / n_successful * 100:.1f}%)")
    if duplicate_detail_counter:
        print(f"  重复详情（工具名×重复次数 → item 数）:")
        for (tool, repeat), count in duplicate_detail_counter.most_common():
            print(f"    {tool} × {repeat} 次: {count} 条 item")

    # 工具组合模式（去重后）
    combo_counter = Counter(tuple(sorted(x["tools_unique"])) for x in successful)
    print(f"\n--- 工具组合模式（去重后） ---")
    for combo, count in combo_counter.most_common(10):
        print(f"  {combo}: {count} 条")

    # 工具组合模式（不去重，含重复）
    combo_raw_counter = Counter(tuple(x["tools_raw"]) for x in successful)
    print(f"\n--- 工具组合模式（不去重，含重复）- top 10 ---")
    for combo, count in combo_raw_counter.most_common(10):
        print(f"  {combo}: {count} 条")

    # 保存详细数据
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "eval_results" / "tool_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "item_tool_details.json", "w", encoding="utf-8") as f:
        json.dump(item_tools, f, ensure_ascii=False, indent=2)

    summary = {
        "total_items": len(items),
        "backend": args.backend,
        "successful": n_successful,
        "failed": len([x for x in item_tools if x.get("error")]),
        "tool_item_count_dedup": dict(tool_counter.most_common()),
        "tool_step_count_raw": dict(tool_step_counter.most_common()),
        "ppr_items": ppr_items,
        "ppr_call_rate": len(ppr_items) / n_successful if n_successful else 0,
        "complexity_distribution": dict(complexity_counter),
        "items_with_duplicate_tools": items_with_duplicates,
        "duplicate_tools_pct": items_with_duplicates / n_successful * 100 if n_successful else 0,
        "duplicate_detail": {f"{t}×{r}": c for (t, r), c in duplicate_detail_counter.most_common()},
        "tool_combos_dedup": {str(k): v for k, v in combo_counter.most_common(10)},
        "tool_combos_raw": {str(k): v for k, v in combo_raw_counter.most_common(10)},
    }
    with open(out_dir / "tool_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n详细数据已保存: {out_dir}")


if __name__ == "__main__":
    main()