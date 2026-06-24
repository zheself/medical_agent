"""
scripts/diagnose_tool_usage.py — 诊断 Planner 工具选择

直接调用 Planner LLM，生成每条评测数据的计划，
统计工具调用分布。不执行工具，不跑完整 pipeline。

用法:
    conda run -n cjz_opd python scripts/diagnose_tool_usage.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.factory import build_system


def main():
    # 加载评测数据
    eval_path = ROOT / "data" / "eval_cmb_clin.jsonl"
    items = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append(d)

    print(f"加载评测数据: {len(items)} 条\n")

    # 构建系统，提取 Planner（用 db backend，真实 KG）
    agent, _ = build_system(backend="db")
    planner = agent.planner

    # 统计
    tool_counter = Counter()       # 各工具被调用的 item 数
    tool_step_counter = Counter()  # 各工具被调用的总 step 数
    item_tools = []                # 每条 item 的工具列表
    ppr_items = []                 # 调用了 PPR 的 item_id 列表

    for i, item in enumerate(items):
        query = item["query"]
        item_id = item.get("item_id", str(i))

        # 调用 Planner 生成计划
        try:
            plan = planner.plan(query, working_memory=None, episodic_hints=None)
        except Exception as e:
            print(f"  [{i+1}/{len(items)}] {item_id}: Planner 失败 — {e}")
            item_tools.append({"item_id": item_id, "tools": [], "error": str(e)})
            continue

        tools_used = [step.tool for step in plan.steps]
        complexity = plan.complexity.value if hasattr(plan.complexity, "value") else str(plan.complexity)

        # 统计
        tools_set = set(tools_used)
        for t in tools_set:
            tool_counter[t] += 1
        for t in tools_used:
            tool_step_counter[t] += 1

        has_ppr = "ppr_reasoner" in tools_set
        if has_ppr:
            ppr_items.append(item_id)

        item_tools.append({
            "item_id": item_id,
            "query": query[:60],
            "tools": tools_used,
            "complexity": complexity,
            "thought_preview": plan.thought[:100],
            "has_ppr": has_ppr,
        })

        # 进度
        if (i + 1) % 10 == 0 or i == 0 or has_ppr:
            ppr_flag = " ★ PPR" if has_ppr else ""
            print(f"  [{i+1}/{len(items)}] {item_id}: {tools_used} (complexity={complexity}){ppr_flag}")

    # ===== 输出统计 =====
    print("\n" + "=" * 80)
    print("  工具调用统计")
    print("=" * 80)
    print(f"\n总 item 数: {len(items)}")
    print(f"成功生成计划: {len([x for x in item_tools if not x.get('error')])}")
    print(f"Planner 失败: {len([x for x in item_tools if x.get('error')])}")

    print(f"\n--- 各工具被多少条 item 调用 ---")
    for tool, count in tool_counter.most_common():
        pct = count / len(item_tools) * 100
        print(f"  {tool}: {count} 条 ({pct:.1f}%)")

    print(f"\n--- 各工具总 step 调用次数 ---")
    for tool, count in tool_step_counter.most_common():
        print(f"  {tool}: {count} 次")

    print(f"\n--- PPR 调用情况 ---")
    if ppr_items:
        print(f"  调用 PPR 的 item: {ppr_items}")
    else:
        print(f"  ⚠️ 没有任何 item 调用 ppr_reasoner")

    # 复杂度分布
    complexity_counter = Counter(x.get("complexity", "?") for x in item_tools)
    print(f"\n--- 复杂度分布 ---")
    for c, count in complexity_counter.most_common():
        print(f"  {c}: {count} 条")

    # 工具组合模式
    combo_counter = Counter(tuple(sorted(set(x["tools"]))) for x in item_tools if not x.get("error"))
    print(f"\n--- 工具组合模式（去重后） ---")
    for combo, count in combo_counter.most_common(10):
        print(f"  {combo}: {count} 条")

    # 保存详细数据
    out_dir = ROOT / "eval_results" / "tool_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "item_tool_details.json", "w", encoding="utf-8") as f:
        json.dump(item_tools, f, ensure_ascii=False, indent=2)

    summary = {
        "total_items": len(items),
        "successful": len([x for x in item_tools if not x.get("error")]),
        "failed": len([x for x in item_tools if x.get("error")]),
        "tool_item_count": dict(tool_counter.most_common()),
        "tool_step_count": dict(tool_step_counter.most_common()),
        "ppr_items": ppr_items,
        "ppr_call_rate": len(ppr_items) / len(item_tools) if item_tools else 0,
        "complexity_distribution": dict(complexity_counter),
        "tool_combos": {str(k): v for k, v in combo_counter.most_common(10)},
    }
    with open(out_dir / "tool_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n详细数据已保存: {out_dir}")


if __name__ == "__main__":
    main()