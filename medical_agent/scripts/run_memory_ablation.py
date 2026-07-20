"""
scripts/run_memory_ablation.py — Memory Gating Ablation (GPU-free)

三种配置对比 long-term memory 注入效果。

用法:
    python scripts/run_memory_ablation.py --backend mock --num-items 5
    python scripts/run_memory_ablation.py --backend db --num-items 5
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.factory import build_system
from src.schemas import UserQuery
from eval.run_eval import load_eval_items


CONFIGS = [
    {"name": "no_long_term",  "enable_memory_injection": False, "enable_gating": False, "threshold": 0.2},
    {"name": "long_term_raw", "enable_memory_injection": True,  "enable_gating": False, "threshold": 0.2},
    {"name": "long_term_gated","enable_memory_injection": True,  "enable_gating": True,  "threshold": 0.2},
]


async def run_config(backend, config, items):
    agent, _ = build_system(backend=backend)
    agent.enable_memory_injection = config["enable_memory_injection"]
    if config["enable_gating"]:
        agent.enable_memory_gating = True
        agent.memory_gate_threshold = config["threshold"]

    results = []
    for item in items[:args.num_items]:
        q = UserQuery(user_id=f"mem_ablation_{item.item_id}", text=item.query)
        try:
            ans = await agent.answer_async(q, session_id=f"mem_ablation_{item.item_id}")
            em = ans.memory_meta
            results.append({
                "item_id": item.item_id,
                "success": True,
                "injected_count": em["episodic"]["injected_count"],
                "filtered_count": em["episodic"]["filtered_count"],
                "retrieved_count": em["episodic"]["retrieved_count"],
                "long_term_chars": em["injection"]["long_term_context_chars"],
                "total_chars": em["injection"]["total_context_chars_with_working"],
            })
        except Exception as e:
            results.append({"item_id": item.item_id, "success": False, "error": str(e)})
    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--num-items", type=int, default=5)
    global args
    args = parser.parse_args()

    items = load_eval_items(str(ROOT / "data" / "eval_cmb_clin.jsonl"), "cmb_clin")

    print(f"{'Config':<16s} {'OK':>4s} {'AvgRetrieved':>12s} {'AvgInjected':>12s} {'AvgFiltered':>12s} {'AvgLongTermChars':>16s} {'AvgTotalChars':>14s}")
    print("-" * 90)

    all_rows = []
    for cfg in CONFIGS:
        rows = await run_config(args.backend, cfg, items)
        all_rows.append({"config": cfg["name"], "rows": rows})
        n = len(rows)
        ok_rows = [r for r in rows if r["success"]]
        ok = len(ok_rows)
        denom = max(1, ok)
        avg_retrieved = sum(r["retrieved_count"] for r in ok_rows) / denom
        avg_injected = sum(r["injected_count"] for r in ok_rows) / denom
        avg_filtered = sum(r["filtered_count"] for r in ok_rows) / denom
        avg_lt_chars = sum(r["long_term_chars"] for r in ok_rows) / denom
        avg_total = sum(r["total_chars"] for r in ok_rows) / denom
        print(f"{cfg['name']:<16s} {ok:>3d}/{n} {avg_retrieved:>12.1f} {avg_injected:>12.1f} {avg_filtered:>12.1f} {avg_lt_chars:>16.0f} {avg_total:>14.0f}")

    out = ROOT / "eval_results" / "memory_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
