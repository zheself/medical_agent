"""
eval/run_eval.py — 评测入口

用法:
    python -m eval.run_eval --backend mock --output ./eval_results
    python -m eval.run_eval --backend vllm --output ./eval_results
    python -m eval.run_eval --backend vllm --data-path ./data/self_built.jsonl --num-items 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import UserQuery, EvalItem

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mock",
                   help="后端模式: mock | db | vllm")
    p.add_argument("--data-path", default=None, help="评测数据文件 (JSONL)")
    p.add_argument("--num-items", type=int, default=None, help="只跑前 N 条")
    p.add_argument("--output-dir", default="./eval_results")
    p.add_argument("--concurrency", type=int, default=1,
                   help="并发数（vllm 建议 1 避免服务过载）")
    p.add_argument("--no-ppr", action="store_true",
                   help="消融：禁用 PPR 多跳推理（用于评估 PPR 净贡献）")
    p.add_argument("--max-verifier-level", default=None, choices=["L1", "L2", "L3"],
                   help="限制反思最高级别（消融用）")
    p.add_argument("--enable-memory-gating", action="store_true",
                   help="消融：启用 long-term memory relevance gating")
    p.add_argument("--memory-gate-threshold", type=float, default=0.2,
                   help="Memory gating relevance threshold (default 0.2)")
    return p.parse_args()


# ============================================================
# 数据集加载
# ============================================================

def load_eval_items(data_path: str, dataset_type: str) -> List[EvalItem]:
    """从文件加载评测数据"""
    items = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            items.append(EvalItem(
                item_id=data.get("item_id", str(len(items))),
                query=data["query"],
                gold_answer=data.get("gold_answer"),
                gold_diagnoses=data.get("gold_diagnoses", []),
                gold_reasoning_path=data.get("gold_reasoning_path", []),
                min_hops_required=data.get("min_hops_required", 1),
                difficulty=data.get("difficulty", "medium"),
                history=data.get("history", []),
                user_id=data.get("user_id"),
            ))
    return items


# ============================================================
# 系统组装（支持 mock 和 real 两种模式）
# ============================================================

def build_agent(backend: str = "mock"):
    """构造 Agent 实例，统一走 src.factory.build_system"""
    from src.factory import build_system
    agent, episodic = build_system(backend=backend)
    return agent


# ============================================================
# 跑评测
# ============================================================

async def run_single_item(agent, item: EvalItem) -> Dict[str, Any]:
    """跑单条数据"""
    start = time.time()
    
    # 评测隔离：每条用独立 user_id，防止 episodic memory 跨 item 污染
    query = UserQuery(user_id=f"eval_{item.item_id}", text=item.query)
    answer = await agent.answer_async(query)
    
    elapsed_ms = (time.time() - start) * 1000
    
    # 抽取 diagnoses 从 FinalAnswer.diagnoses（LLM 结构化输出）
    predicted_diagnoses = answer.diagnoses
    
    # 抽取 module latencies
    module_latencies = {}
    if answer.plan:
        tool_latencies = [s.elapsed_ms or 0 for s in answer.plan.steps]
        module_latencies["tools_sum"] = sum(tool_latencies)
    if answer.verification_results:
        verify_latency = sum(vr.elapsed_ms for vr in answer.verification_results)
        module_latencies["verify_sum"] = verify_latency
    
    verify_level = (
        answer.verification_results[-1].level.value
        if answer.verification_results else "none"
    )
    
    return {
        "item_id": item.item_id,
        "query": item.query,
        "gold_answer": item.gold_answer,
        "gold_diagnoses": item.gold_diagnoses,
        "predicted_answer": answer.content,
        "predicted_diagnoses": predicted_diagnoses[:5],
        "predicted_citations": answer.citations,
        "total_elapsed_ms": elapsed_ms,
        "agent_reported_elapsed_ms": answer.total_elapsed_ms,
        "module_latencies": module_latencies,
        "verify_level_reached": verify_level,
        "verification_meta": answer.verification_meta,
        "execution_meta": answer.execution_meta,
        "memory_meta": answer.memory_meta,
        "total_tokens": answer.total_tokens,
        "difficulty": item.difficulty,
    }


async def run_eval_async(
    agent,
    items: List[EvalItem],
    output_dir: Path,
    concurrency: int = 1,
) -> List[Dict]:
    """并发跑所有评测数据（限制并发数避免 OOM）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(item):
        async with semaphore:
            try:
                return await run_single_item(agent, item)
            except Exception as e:
                logger.error(f"Failed on {item.item_id}: {e}")
                return {"item_id": item.item_id, "error": str(e)}

    logger.info(f"Running {len(items)} items with concurrency={concurrency}")
    tasks = [bounded(item) for item in items]
    results = []
    for i, task in enumerate(asyncio.as_completed(tasks)):
        result = await task
        results.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"  Done {i + 1}/{len(items)}")
    
    # 保存原始结果
    with open(output_dir / "raw_predictions.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    
    return results


# ============================================================
# Main
# ============================================================

async def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    if args.data_path:
        data_path = args.data_path
    else:
        # 默认用 CMB-Clin 评测集
        default_path = Path(__file__).resolve().parent.parent / "data" / "eval_cmb_clin.jsonl"
        if default_path.exists():
            data_path = str(default_path)
            logger.info(f"Using default CMB-Clin dataset: {data_path}")
        else:
            logger.error(f"Default dataset not found at {default_path}, use --data-path")
            return

    items = load_eval_items(data_path, "cmb_clin")

    if args.num_items:
        items = items[:args.num_items]

    logger.info(f"Loaded {len(items)} eval items")

    # 组装 Agent
    agent = build_agent(backend=args.backend)

    # 消融开关
    if args.no_ppr:
        agent.enable_ppr = False
        logger.info("消融: PPR OFF — ppr_reasoner 步骤将被跳过")
    if args.max_verifier_level:
        agent.max_verifier_level = args.max_verifier_level
        if hasattr(agent.verifier, "max_level"):
            agent.verifier.max_level = args.max_verifier_level
        logger.info(f"消融: max_verifier_level={args.max_verifier_level}")
    if args.enable_memory_gating:
        agent.enable_memory_gating = True
        agent.memory_gate_threshold = args.memory_gate_threshold
        logger.info(f"消融: memory_gating enabled, threshold={args.memory_gate_threshold}")

    if args.backend != "vllm":
        logger.warning(f"Backend={args.backend} — 使用 mock/db 数据，数字不代表真实 LLM 表现")

    # 跑评测
    results = await run_eval_async(
        agent, items, output_dir, concurrency=args.concurrency
    )

    # 计算指标
    from eval.metrics import generate_full_report, compute_multihop_f1, compute_topk_hit_rate

    successful = [r for r in results if "error" not in r]
    report = generate_full_report(successful)

    # 分难度报告
    by_difficulty = {}
    for diff in ["medium", "hard"]:
        subset = [r for r in successful if r.get("difficulty") == diff]
        if subset:
            by_difficulty[diff] = {
                "count": len(subset),
                "top1_exact": compute_topk_hit_rate(subset, k=1, loose_match=False),
                "top1_loose": compute_topk_hit_rate(subset, k=1, loose_match=True),
                "top3_exact": compute_topk_hit_rate(subset, k=3, loose_match=False),
                "top3_loose": compute_topk_hit_rate(subset, k=3, loose_match=True),
                "top5_loose": compute_topk_hit_rate(subset, k=5, loose_match=True),
                "f1_exact": compute_multihop_f1(subset, loose_match=False),
                "f1_loose": compute_multihop_f1(subset, loose_match=True),
            }
    report["by_difficulty"] = by_difficulty

    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 60)
    print("评测报告 — CMB-Clin 77条 vllm")
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n详细结果: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
