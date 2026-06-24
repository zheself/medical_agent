"""
eval/run_ablation.py — 消融实验

逐个开关组件，观察指标变化，产出对比表。

================================================================
⚠️  关于 mock 数字的重要声明
================================================================
本脚本默认用 MOCK 系统运行（mock LLM / mock KG / mock 向量库）。

  - ✅ 真实有效的部分：消融**流程**本身——数据集加载、逐配置运行、
       指标计算、对比表生成。各配置的开关（prefetch / memory / 反思级别）
       都是代码层面**真实生效**的。
  - ⚠️ 不具科研意义的部分：mock 系统跑出的**绝对数字**。mock LLM 行为固定，
       不会因为"关掉 Memory"就真的变笨，mock KG 数据量也极小。

  ➜  要得到可写进简历的真实消融数字，请在 GPU 服务器上：
       1. 把 build_ablation_agent 里的 mock 组件替换为真实组件
          （vLLM / Neo4j / Milvus / 蒸馏 Verifier，参考 demo 的 build_system）
       2. 用真实评测集（CMB / MedQA / 自建集）替换 --data-path
       3. 重新跑本脚本，同一套流程直接产出真实对比表
================================================================

用法:
    # mock 模式（验证流程，默认）
    python -m eval.run_ablation

    # 指定数据集 + 真实 agent（需自行实现 build_real_agent）
    python -m eval.run_ablation --data-path ./data/self_built.jsonl --real
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import EvalItem, UserQuery


# ============================================================
# 消融配置：每个配置是"在完整系统基础上关掉某个组件"
# ============================================================

@dataclass
class AblationConfig:
    name: str
    description: str
    enable_prefetch: bool = True
    enable_memory_injection: bool = True
    max_verifier_level: str = "L3"
    use_ppr_idf: bool = True


# 消融阶梯：从"完整系统"逐步关掉组件
# 注意：mock 下 LLM 行为固定，这里主要验证"开关生效 + 流程跑通"
ABLATION_CONFIGS: List[AblationConfig] = [
    AblationConfig(
        name="full",
        description="完整系统（所有组件开启）",
    ),
    AblationConfig(
        name="no_memory",
        description="关闭 Memory 注入（无历史/冷启动）",
        enable_memory_injection=False,
    ),
    AblationConfig(
        name="no_idf",
        description="关闭 PPR IDF 边加权",
        use_ppr_idf=False,
    ),
    AblationConfig(
        name="verify_l1_only",
        description="反思仅 L1 规则（无 L2/L3）",
        max_verifier_level="L1",
    ),
    AblationConfig(
        name="minimal",
        description="最小系统（关 prefetch + 关 memory + 仅 L1 + 无 IDF）",
        enable_prefetch=False,
        enable_memory_injection=False,
        max_verifier_level="L1",
        use_ppr_idf=False,
    ),
]


# ============================================================
# 构造带开关的 agent
# ============================================================

def build_ablation_agent(config: AblationConfig, backend: str = "mock"):
    """按消融配置构造 agent，统一走 src.factory.build_system"""
    from src.factory import build_system
    agent, episodic = build_system(backend=backend)

    # 应用消融开关（真实生效）
    agent.enable_prefetch = config.enable_prefetch
    agent.enable_memory_injection = config.enable_memory_injection
    agent.max_verifier_level = config.max_verifier_level
    if hasattr(agent.verifier, "max_level"):
        agent.verifier.max_level = config.max_verifier_level

    # PPR IDF 开关
    if not config.use_ppr_idf:
        ppr_tool = agent.tools.get("ppr_reasoner")
        if ppr_tool:
            ppr_tool.use_idf = False

    return agent, episodic


# ============================================================
# 评测数据
# ============================================================

def builtin_eval_items() -> List[EvalItem]:
    """内置的演示评测集（mock 模式用）"""
    return [
        EvalItem(
            item_id="d1",
            query="我最近三天持续头痛伴发烧 38.5°C，颈部僵硬，可能是什么病？",
            gold_diagnoses=["脑膜炎", "脑炎", "蛛网膜下腔出血"],
            min_hops_required=2, difficulty="hard", user_id="u1",
        ),
        EvalItem(
            item_id="d2",
            query="二甲双胍是什么药？",
            gold_diagnoses=[], difficulty="easy", user_id="u1",
        ),
        EvalItem(
            item_id="d3",
            query="糖尿病人最近视力模糊，需要注意什么？",
            gold_diagnoses=["视网膜病变"],
            min_hops_required=2, difficulty="medium", user_id="u1",
        ),
        EvalItem(
            item_id="d4",
            query="头痛发烧应该挂什么科？",
            gold_diagnoses=["流感", "脑膜炎"],
            difficulty="medium", user_id="u2",
        ),
    ]


def load_items(data_path: str) -> List[EvalItem]:
    items = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append(EvalItem(
                item_id=d.get("item_id", str(len(items))),
                query=d["query"],
                gold_answer=d.get("gold_answer"),
                gold_diagnoses=d.get("gold_diagnoses", []),
                min_hops_required=d.get("min_hops_required", 1),
                difficulty=d.get("difficulty", "medium"),
                user_id=d.get("user_id", "eval_user"),
            ))
    return items


# ============================================================
# 跑单个配置
# ============================================================

async def run_config(config: AblationConfig, items: List[EvalItem], backend: str) -> Dict[str, Any]:
    agent, _ = build_ablation_agent(config, backend=backend)

    predictions = []
    for item in items:
        ans = await agent.answer_async(UserQuery(user_id=f"eval_{item.item_id}", text=item.query))

        # 抽取预测诊断（从 FinalAnswer.diagnoses，LLM 结构化输出）
        pred_diag = ans.diagnoses
        pred_diag = [d for d in dict.fromkeys(pred_diag) if d][:5]

        predictions.append({
            "item_id": item.item_id,
            "gold_diagnoses": item.gold_diagnoses,
            "predicted_diagnoses": pred_diag,
            "predicted_answer": ans.content,
            "predicted_citations": ans.citations,
            "total_elapsed_ms": ans.total_elapsed_ms,
            "verify_level_reached": (
                ans.verification_results[-1].level.value
                if ans.verification_results else "none"
            ),
        })

    # 计算指标
    from eval.metrics import (
        compute_multihop_f1,
        compute_topk_hit_rate,
        compute_latency_stats,
        compute_verify_level_distribution,
    )
    f1_exact = compute_multihop_f1(predictions, loose_match=False)
    f1_loose = compute_multihop_f1(predictions, loose_match=True)
    metrics = {
        "config": config.name,
        "description": config.description,
        "f1_exact": round(f1_exact["f1"], 3),
        "f1_loose": round(f1_loose["f1"], 3),
        "top3_exact": round(compute_topk_hit_rate(predictions, k=3, loose_match=False), 3),
        "top3_loose": round(compute_topk_hit_rate(predictions, k=3, loose_match=True), 3),
        "mean_latency_ms": round(compute_latency_stats(predictions)["mean"], 2),
        "verify_dist": compute_verify_level_distribution(predictions),
        "prefetch_hit_rate": round(agent.get_prefetch_hit_rate(), 3),
    }
    return metrics


# ============================================================
# 打印对比表
# ============================================================

def print_ablation_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 100)
    print("  消融实验结果")
    print("=" * 100)
    header = f"{'配置':<12}{'F1_exact':>10}{'F1_loose':>10}{'Top3_exact':>10}{'Top3_loose':>10}{'延迟ms':>10}{'预取命中':>10}  {'说明'}"
    print(header)
    print("-" * 100)
    for r in results:
        print(
            f"{r['config']:<12}"
            f"{r['f1_exact']:>10}"
            f"{r['f1_loose']:>10}"
            f"{r['top3_exact']:>10}"
            f"{r['top3_loose']:>10}"
            f"{r['mean_latency_ms']:>10}"
            f"{r['prefetch_hit_rate']:>10}"
            f"  {r['description']}"
        )
    print("=" * 100)


WARNING_BANNER = """
████████████████████████████████████████████████████████████████████████████
⚠️  MOCK 模式 — 下面的数字仅用于验证消融【流程】跑通，不具科研意义！
    mock LLM 行为固定，不会因关掉组件就真变笨。
    要得到可写进简历的真实数字，请在 GPU 服务器上替换为真实组件后重跑。
    详见本文件顶部说明。
████████████████████████████████████████████████████████████████████████████
"""


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=None, help="评测集 JSONL；不给则用内置 demo 集")
    parser.add_argument("--backend", default="mock", help="后端模式: mock | db | vllm")
    parser.add_argument("--num-items", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--output-dir", default="./eval_results")
    args = parser.parse_args()

    backend = args.backend
    if backend != "vllm":
        print(WARNING_BANNER)

    items = load_items(args.data_path) if args.data_path else builtin_eval_items()
    if args.num_items:
        items = items[:args.num_items]
    print(f"评测集: {len(items)} 条 | 配置数: {len(ABLATION_CONFIGS)} | backend: {backend}\n")

    results = []
    for config in ABLATION_CONFIGS:
        print(f"  跑配置: {config.name} ({config.description}) ...")
        metrics = await run_config(config, items, backend)
        results.append(metrics)

    print_ablation_table(results)

    if backend != "vllm":
        print(WARNING_BANNER)

    # 保存
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
