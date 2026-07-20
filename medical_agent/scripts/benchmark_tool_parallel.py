"""
scripts/benchmark_tool_parallel.py — DAG executor 并行机制可复现 benchmark

用 fake sleep tool 证明 executor 并行调度有效，不依赖 vLLM 或真实工具。

用法:
    cd medical_agent
    python scripts/benchmark_tool_parallel.py
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import ComplexityLevel, Plan, PlanStep, ToolResult


# ============================================================
# Fake tool + registry
# ============================================================

class FakeSleepTool:
    """固定 sleep 的 fake 工具"""
    def __init__(self, name="sleep", sleep_s=0.1):
        self.name = name
        self.sleep_s = sleep_s

    async def ainvoke(self, resolved_input, **kwargs):
        await asyncio.sleep(self.sleep_s)
        return ToolResult(
            tool_name=self.name, success=True, elapsed_ms=self.sleep_s * 1000,
        )


class FakeToolRegistry:
    def __init__(self):
        self._tools = {}
    def get(self, name):
        if name not in self._tools:
            self._tools[name] = FakeSleepTool(name=name)
        return self._tools[name]


# ============================================================
# Executor (minimal copy of orchestrator method, standalone)
# ============================================================

async def execute_plan_with_meta(orchestrator, plan, query="benchmark"):
    """复用 orchestrator 的 _execute_plan_with_meta。"""
    return await orchestrator._execute_plan_with_meta(plan, query=query)


# ============================================================
# Benchmark plans
# ============================================================

SLEEP_S = 0.15  # 每个 fake step sleep 150ms

SERIAL_PLAN = Plan(
    thought="serial chain", complexity=ComplexityLevel.MEDIUM,
    steps=[
        PlanStep(step_id=1, tool="s1", input={}, depends_on=[]),
        PlanStep(step_id=2, tool="s2", input={}, depends_on=[1]),
        PlanStep(step_id=3, tool="s3", input={}, depends_on=[2]),
    ],
)

PARALLEL_PLAN = Plan(
    thought="parallel layer", complexity=ComplexityLevel.MEDIUM,
    steps=[
        PlanStep(step_id=1, tool="p1", input={}, depends_on=[]),
        PlanStep(step_id=2, tool="p2", input={}, depends_on=[]),
        PlanStep(step_id=3, tool="p3", input={}, depends_on=[]),
    ],
)

MIXED_PLAN = Plan(
    thought="mixed dag", complexity=ComplexityLevel.MEDIUM,
    steps=[
        PlanStep(step_id=1, tool="m1", input={}, depends_on=[]),
        PlanStep(step_id=2, tool="m2", input={}, depends_on=[1]),
        PlanStep(step_id=3, tool="m3", input={}, depends_on=[1]),
        PlanStep(step_id=4, tool="m4", input={}, depends_on=[2, 3]),
    ],
)


# ============================================================
# Main
# ============================================================

async def main():
    from demo.demo_full_flow import build_system
    orch, _ = build_system()
    orch.tools = FakeToolRegistry()
    # 确保 max_parallel_tools 足够大
    orch.max_parallel_tools = 8
    orch._tool_semaphore = asyncio.Semaphore(8)

    results_list = []

    for label, plan in [
        ("serial_chain", SERIAL_PLAN),
        ("parallel_layer", PARALLEL_PLAN),
        ("mixed_dag", MIXED_PLAN),
    ]:
        results, meta = await execute_plan_with_meta(orch, plan)
        results_list.append({"label": label, "meta": meta})
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        print(f"  Wall time:        {meta['tool_wall_ms']:.0f} ms")
        print(f"  Sum tool elapsed: {meta['tool_sum_elapsed_ms']:.0f} ms")
        print(f"  Parallelism ratio:{meta['parallelism_ratio']:.2f}")
        print(f"  Layer widths:     {meta['layer_widths']}")
        print(f"  Max layer width:  {meta['max_layer_width']}")
        print(f"  Layer count:      {meta['layer_count']}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    serial = results_list[0]["meta"]
    parallel = results_list[1]["meta"]
    mixed = results_list[2]["meta"]

    print(f"  serial_chain:   ratio={serial['parallelism_ratio']:.2f}, layers={serial['layer_widths']}")
    print(f"  parallel_layer: ratio={parallel['parallelism_ratio']:.2f}, layers={parallel['layer_widths']}")
    print(f"  mixed_dag:      ratio={mixed['parallelism_ratio']:.2f}, layers={mixed['layer_widths']}")

    # Assertions
    assert serial['parallelism_ratio'] < 1.15, f"serial ratio should be ~1, got {serial['parallelism_ratio']}"
    assert parallel['parallelism_ratio'] > 2.0, f"parallel ratio should be >2, got {parallel['parallelism_ratio']}"
    assert parallel['layer_widths'] == [3], f"parallel should be single layer of 3"
    assert mixed['parallelism_ratio'] > 1.25, f"mixed ratio should be >1.25, got {mixed['parallelism_ratio']}"
    assert mixed['layer_widths'] == [1, 2, 1], f"mixed layers should be [1,2,1], got {mixed['layer_widths']}"

    print("\n✅ All assertions passed — parallel executor works correctly.")

    # Save
    out_path = ROOT / "eval_results" / "benchmark_parallel.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
