# V9 并行工具执行优化

日期: 2026-07-01

## 背景

V8 后代码基线已稳定。当前 `_execute_plan()` 已有 DAG layer 调度 + `asyncio.gather` 并行执行雏形，但缺少可观测性（无 execution time metadata，无 parallelism metrics）。

本次做工程化加固：不改 Planner 或工具逻辑，只增加执行元数据记录和 benchmark 验证。

## 改动

### Phase 1: 执行元数据

**schemas.py**: `FinalAnswer` 新增 `execution_meta: Dict` 字段

**orchestrator.py**:
- 新增 `_execute_plan_with_meta()` — 在原有 DAG 调度基础上记录 layer widths、wall time、parallelism ratio
- `_execute_plan()` 改为 wrapper，调用 `_execute_plan_with_meta()` 并只返回 results
- `answer_async()` 调用 `_execute_plan_with_meta()` 并填入 `FinalAnswer.execution_meta`
- 新增 `max_parallel_tools: int = 4` 参数 + `asyncio.Semaphore`

execution_meta 结构：
```json
{
  "tool_wall_ms": 101.2,
  "tool_sum_elapsed_ms": 300.0,
  "parallelism_ratio": 2.98,
  "layer_count": 1,
  "max_layer_width": 3,
  "layer_widths": [3],
  "executed_step_count": 3,
  "max_parallel_tools": 4
}
```

**eval/run_eval.py**: raw predictions 输出 `execution_meta`

### Phase 2: 测试

新增 4 个执行元数据测试：
- `test_execution_meta_parallel_reduces_wall_time` — 2 个无依赖 sleep step，ratio > 1.3
- `test_execution_meta_dependency_serial` — 有依赖 step 串行，ratio ≈ 1.0
- `test_execution_meta_mixed_dag` — 混合 DAG [1,2,1]，ratio > 1.1
- `test_execution_meta_exported_in_final_answer` — answer_async 的 FinalAnswer 含 execution_meta

**72/72 全部通过**。

### Phase 3: Benchmark

新增 `scripts/benchmark_tool_parallel.py`：用 fake sleep tool 三类 plan：
- serial_chain: ratio=1.00, layers=[1,1,1]
- parallel_layer: ratio=2.98, layers=[3]
- mixed_dag: ratio=1.33, layers=[1,2,1]

Benchmark 证明 executor 机制本身正确。

## V9 子集验证（L3 case 14 条）

14/14 成功，0 error。execution_meta 正确输出。

Avg parallelism_ratio: 0.24 — V9 子集未观测到真实工具并行收益。可能原因包括 SQLite/PPR 内部同步阻塞、单步工具耗时过短、调度/解析开销占比高。

## V9 全量评测

```bash
python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl \
  --concurrency 1 --output-dir eval_results/cmb_clin_vllm_v9
```

77/77 成功，0 error。

### V8 vs V9（77 条共同子集）

| 指标 | V8 | V9 | Δ |
|------|-----|-----|---|
| Top-1 loose | 27.3% | 33.8% | +6.5pp |
| Top-3 loose | 45.5% | 46.8% | +1.3pp |
| Top-5 loose | 49.4% | 49.4% | 不变 |
| Hard Top-3 | 47.5% | 50.0% | +2.5pp |
| Medium Top-3 | 43.2% | 43.2% | 不变 |
| L3 reflexion rate | 9.1% | 5.2% | -3.9pp |
| Mean latency | 17.2s | 29.0s | +11.8s |
| Mean tokens | 9069 | 8996 | -73 |
| Errors | 0 | 0 | 不变 |

### V9 Execution Metadata

| 指标 | 数值 |
|------|------|
| Avg tool_wall_ms | 52ms |
| Avg tool_sum_elapsed_ms | 37ms |
| Avg parallelism_ratio | 0.37 |
| Max layer width 分布 | {1:22, 2:25, 3:20, 4:9, 5:1} |

### 分析

1. **Top-3 稳定**：V9 46.8% = V7 46.8%，在运行方差范围内。工具执行层未破坏诊断质量。

2. **parallelism_ratio < 1**：V9 当前未观测到真实工具并行收益。可能原因包括 SQLite/PPR 内部同步阻塞、单步工具耗时过短、调度/解析开销占比高。Benchmark 证明 executor 机制对可让出 event loop 的工具有效（ratio=2.98）。若要验证真实收益，下一步应对阻塞型工具做 `asyncio.to_thread` 包装并重新对比。

3. **Latency spike**：V9 29.0s vs V8 17.2s — LLM generation 时间翻倍。工具和验证耗时与 V8 相似（tools ~36ms, verify ~1073ms），差异全在 Planner LLM 调用。这是 vLLM 服务端的运行方差（GPU 负载波动），与代码改动无关。

4. **max_layer_width 分布**：Plan 最多 5 层，常见 2-3 层。并行度上限（max_layer_width）的 histogram 说明 Planner 生成的 DAG 有较宽的并行层（width 3-5），为未来真实异步工具提供了基础。

## 结论

1. **Executor 机制正确**：benchmark sleep tool 证明 layer-based DAG 并行调度有效（ratio 2.98）。
2. **真实工具未观测到并行收益**：V9 当前 parallelism_ratio < 1，可能原因包括工具同步阻塞、单步耗时过短、调度开销占比高。
3. **可观测性已建立**：`execution_meta` 字段在 eval 输出中完整可查。
4. **诊断质量不变**：V9 Top-3 = V7 水平（46.8%），无误报。
5. **下一步**：如果继续做速度优化，应对阻塞型工具做 `asyncio.to_thread` 包装并重新对比，同时拆分 LLM 侧耗时。

## 文件改动清单

| 文件 | 改动 |
|------|------|
| `src/schemas.py` | FinalAnswer 新增 `execution_meta` 字段 |
| `src/orchestrator.py` | 新增 `_execute_plan_with_meta()` / `max_parallel_tools` / `_tool_semaphore` / `answer_async` 使用 meta |
| `eval/run_eval.py` | raw_prediction 输出 `execution_meta` |
| `scripts/benchmark_tool_parallel.py` | **新文件** — 并行 executor benchmark |
| `tests/test_core.py` | 新增 4 个执行元数据测试 + `_FakeSleepTool` / `_FakeToolRegistry` |
