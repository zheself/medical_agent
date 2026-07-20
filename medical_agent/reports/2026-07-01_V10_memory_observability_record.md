# V10a Memory Observability 工作记录

日期: 2026-07-01

## 背景

V8/V9 后系统已有 working / episodic / semantic 三层 memory，但缺少统一观测：
- 每次回答用了哪些 memory？
- episodic 检索到多少条？
- 注入 prompt 的 memory 有多长？
- eval 是否发生 memory scope 混用？
- 关闭 memory 后行为是否真的生效？

V10a 只做可观测性，不改变 memory 检索策略和诊断逻辑。

## V10 定位

V10 是项目从“已有 memory 模块”走向“可控 Agent Memory 系统”的阶段。目标不是立刻提升 CMB-Clin Top-K，而是把 memory 的使用边界、注入规模、scope 隔离和后续 gating/ablation 所需指标先工程化。

本阶段将 memory 明确拆成两类：

- **Working memory**：当前会话/当前任务状态，属于 Agent 运行时上下文，默认始终参与 Planner。
- **Long-term memory**：episodic / semantic memory，属于跨轮次、跨会话可检索记忆，受 `enable_memory_injection` 控制，也是后续 V10b gating 的主要对象。

因此，V10a 的验收标准不是准确率提升，而是：

- 每次回答都能追踪 memory scope 和注入规模；
- 能区分 working memory 与 long-term memory；
- 能验证 eval item / session / user 的隔离边界；
- 能为 V10b memory gating 和 memory ablation 提供稳定元数据。

## 改动

### memory_meta 语义

- **working**：短期会话状态，始终参与 Planner。不受 `enable_memory_injection` 控制。
- **episodic / semantic**：long-term memory，由 `enable_memory_injection` 控制注入。
- **injection.long_term_enabled**：明确指 episodic/semantic 是否注入，不等于 working 是否运行。

### memory_meta 结构

```json
{
  "scope": {
    "user_id": "eval_cmb_4_4",
    "session_id": "",
    "memory_key": "eval_cmb_4_4"
  },
  "working": {
    "enabled": true,
    "context_chars": 732,
    "turn_count": 0
  },
  "episodic": {
    "enabled": true,
    "retrieved_count": 0,
    "injected_count": 0,
    "context_chars": 0
  },
  "semantic": {
    "enabled": false,
    "retrieved_count": 0,
    "context_chars": 0
  },
  "injection": {
    "long_term_enabled": true,
    "long_term_context_chars": 0,
    "working_context_chars": 732,
    "total_context_chars_with_working": 732,
    "total_retrieved_count": 0
  }
}
```

### 文件改动

| 文件 | 改动 |
|------|------|
| `src/schemas.py` | FinalAnswer 新增 `memory_meta: Dict` |
| `src/orchestrator.py` | answer_async 中收集 working/episodic/injection 元数据，写入 FinalAnswer |
| `eval/run_eval.py` | raw_prediction 输出 `memory_meta` |
| `tests/test_core.py` | 新增 4 个 memory meta 测试 |

### 测试

**76/76 全部通过**。

新增测试：
- `test_memory_meta_exported_in_final_answer` — answer_async 输出含 memory_meta
- `test_memory_meta_disabled_when_memory_injection_off` — 关闭 injection 后 enabled=False，episodic=0
- `test_memory_meta_scope_is_session_specific` — 不同 session 的 scope 不同
- `test_memory_meta_scope_includes_user_id` — eval 隔离 user_id=f"eval_{item_id}"

### Mock eval 验证

```bash
python -m eval.run_eval --backend mock --num-items 5 \
  --output-dir eval_results/v10_memory_observability_mock
```

5/5 成功。raw_predictions 全部含 `memory_meta`，字段完整。

## 结论

V10a 建立了 memory 可观测基础：
- 每次回答的 memory 使用情况可追溯
- scope 隔离可验证
- injection 开/关行为可对比
- 不改变任何诊断逻辑、Planner prompt 或 memory 检索策略

## 下一步

V10b memory gating 只作用于 long-term memory，不动 working memory。计划包括 relevance score、filtered_count、gating threshold 和 memory ablation，用 V10a 的 `memory_meta` 作为观测基础。
