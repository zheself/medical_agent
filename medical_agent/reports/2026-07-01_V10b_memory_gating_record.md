# V10b Long-term Memory Gating 工作记录

日期: 2026-07-01

## 背景

V10a 建立了 memory_meta。V10b 在 long-term memory（episodic/semantic）注入 Planner 前增加 relevance gating，过滤低相关记忆，避免 prompt 污染。

## 目标

只对 episodic/semantic long-term memory 做 gating。Working memory 不受影响。不承诺 Top-K 提升。

## 设计

- **Working memory**：不 gating（当前会话状态始终参与 Planner）
- **Episodic/semantic**：rule-based relevance scoring → 低于阈值过滤
- **默认关闭**：`enable_memory_gating=False`，保持原行为

### scoring 规则

纯规则，不需要 GPU：
1. 从 query 和 memory 中提取医学实体词（基于 MOCK_ENTITY_LEXICON + 常见症状词表）
2. `base_score = overlap / max(1, len(query_terms))`
3. diagnosis overlap bonus: +0.4
4. summary overlap bonus: +0.2
5. score ≥ threshold → keep

### memory_meta.episodic 新增字段

```json
"episodic": {
  "retrieved_count": 5,
  "injected_count": 2,
  "filtered_count": 3,
  "gating_enabled": true,
  "gate_threshold": 0.2,
  "gate_records": [{"keep": true, "score": 0.5, "reason": "term_overlap", "matched_terms": ["发热"]}]
}
```

## 改动文件

| 文件 | 改动 |
|------|------|
| `src/memory/memory_gate.py` | **新文件** — score_memory_relevance() + gate_episodic_hints() |
| `src/orchestrator.py` | 新增 enable_memory_gating + memory_gate_threshold 参数；接入 gating 流程；更新 memory_meta |
| `src/factory.py` | build_system 新增 enable_memory_gating/memory_gate_threshold 参数 |
| `eval/run_eval.py` | 新增 --enable-memory-gating / --memory-gate-threshold flags |
| `scripts/run_memory_ablation.py` | **新文件** — mock/db memory ablation |
| `tests/test_core.py` | 新增 5 个 gate 测试 |

## 测试

**81/81 全部通过**。

新增：
- `test_memory_gate_keeps_relevant_episode` — 相关记忆保持
- `test_memory_gate_filters_irrelevant_episode` — 不相关过滤
- `test_memory_gate_disabled_keeps_all` — disabled 全保留
- `test_memory_gate_empty_query` — 空查询不保留
- `test_gate_records_in_memory_meta` — 默认 metadata 正确

## Mock eval

Mock 模式无 episodic 数据，所有配置 retrieved=0。gating metadata 正确记录。

Ablation 主要验证 metadata 和开关链路正确；真正的过滤收益需要 V10c memory-specific cases（构造有 episodic history 的测试集）。

```bash
python scripts/run_memory_ablation.py --backend mock --num-items 5
```

集成测试（`test_memory_gating_orchestrator_filters_irrelevant`）用 fake episodic memory 验证了完整链路：查询"头痛发烧" → 脑膜炎(keep) + 糖尿病(filtered) → 注入 1/2。

## 结论

V10b 建立了 long-term memory relevance gating 机制：
- Rule-based scoring，不需要 GPU
- 默认关闭（向后兼容）
- Metadata 完整记录 kept/filtered/score/reason
- Working memory 不受影响

下一步 V10c：memory-specific eval cases（same-user carryover, cross-user isolation, allergy memory）。
