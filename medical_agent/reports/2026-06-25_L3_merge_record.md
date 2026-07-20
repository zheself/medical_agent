# L3 Merge Guard 工作记录

日期: 2026-06-25

## 背景

P0 2×2 PPR×L3 消融发现 L3 reflexion 是比 PPR 更强的性能杠杆（PPR OFF 时 L3 带来 +4.5pp Top-3），但 case analysis 揭示 L3 当前行为是"重写而非修正"——每次触发都完全替换诊断列表，而非在原诊断基础上修改。

### L3 case analysis 发现（16 条，PPR ON + PPR OFF 合并）

| L3 行为 | 数量 | 描述 |
|----------|------|------|
| Empty rescue → hit | 5 | L2 空预测，L3 补上正确答案 |
| Wrong→correct fix | 2 | L2 错误，L3 修正 |
| **Correct wipe** | **4** | L2 已有正确答案，L3 完全替换为错误 |
| Empty rescue → miss | 4 | L2 空预测，L3 也未能命中 |
| No change | 2 | L3 输出与 L2 相同 |
| Rank push-down | 1 | L3 把正确诊断从 Rank-2 推到 Rank-5 |

**关键发现**：L3 从不 merge——100% 的 case 都是全量重写。

## 实现：Diagnosis Merge Guard

### 设计

不修改 L3 LLM prompt，在 `GradedVerifierOrchestrator.verify()` 的两个 L3 返回路径上，对 `corrected_answer` 的 `possible_diagnoses` 做 merge：

```python
def merge_l3_diagnoses(original_diagnoses, l3_diagnoses, l3_content, max_candidates=5):
    # 原诊为空 → 使用 L3 候选
    # 原诊非空 → 保留原 Top-3，追加 L3 新候选（去重截断到 5）
    # L3 无候选 → 保留原诊断
    # 在 content 末尾追加一行合并候选摘要
```

### 改动文件

| 文件 | 改动 |
|------|------|
| `src/verifiers/l3_reflexion.py` | 新增 `merge_l3_diagnoses()` + 两处 L3 返回点调用 + Tuple import |
| `src/verifiers/l1_rule_verifier.py` | draft_answer 类型 guard（防止 citation string bug） |
| `src/orchestrator.py` | `_synthesize_draft` 返回后 dict 类型 guard（防止 citation string bug） |
| `src/orchestrator.py` | PPR OFF 过滤逻辑 bug fix：先记录 removed_ids 再过滤 |
| `tests/test_core.py` | 新增 5 个 merge guard 测试 + 3 个 citation/L1 guard 测试 + 1 个 PPR OFF 依赖清理测试 |

### 测试

**57/57 全部通过**。

新增测试：
- `test_merge_keeps_original_top3_when_l3_wipes` — correct wipe 防御
- `test_merge_uses_l3_when_original_empty` — empty rescue 保持
- `test_merge_keeps_original_when_l3_empty` — L3 空候选不破坏
- `test_merge_dedup_and_truncate` — 去重+截断
- `test_merge_handles_non_list_inputs` — 类型安全
- `test_l1_handles_string_draft_answer` — citation string guard
- `test_l3_citation_string_normalized` / `test_l3_citation_none_normalized` — L3 citation 规范化
- `test_ppr_off_sibling_dep_cleaned` — PPR OFF 依赖清理

## 验证

### 子集定向验证（14 条 L3 相关 case）

14/14 成功，0 error。L3 触发 2/14，均有 MERGE tag。merge guard 机制在真实链路中确认生效。

### V7 全量评测（77 条）

```bash
python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl --concurrency 1 --output-dir eval_results/cmb_clin_vllm_v7
```

**77/77 成功，0 error。**

### V7 vs V6 对照（75 条共同成功子集）

| 指标 | V6 | V7 | Δ |
|------|-----|-----|---|
| Top-1 loose | 26.7% | 32.0% | +5.3pp |
| Top-3 loose | 41.3% | **46.7%** | +5.4pp |
| Top-5 loose | 45.3% | **49.3%** | +4.0pp |
| Hard Top-3 | 42.5% | **50.0%** | +7.5pp |
| Medium Top-3 | 40.0% | 42.9% | +2.9pp |
| L3 trigger rate | 6.7% | 6.7% | 不变 |
| Mean latency | 17.1s | 16.8s | -0.3s |
| Mean tokens | 8248 | 8927 | +679 |
| Avg diagnoses | 4.1 | 4.4 | +0.3 |
| Errors | 2 | **0** | -2 |

### V7 L3 行为

L3 触发 5/77（6.7%），全部有 MERGE tag：
- 5/5 均为 "both miss" — L3 和 L2 都未能命中 gold
- 本轮 correct wipe case 均停在 L2 未触发 L3
- merge guard 机制确认安全，未破坏 empty rescue

### 改善归因

V6→V7 改善（+5.4pp Top-3）不独占功于 merge guard。本轮代码变化还包括：
- Citation string bug 修复（3 guards）
- PPR OFF 依赖清理 bug fix
- LLM 运行间方差

merge guard 的主要价值是**防御 L3 correct wipe**——本轮 L3 未触发 correct wipe case，所以防御未激活。但当 correct wipe 发生时，merge guard 保证原 Top-3 不被覆盖。

## V7 里程碑

V7 = V6 baseline + merge guard + citation guards + PPR OFF fix。

| 指标 | 数值 | 历史排名 |
|------|------|----------|
| Top-3 loose | 46.7% | **最高** |
| Hard Top-3 | 50.0% | 持平 PPR OFF L3 |
| Errors | 0 | 持平 V4 |
| L3 merge tag | 5/5 (100%) | ✅ |

## 工程决策

- **L3 merge guard 保留为默认行为**。机制安全，无副作用。
- **V7 作为新的诚实基线**。替代 V6。
- **L3 prompt/trigger 暂不改**。merge guard 是第一阶段收口。
- **下一步 L3 优化**：改 L3 prompt（要求保留原诊断而非重写）+ trigger 策略优化（减少 both-miss 浪费）。

## 测试结果（V7 收口时）

```bash
$ python tests/run_all.py
==================================================
[1/3] tests/test_graphrag.py — GraphRAG + PPR + 社区检测 + 摘要
  通过 12 / 12

[2/3] tests/test_core.py — 核心链路 + Planner + Verifier + Merge Guard
  通过 37 / 37

[3/3] tests/test_database.py — SQLite .db 后端
  通过 8 / 8

============================================================
  ✅ 全部测试通过 (57/57)
```

## 验证指引

1. **确认 merge guard**: `grep -n "merge_l3_diagnoses" src/verifiers/l3_reflexion.py`
2. **跑 tests**: `python tests/run_all.py` 应 57/57 通过
3. **确认 V7**: `cat eval_results/cmb_clin_vllm_v7/report.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Top3={d[\"capability\"][\"top3_hit_rate\"]:.1%} HardTop3={d[\"by_difficulty\"][\"hard\"][\"top3_loose\"]:.1%}')"`
4. **确认 L3 merge**: `grep -c "综合候选诊断" eval_results/cmb_clin_vllm_v7/raw_predictions.jsonl` 应 ≥ L3 触发数
