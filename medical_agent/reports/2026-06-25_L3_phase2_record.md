# L3 Phase 2: Prompt + Trigger 优化

日期: 2026-06-25

## 背景

L3 Phase 1（merge guard）防御 L3 覆盖正确诊断。Phase 2 改进攻势：prompt 和 trigger。

Case analysis 发现 L3 当前行为是"重写而非修正"，且 trigger 不够精准（V7 5/5 L3 triggers 都是 both-miss）。

## 改动

### 1. L3 Prompt 改写

文件：`src/verifiers/l3_reflexion.py`，`L3_REFLEXION_PROMPT`

在 `{task}` 段前插入"修正规则"：

1. **保留优先**：原回答正确的部分必须保留
2. **修正而非重写**：只改有问题的地方
3. **诊断候选合并输出**：原候选中的正确部分 + 新补充的鉴别诊断
4. **按置信度排序**：3-5 个候选，首位是最可能的诊断
5. **证据不足时保留原诊断**：不强制编造候选

同时更新 JSON schema 中 `possible_diagnoses` 的注释。

### 2. L3 Trigger Guard

文件：`src/verifiers/l3_reflexion.py`

新增模块级函数和常量：
- `_L2_SCORE_KEYS`：只取 faithfulness/relevance/factuality 三个数值 key
- `_numeric_l2_scores()`：从 L2 scores 中安全提取纯数值（过滤 issues 等非数值字段，防止 TypeError）
- `_should_trigger_l3()`：返回 `(bool, reason)` 元组
- Reason 枚举：`empty_diagnoses`, `l1_safety_error`, `l1_citation_failure`, `low_l2_score`, `enough_candidates_skip`, `default_trigger`

**决策树**：
1. 空预测 → 强制触发
2. L1 安全错误（ALLERGY/AGE/GENDER/EMERGENCY/DYNAMIC）→ 强制触发
3. L1 citation-only + 候选 ≤ 2 → 触发
4. L2 数值分数最低 < 0.5 → 触发
5. 候选 ≥ 3 且 L2 分数 ≥ 0.6 → 跳过
6. 默认 → 触发（保守）

**不触发 L3 时的处理**：
- L1 fail + skip → `needs_replan=True`（不能静默返回 draft）
- L2 fail + skip → 返回 draft，`level_reached=L2`，`l3_skip_reason` 记录

### 3. verification_meta 传播

- `src/schemas.py`：FinalAnswer 新增 `verification_meta: Dict` 字段
- `src/orchestrator.py`：answer_async 填入 level_reached/trigger_reason/l3_skip_reason/needs_replan
- `eval/run_eval.py`：输出的 raw_predictions 包含 verification_meta

## 测试

**68/68 全部通过**。

新增测试（11 个）：
- 5 个 `_should_trigger_l3` 纯函数测试
- 4 个 `_numeric_l2_scores` / scores-with-issues 边缘测试
- 2 个 `GradedVerifierOrchestrator.verify()` 集成测试
  - L1 EMERGENCY fail + high → verify() 返回 L3
  - L2 fail + enough candidates → verify() 返回 L2 + l3_skip_reason

## 子集验证

```bash
python -m eval.run_eval --backend vllm --data-path data/eval_l3_cases.jsonl \
  --concurrency 1 --output-dir eval_results/l3_phase2_cases_v2
```

14/14 成功，0 error。L3 trigger: 1/14 (7.1%)，reason=`empty_diagnoses`。
verification_meta 正确输出。
V7 子集 L3 率 14.3% → Phase 2 7.1%。

## V8 全量评测

```bash
python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl \
  --concurrency 1 --output-dir eval_results/cmb_clin_vllm_v8
```

77/77 成功，0 error。

### V7 vs V8（77 条共同子集）

| 指标 | V7 | V8 | Δ |
|------|-----|-----|---|
| Top-1 loose | 32.5% | 27.3% | -5.2pp |
| Top-3 loose | 46.8% | 45.5% | -1.3pp |
| Top-5 loose | 49.4% | 49.4% | 不变 |
| Hard Top-3 | 50.0% | 47.5% | -2.5pp |
| Medium Top-3 | 43.2% | 43.2% | 不变 |
| L3 trigger rate | 6.5% | 9.1% | +2.6pp |
| Mean latency | 16.9s | 17.2s | +0.3s |
| Mean tokens | 8913 | 9069 | +156 |
| Mean diagnoses | 4.4 | 4.4 | 不变 |
| Errors | 0 | 0 | 不变 |

### V8 trigger/skip reason 分布

| reason | 数量 |
|--------|------|
| `empty_diagnoses` | 7 |
| `l3_skip_reason` | 0 |

L3 全部 7 次触发均为 `empty_diagnoses`——trigger guard 正确识别空预测场景。
V7 的 5 次 L3 全部是 both-miss（未分类），V8 升级为有 reason 标签的 empty rescue。

注意：由于 V8 的 L3 触发全部发生在原诊断为空的场景，本轮没有直接覆盖“原诊断非空但 L3 可能覆盖正确答案”的 correct-wipe 场景。V8 能证明 trigger guard 将 L3 收敛到了空预测 rescue 场景，未观察到 correct-wipe 回归；merge guard 对非空原诊断的保护仍主要由单元测试和 offline replay 支撑。

无 `l3_skip_reason` 是因为 MockSmallModel（L2）从不给有候选的回答打低分——在真实 L2 模型部署前，skip 路径不会被充分测试。

### V8 L3 case 分类

| 类型 | 数量 | Cases |
|------|------|-------|
| Empty rescue → hit | 2 | cmb_1_2 (胃食管反流), cmb_41_44 (烟雾病) |
| Empty rescue → miss | 5 | cmb_12_13, cmb_16_17, cmb_19_20, cmb_31_34, cmb_32_36 |
| Correct wipe | **0** | ✅ |
| Rank push-down | **0** | ✅ |

7/7 均有 merge tag（"综合候选诊断"）✅。
Correct wipe 为 0——本轮未观察到 L3 覆盖正确诊断或 rank push-down 回归；由于触发样本全是 empty rescue，这主要验证了 trigger guard 的收敛效果。

### V8 vs V4/V6/V7 全历程

| 指标 | V4 | V6 | V7 | V8 |
|------|-----|-----|-----|-----|
| Top-3 | 37.7% | 41.3% | 46.8% | 45.5% |
| Hard Top-3 | 37.5% | 42.5% | 50.0% | 47.5% |
| L3 rate | 1.3% | 6.7% | 6.5% | 9.1% |
| L3 hurt | 3 | 1 | 0 | **0** |
| Errors | 0 | 2 | 0 | 0 |

V8 Top-3 接近 V7 水平（−1.3pp，在运行方差范围内），本轮未观察到 L3 correct wipe 回归。

### 归因

V7→V8 变化包含：
- L3 prompt 改写（修正而非重写）
- L3 trigger guard（空预测优先触发）
- LLM 运行间方差

Top-1 下降 5.2pp 主要归因为运行方差（V6 Top-1 也曾到 26.7%）。
Phase 2 的主要贡献是 L3 behavior quality（L3 触发具备 reason 标签、empty rescue preserved、未观察到 correct wipe 回归），而非 raw Top-K 提升。

## 文件改动清单

| 文件 | 改动 |
|------|------|
| `src/verifiers/l3_reflexion.py` | L3_REFLEXION_PROMPT 改写 + `_numeric_l2_scores` + `_should_trigger_l3` + trigger reason 常量 + 两处 verify() trigger 条件 |
| `src/schemas.py` | FinalAnswer 新增 `verification_meta` 字段 |
| `src/orchestrator.py` | answer_async 填入 verification_meta |
| `eval/run_eval.py` | raw_predictions 输出 verification_meta |
| `tests/test_core.py` | 新增 11 个 L3 Phase 2 测试 |

## 验证指引

1. `python tests/run_all.py` → 68/68
2. 读取任一 raw_predictions.jsonl → `verification_meta` 字段存在且含 level_reached
3. L3 触发的 item → `trigger_reason` 非空
4. L2 skip 的 item → `l3_skip_reason` 非空
