# V10c Memory Benchmark 工作记录

日期：2026-07-15

## 定位

V10a 建立 Memory 可观测性，V10b 增加了规则相关性门控。两阶段主要验证机制，普通 CMB-Clin 评测由于每题使用独立用户，无法衡量跨会话长期记忆的真实价值。

V10c 建立独立的多会话 Episodic Memory benchmark，并修正记忆写入生命周期。目标是回答以下问题：

1. 相关历史能否被检索并注入 Planner？
2. 不相关或过时记忆会引入多少噪声？
3. 用户之间是否存在记忆泄漏？
4. 当前 rule gate 在压缩上下文时损失多少关键事实？
5. 真实 Qwen3-8B 是否实际利用注入的历史事实？

本阶段仍使用 `MockEmbedder`，不把结果表述为真实 dense retrieval 的最终性能。V11 将替换为 BGE-M3。

## 核心修复

### Episodic 写入内容

`_async_write_episode()` 原先把 `plan.speculative_prefetch` 写入 `Episode.diagnoses`。该字段只是预取候选，不等于最终诊断。

V10c 改为写入 `final_answer.diagnoses`，并增加回归测试验证 Episode 与最终回答一致。

### 可等待的后台写入

新增 `MedicalAgentOrchestrator.flush_memory_writes()`：

- 多轮评测可在 setup turn 后等待写入完成；
- 后台任务异常被收集并由 flush 明确抛出；
- 避免下一轮检索早于上一轮写入的竞态；
- 避免后台异常只显示 `Task exception was never retrieved`。

### 检索可观测性

`memory_meta.episodic` 新增：

- `retrieved_episode_ids`
- `injected_episode_ids`

Gate record 同时记录 `episode_id`。因此评测可以计算 Recall、MRR、NDCG、旧记忆误注入和跨用户泄漏，而不再只有 count。

### 评测存储隔离

每个配置使用独立的内存 Episodic/Semantic store：

- 不读取或写入主数据库中的用户历史；
- L3 Reflexion 不向主 failure-case 表写入评测副产物；
- 场景使用唯一 user ID；
- 不修改现有 CMB-Clin 的逐 item 隔离逻辑。

## 数据集

新增 `data/eval_memory_scenarios.jsonl`，共 60 条，六类各 10 条：

| 类别 | 目标 |
|---|---|
| allergy | 药物过敏等 safety-critical memory |
| chronic | 跨会话慢病史 |
| medication | 当前/长期用药 |
| history | 既往诊断和检查结果 |
| temporal | 新事实应优先于冲突的旧事实 |
| irrelevant | 当前问题不需要任何长期记忆 |

每个有 gold memory 的场景包含：

- 1 条相关目标记忆；
- 1-2 条同用户干扰记忆；
- 1 条其他用户的相似记忆，用于验证 scope 隔离；
- expected / forbidden memory IDs；
- 部分场景包含回答必须覆盖的事实词。

数据可由 `scripts/build_memory_eval_dataset.py` 确定性重建。

## 评测配置

| 配置 | Long-term retrieval | Gate | 注入策略 |
|---|---:|---:|---|
| no_long_term | 关闭 | 关闭 | 不注入 |
| long_term_raw | 开启 | 关闭 | top-k 全部注入 |
| long_term_rule_gate | 开启 | 开启 | V10b 词表规则过滤 |

运行命令：

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python -u -m eval.run_memory_eval \
  --backend vllm \
  --llm-endpoint http://127.0.0.1:8001/v1 \
  --llm-model Qwen3-8B \
  --output-dir eval_results/memory_v10c_vllm
```

## 指标

- Retrieval Recall：有 gold memory 的场景是否召回目标 Episode。
- Injection Recall：目标 Episode 是否通过 gate 并进入 Planner。
- MRR / NDCG@5：相关记忆排序质量。
- Critical Episodic Injection Recall：allergy + chronic 目标 Episode 通过普通 gate 的比例。
- Critical Profile Recall：过敏/慢病关键事实通过冷启动 PatientProfile bypass 保留的比例。
- Irrelevant Injected：每题平均注入的非 gold 记忆数。
- Forbidden Injection Rate：注入旧冲突记忆或纯干扰记忆的场景比例。
- Cross-user Leakage Rate：检索到其他用户 Episode 的场景比例。
- Answer Required Recall：仅在定义了 `must_include` 的场景上计算回答事实覆盖率。
- Latency / context chars：性能与上下文成本。

## 测试

`python tests/run_all.py`：**90/90 通过**。

V10c 新增覆盖：

- 最终诊断写入 Episode；
- flush 等待后台写入；
- 后台写入错误显式上报；
- 同用户跨 session 可见；
- 跨用户零泄漏；
- gate record 保留 episode ID；
- Recall/MRR/forbidden 指标；
- 60 条场景数量、唯一性和类别完整性。

## 结果

真实 Qwen3-8B，60 条 × 3 配置，共 **180/180 成功，0 error**。

| 指标 | No long-term | Raw memory | Rule gate |
|---|---:|---:|---:|
| Retrieval Recall | 0% | 100% | 100% |
| Episodic Injection Recall | 0% | 100% | **28%** |
| Critical Profile Recall | 0% | **100%** | **100%** |
| Answer Required Recall | 46% | **92%** | 76% |
| Answer Constraint Pass | 46% | **90%** | 74% |
| Avg irrelevant injected | 0.00 | 2.00 | **0.18** |
| Forbidden Injection Rate | 0% | 33.3% | 15.0% |
| Cross-user Leakage Rate | **0%** | **0%** | **0%** |
| Avg episodic context chars | 0 | 374 | **51** |
| Avg latency | 8.48s | 10.27s | 9.05s |

Rule gate 相对 Raw Memory：

- Episodic 上下文减少约 **86.3%**；
- 平均无关注入减少约 **90.8%**；
- 延迟减少约 **1.21s**；
- 但 Episodic Injection Recall 从 100% 降到 28%；
- Answer Constraint Pass 从 90% 降到 74%。

### 类别结果

| 类别 | Rule Gate Injection Recall | Rule Gate Answer Constraint Pass | 解释 |
|---|---:|---:|---|
| allergy | 50% | 100% | 常见过敏原通过 critical profile bypass 保留 |
| chronic | 0% | 100% | 慢病通过 critical profile bypass 保留 |
| history | 10% | 100% | 部分问题本身含诊断词，回答指标受模型先验影响 |
| medication | 0% | **10%** | 规则 gate 无法由症状查询关联到历史药物 |
| temporal | 80% | 60% | 新旧记录同时进入时缺少显式冲突消解 |
| irrelevant | N/A | N/A | 0 条注入，纯干扰过滤成功 |

### 关键发现

1. **Memory 对真实回答有效。** Raw Memory 将 Answer Constraint Pass 从 46% 提升到 90%，其中 medication 从 0% 提升到 80%，证明模型会使用注入的历史，而不只是 metadata 链路生效。
2. **直接注入的噪声代价明显。** Raw 配置每题平均注入 2 条非 gold 记忆；纯干扰和 temporal 场景的 forbidden injection 都是 100%。
3. **V10b Rule Gate 过度过滤。** 它显著压缩 prompt，但普通 episodic recall 只有 28%，用药场景完全无法保留目标历史。
4. **Critical bypass 是必要的安全层。** Rule Gate 的 allergy/chronic episodic recall 分别只有 50%/0%，但 critical profile recall 为 100%，因此回答仍保持 100% constraint pass。
5. **Temporal conflict 仍未解决。** Raw 和 Rule Gate 的 temporal constraint pass 分别只有 70% 和 60%。只做相关性过滤不足以判断新旧事实，应增加时间状态和 supersede 机制。
6. **用户隔离正确。** 三配置 cross-user leakage 均为 0%。

### Case 抽查

- `medication_01`：Raw Memory 识别历史二甲双胍并关联腹泻/恶心；Rule Gate 注入 0 条，只能泛化为“某种药物副作用”。
- `temporal_02`：新记录为“已停用二甲双胍”，但 Raw/Rule 都同时看到旧记录，回答转为“存在矛盾、需要确认”，未可靠采用最新状态。
- `temporal_07`：最新阿司匹林停药记录被正确采用，回答明确“目前不再服用”。
- `irrelevant_01`：Raw 注入 2 条无关慢病记录；Rule Gate 注入 0 条，回答内容未受污染。

## 边界与限制

- 受控语料每个用户只有 2-3 条候选，且仍使用 `MockEmbedder`。100% Retrieval Recall 只表示当前 benchmark 链路可工作，不代表真实 dense retrieval 上限。
- Answer 指标采用字符串约束，虽然 temporal 增加了“不确定/矛盾”失败词，仍不能替代医生标注或独立模型 judge。
- 当前仅单次 Qwen3-8B 运行；回答指标存在生成方差。
- Critical fact 提取仍基于固定过敏原和慢病词表，覆盖面有限。
- V10c 使用直接 seed 的受控 Episode 测检索；端到端写入生命周期由单元测试覆盖，尚未评估 LLM 自动抽取事实的准确率。

## 下一步

V11 使用真实 BGE-M3 embedding，比较 dense-only、dense hybrid 和 dense + reranker。优先解决 medication 语义召回和 temporal conflict；V10c 的 rule gate 作为低成本基线保留，不继续扩充词表以避免对 benchmark 过拟合。
