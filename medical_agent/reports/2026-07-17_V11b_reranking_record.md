# V11b Episodic Memory Reranking 工作记录

日期：2026-07-17

## 定位

V11a 已将 Episodic Memory 接入真实 BGE-M3，但原 60 条 benchmark 每个用户只有 2-3 条候选，无法充分评价 reranker。V11b 构造 10-12 条候选的 hard-negative benchmark，使用独立 dev/test split 调参，并接入 `BAAI/bge-reranker-v2-m3` CrossEncoder。目标是提高 Top-1 精度、减少无关记忆和过期事实注入，同时量化召回与延迟代价。

V11b 不修改诊断基线，不训练模型，也不使用 test 集选择权重或阈值。

## 实现

- 新增 `scripts/build_memory_reranking_dataset.py`：从 V10c 场景生成 60 条 V11b 场景，每题 12 个唯一 target 候选。
  - dev 24 条、test 36 条；六类场景均按 4/6 分层切分。
  - hard negative 优先来自同类别，再用跨类别候选补齐。
  - 构造时跳过同摘要副本；修复了早期版本中 22 条场景存在复制正例、指标虚高的问题。
- 新增 `src/memory/rerankers.py`：`Reranker`、测试用 `IdentityReranker` 和延迟加载的 `CrossEncoderReranker`。
  - CUDA 默认 FP16；本地模型和 mock 路径不增加导入时依赖。
  - reranker 返回分数数量必须与候选数一致，否则显式失败，禁止 dense/rerank 分数静默混排。
- `EpisodicMemory` 支持 `reranker_candidate_k`、score threshold 和可配置 `retrieval_weights`。
- Factory、Orchestrator 和 `memory_meta` 全链路导出模型、权重、candidate_k、threshold、base/reranker score。
- `eval/run_memory_reranking_eval.py`：
  - dev 网格搜索 hybrid 权重并校准 abstention threshold；
  - test 冻结比较 dense、tuned hybrid、dense rerank、hybrid rerank；
  - 指标包含 Recall@1/@3/@5、MRR、NDCG@5、temporal forbidden@1/@5、irrelevant injection、abstention、leakage 和 warmup 后 p50/p95 延迟；
  - 调参按 Orchestrator 实际注入 Top-5 计算 forbidden，而不是只看 Top-1。
- `eval/run_memory_eval.py` 新增 reranker、hybrid 权重和 `--split dev|test` 参数，可复现冻结配置的下游 Qwen 对照。

## 数据与环境

- 数据：`data/eval_memory_reranking_v11b.jsonl`，60 条，24 dev / 36 test，每条 12 个唯一候选。
- Embedder：本地 `BAAI/bge-m3`，FP16，GPU0。
- Reranker：本地 `BAAI/bge-reranker-v2-m3`，FP16，GPU0，权重 2,271,071,852 bytes。
- Python：`/mnt/sda/ubuntu/anaconda3/envs/cjz_opd`。
- V11b 未新增 Python 包；复用 V11a 已验证的 `sentence-transformers==3.4.1`。
- 模型目录和 `eval_results/` 均由 `.gitignore` 排除。

模型 smoke：相关用药文本得分 `0.0007825`，无关病史得分 `0.0001535`；参数 dtype 为 FP16。

## Dev 冻结参数

最终配置只由 24 条 dev 选择：

```text
hybrid weights = similarity 0.8, importance 0.0, time 0.3, frequency 0.0
candidate_k    = 8
threshold      = 0.00022339820861816406
```

`candidate_k=12` 作为补充消融，其 dev selection score 为 0.354，低于 k=8 的 0.692，因此被拒绝；没有依据 test 结果改参。

## Held-out 检索结果

36 条 test 中 30 条有正例、6 条为 irrelevant；temporal 6 条。

| 指标 | BGE dense | Tuned hybrid | Dense rerank | Hybrid rerank |
|---|---:|---:|---:|---:|
| Recall@1 | 30.0% | 36.7% | 33.3% | **50.0%** |
| Recall@3 | **83.3%** | 53.3% | 63.3% | 66.7% |
| Recall@5 | **83.3%** | 66.7% | 63.3% | 73.3% |
| MRR | 0.559 | 0.507 | 0.489 | **0.605** |
| NDCG@5 | 0.615 | 0.515 | 0.523 | **0.634** |
| Temporal forbidden@1 | 83.3% | **0%** | 83.3% | 16.7% |
| Temporal forbidden@5 | 100% | **0%** | 100% | 16.7% |
| Irrelevant injection | 100% | 100% | **0%** | **0%** |
| Cross-user leakage | 0% | 0% | 0% | 0% |
| P95 total retrieval | 31.8ms | 31.8ms | 76.2ms | 76.2ms |

相对 dense，hybrid rerank 的 Recall@1 `+20.0pp`、MRR `+0.046`，满足预设 go/no-go；同时显著改善 temporal 和 irrelevant 注入。代价是 Recall@3 `-16.7pp`、Recall@5 `-10.0pp`，主要来自 hybrid first stage 的 Top-8 候选截断和全局 threshold。

10,000 次 paired bootstrap：Recall@1 差值 95% CI `[+3.3pp, +36.7pp]`；MRR 差值 CI `[-0.089, +0.185]`。因此可以认为 Top-1 精度改善有信号，但不能声称 MRR 已在总体分布上稳定提升。

## Qwen3-8B 下游结果

检索 go/no-go 通过后，在同一 36 条 test 上运行 dense 与冻结 hybrid rerank。两次均为 36/36 success、0 error。首次 dense 尝试因 vLLM 被关闭而产生的 36 条 connection error 已被服务恢复后的成功结果覆盖，不计入对照。

| 指标 | Dense | Hybrid rerank | 变化 |
|---|---:|---:|---:|
| Answer constraint pass | 80.0% | 83.3% | +3.3pp |
| Temporal constraint pass | 50.0% | **100%** | +50.0pp |
| Medication constraint pass | 50.0% | 16.7% | -33.3pp |
| Avg injected memories | 5.00 | **2.75** | -45.0% |
| Avg irrelevant injected | 4.31 | **2.14** | -50.3% |
| Avg long-term context | 1538.9 chars | **1040.3 chars** | -32.4% |
| Critical profile recall | 100% | 100% | 0 |
| Cross-user leakage | 0% | 0% | 0 |
| Avg end-to-end latency | 20.7s | 18.4s | -2.3s* |

`*` 完整 Agent 延迟主要由 vLLM 决定；36ms reranker 开销无法解释 2.3s 变化，该变化按运行方差处理。

逐例配对共有 3 条 temporal 从 fail 变 pass、2 条 medication 从 pass 变 fail，净增 1/30 条受约束场景。这个小样本差异不具备统计确定性。`medication_08` 是明确召回损失：threshold 后只保留 1 条错误候选，目标记忆被过滤；`medication_03` 的目标记忆仍在 Top-5，但生成结果变差，更可能是候选顺序与 LLM 运行方差共同作用。

## 验证

```bash
python tests/run_all.py
```

结果：**103/103 通过**（P0 12、core 83、database 8）。

额外验证：

- hard-negative 数据：60 条、24/36 split、每题 12 个候选、0 重复摘要、Episode ID 全局唯一；
- mock reranker 和 mock 下游链路成功；
- real model safetensors 可读取 393 个 tensor；
- dense/hybrid 基线不计 reranker latency，Top-5 forbidden 口径有回归测试；
- `git diff --check` 通过。

## 结论

1. V11b 完成了从 hard-negative 数据、dev/test 隔离、CrossEncoder、阈值校准到 Qwen 下游验证的闭环。
2. hybrid rerank 是明显的 precision/selectivity 优化：Top-1、temporal 冲突和 irrelevant abstention 均改善，检索 P95 仍低于 100ms。
3. 它不是无条件替代 dense 的默认方案：Top-3/Top-5 recall 和 medication 下游表现下降。当前应保留为可配置模式，而不是直接改默认值。
4. 下阶段优先做结构化 Temporal Memory（status、provenance、superseded_by），从数据语义层解决新旧事实冲突；同时可研究按记忆类型设置 threshold，避免全局阈值过滤 medication/critical 记忆。

## 边界

- 数据是受控合成 benchmark，候选池仅 12 条，不能外推到生产规模。
- CrossEncoder 未在本项目数据上训练；dev 只用于权重和阈值选择。
- critical profile 是独立 bypass 路径，部分 allergy/chronic 回答不完全依赖 reranked episodic Top-5。
- 两次 Qwen 生成不是确定性 paired decoding；回答差异需结合检索记录解释，不能只看总分。
