# V11a BGE-M3 Episodic Retrieval 工作记录

日期：2026-07-16

## 定位

V10c 证明跨会话记忆对 Qwen3-8B 的回答有价值，但检索端仍为规则化 `MockEmbedder`。V11a 将 Episodic Memory 接入真实 `BAAI/bge-m3`，先单独测量检索排序；不在本阶段混入 reranker、时间冲突消解或下游生成评测。

## 实现

- 新增 `src/memory/embedders.py`：`MockEmbedder`、延迟加载的 `BGEEmbedder` 与 `create_embedder()`。
- `Episode` 和 SQLite `episodes` 表新增 `embedding_model`、`embedding_dim`；旧向量模型或维度不兼容时，在检索前自动重建，避免 Mock/BGE 向量混用。
- `EpisodicMemory` 新增 `retrieval_mode=dense|hybrid`：
  - dense：只使用 cosine similarity；
  - hybrid：similarity + importance + time decay + frequency。
- 检索结果在 `memory_meta.episodic` 输出模型、模式、排序分数和分数组成。
- 新增 `eval/run_memory_retrieval_eval.py`，使用 V10c 60 条场景做无 LLM 的检索对照，输出 Recall@1/@2/@5、MRR、NDCG@5、forbidden@1/@5 和跨用户泄漏率。
- 新增 `scripts/build_memory_embeddings.py`：显式复制数据库或指定 `--in-place` 后重建向量，避免默认修改主库。

## 环境与模型

- 环境：`/mnt/sda/ubuntu/anaconda3/envs/cjz_opd`。
- 新增依赖：`sentence-transformers==3.4.1`，使用 `--no-deps` 安装。dry-run 确认不会变更现有 torch、transformers、tokenizers、numpy、scipy、scikit-learn 或 vLLM 依赖。
- 模型：`BAAI/bge-m3`，本地目录 `data/models/bge-m3/`，FP16、GPU0、batch size 8。
- 主权重为 2.27GB，仅本地使用，已加入 `.gitignore`。

## 验证

`cjz_opd` 环境下：

```bash
python tests/run_all.py
```

结果：**96/96 通过**。

本地 BGE smoke test：两条中文医疗文本正常编码，输出维度为 **1024**。

端到端 smoke：`run_memory_eval.py` 使用 BGE-M3 dense + mock Agent 跑 2 条 allergy 场景，**2/2 success，0 error**；两条目标 Episode 均被检索并注入（retrieval/injection recall 均为 100%），确认 `Orchestrator -> memory_meta -> Planner` 链路使用真实 embedding 正常。

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/sda/ubuntu/anaconda3/envs/cjz_opd/bin/python \
  -m eval.run_memory_retrieval_eval \
  --configs bge_dense bge_hybrid \
  --model-name data/models/bge-m3 --device cuda --batch-size 8 \
  --output-dir eval_results/memory_v11_bge
```

## BGE-M3 检索结果

60 条 V10c 场景，均为同用户检索；`Recall@5=100%` 受每个用户仅 2-3 条候选的受控数据集限制，主要看 Top-1/Top-2 和 MRR。

| 指标 | BGE dense | BGE hybrid |
|---|---:|---:|
| Recall@1 | **72%** | 66% |
| Recall@2 | **94%** | 78% |
| MRR | **0.850** | 0.793 |
| NDCG@5 | **0.889** | 0.846 |
| Forbidden@1 | 30.0% | **16.7%** |
| Cross-user leakage | 0% | 0% |
| Avg query latency | 24.1ms | 22.5ms |

类别观察：

- dense 对 chronic（100% Top-1）、history（70%）和 medication（90%）更好；
- hybrid 对 temporal 明显有效（Top-1 20% -> 100%，forbidden@1 80% -> 0%），但固定 importance/time 权重严重压低了 chronic（100% -> 10%）和 history（70% -> 20%）；
- 两种模式都在 irrelevant 类检索到候选，说明检索层没有 no-memory abstention，仍需 gate/reranker 控制注入。

## 结论与下一步

1. BGE-M3 已可替代 MockEmbedder，真实 dense retrieval 在受控 benchmark 上具备可解释的 Top-K 指标和低毫秒级单查询延迟。
2. 当前固定 hybrid 权重不应作为默认配置：它改善 temporal 排序，却降低整体 MRR。权重必须在 dev split 调参，不能用同一 60 条场景直接选择。
3. V11b 优先项：dev/test split + cross-encoder reranker，先验证是否同时提升 history/chronic 和 temporal；随后把胜出的检索配置接回 `run_memory_eval.py`，做 Qwen 端到端回答约束评测。
4. V12 再引入结构化 `status/provenance/superseded_by`，解决 temporal 场景中旧事实仍进入 Top-5 的问题。

## 边界

- 本阶段没有 reranker，也没有新的 Qwen 生成评测；V10c 的回答约束结论不能直接外推到 BGE 配置。
- 该 benchmark 候选池很小，不能把 Recall@5=100% 当作真实大规模检索性能。
- `avg_query_ms` 不含模型首次加载和离线建索引时间。
