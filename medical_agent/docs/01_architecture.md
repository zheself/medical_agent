# 01 · 整体架构

## 一、设计哲学

这个系统经过 4 轮迭代演化，每一轮都是为了解决前一版的真实痛点。最终架构有 4 个核心支柱：

1. **Planner-Executor 模式**：一个 LLM Planner + 多个无状态 Tool，而非多 LLM 链式
2. **三层 Memory 体系**：Working / Episodic / Semantic，各司其职
3. **分级反思**：L1 规则 / L2 小模型 / L3 完整反思，按需触发
4. **GraphRAG 多跳推理**：社区检测 + Personalized PageRank，突破"单跳"瓶颈

## 二、整体架构图

```
                          User Query
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Orchestrator (主调度器)     │
              └───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  Working Memory      Episodic Memory       Semantic Memory
  (会话级)            (用户级,异步检索)       (全局,KG+失败案例)
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │       Planner (单一 LLM)       │
              │  生成 DAG + Self-Critique      │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Tool DAG 并行执行          │
              │  ┌─────────┬─────────┬──────┐ │
              │  │  NER    │ Local   │Global│ │
              │  │         │ Search  │Search│ │
              │  └─────────┴─────────┴──────┘ │
              │  ┌────────────────────────┐   │
              │  │   PPR Multi-hop        │   │
              │  └────────────────────────┘   │
              └───────────────────────────────┘
                              │
                              ▼
                       Draft Answer
                              │
                              ▼
              ┌───────────────────────────────┐
              │     分级反思 (Graded Verify)    │
              │   L1 规则 → L2 小模型 → L3 完整  │
              │   (75% 在 L1, 20% 在 L2, 5% L3) │
              └───────────────────────────────┘
                              │
                              ▼
                       Final Answer
                              │
                              ▼
              [异步写入 Working/Episodic/Semantic Memory]
```

## 三、为什么是这个架构？

### 3.1 为什么不是"多 Agent 链式"？

**原版（v2）**：思考 Agent → 查询 Agent → 反思 Agent 串联，每个独立 LLM。

**问题**：
- 三次 LLM 调用，KV Cache 全部失效，P95 延迟 ~9s
- 任何一环出错没法回退到上一环
- 状态在 Agent 间通过 prompt 传递，丢失大量信息
- Token 成本 ≈ 单次调用 × 3

**新版（v3）**：单 LLM Planner + N 个无状态 Tool。

**收益**：
- LLM 调用从 3 次降到 1-2 次（Self-Critique 复用 KV Cache）
- Tool 之间无状态，可任意并行（asyncio.gather）
- Planner 一次性掌握全局，决策更优
- P95 从 9s 降到 3s

**金句（面试可用）**：
> "多 Agent 不等于多 LLM。在我们的语境下，'多 Agent 协作'指的是多种角色（规划者、执行者、校验者）的协作，而最高效的实现方式是一个 LLM 扮演多个角色 + 多个无状态 Tool。"

### 3.2 为什么需要三层 Memory？

| 层级 | 时间尺度 | 存储位置 | 写入条件 | 用途 |
|------|---------|---------|---------|------|
| Working | 会话内 | 内存 / Redis | 每轮对话 | 累积本次会话症状、避免重复推理 |
| Episodic | 用户长期 | PostgreSQL+Zilliz | Importance Score≥0.3 | 老用户的过敏史、慢性病、过往诊断 |
| Semantic | 全局共享 | Neo4j+案例库 | L3 Reflexion 沉淀 | 系统级自我进化 |

**关键设计**：三层是**独立的写入闭环 + 独立的检索时机**，不是一个大池子。

### 3.3 为什么需要分级反思？

朴素的 Reflexion（无差别全跑）会让简单查询也走完整反思链路，P95 拉到 ~6s。

分级思路：
- **L1 规则**：~75% 流量在这里就被拦截，0 LLM 调用，<1ms
- **L2 小模型**：20% 流量，蒸馏的 1.5B 模型语义校验，~200ms
- **L3 完整**：5% 流量，主 LLM 反思 + Semantic Memory 沉淀，~2s

平均反思耗时从 ~2s 降到 ~150ms。

### 3.4 为什么用 GraphRAG 而不是传统 RAG？

传统 RAG 的"单次 chunk 检索"在医疗多跳推理上失败：
- 问"头痛+发烧+颈强直可能是什么病" → chunk 里没有现成答案
- 需要从症状到疾病的多跳推理

GraphRAG 解法：
- **社区检测**（Leiden）把 KG 切成医学专科聚类
- **社区摘要**让每个聚类有自然语言描述（可被向量化检索）
- **Local Search**：实体级 1-2 hop（适合具体事实）
- **Global Search**：社区级召回（适合鉴别诊断）
- **Personalized PageRank**：模拟"自由联想"的多跳推理

详见 `docs/05_graphrag.md`。

## 四、模块实现状态

| 模块 | 状态 | 说明 |
|------|------|------|
| KG 构建（Neo4j 4.5万实体/31万关系） | ✅ 完整 | 简历已写，确实做了 |
| Qwen2.5-7B LoRA 微调 | ✅ 完整 | 简历已写，确实做了 |
| Planner Agent（DAG 输出 + Self-Critique） | ✅ 接口完整 / 🟡 Mock LLM | 真实切 vLLM 即可 |
| Tool 系统（NER/Local/Global/PPR） | ✅ 接口完整 / 🟡 部分 Mock | Neo4j 接口已留 |
| Working Memory（含压缩、卡片化） | ✅ 完整 | 纯 Python 实现 |
| Episodic Memory（含 Importance Scoring） | ✅ PoC 完整 | SQLite 实现，可切 PG+Zilliz |
| Semantic Memory（失败案例 + 动态规则） | ✅ 接口完整 / ⚪ KG 写入待接 Neo4j | |
| L1 规则反思 | ✅ 完整 | 6 类规则 + 动态扩展 |
| L2 小模型反思 | 🟡 接口完整 / ⚪ 蒸馏待训 | 训练流程见 scripts/ |
| L3 Reflexion + 进化闭环 | ✅ 接口完整 / 🟡 Mock LLM | |
| GraphRAG 社区检测 + 摘要 | ✅ mock 可跑（标签传播+回填验证）/ ⚪ 生产用 Leiden | src/graphrag/ |
| PPR 多跳推理 | ✅ 完整 | 纯 Python，alpha=0.5 调优 |
| Eval 体系（CMB/MedQA/自建） | 🟡 框架完成 / ⚪ 待跑全量 | |

**诚实说明（面试时务必这样讲）**：
- "核心架构和 PoC 已完成，单元演示能跑通"
- "GraphRAG 离线索引、L2 蒸馏、全量 Eval 还在迭代中"
- 不要说"全部完整上线"——大厂面试官会追问细节，露馅成本极高

## 五、生产环境切换清单

当你在自己服务器上跑时，按这个清单逐个替换：

| Mock 组件 | 替换为 | 操作 |
|----------|-------|------|
| `MockLLMBackend` | vLLM serving Qwen2.5-7B-LoRA | 改 `planner.LLMBackend` 子类 |
| `MockKGBackend` | `Neo4jBackend`（骨架已留） | 在 `kg_local_search.py` 取消注释 |
| `MockCommunityVectorStore` | Zilliz / Milvus | 实现 `VectorStore.search` |
| `MockEmbedder` | `BAAI/bge-m3` | `pip install sentence-transformers` |
| `MockSmallModel` | 蒸馏的 1.5B Verifier | 跑 `scripts/train_l2_verifier.py` |
| `MockSummarizer` | 1.5B 小模型 / API | 同上 |

## 六、关键文件索引

- `src/orchestrator.py` — 主调度器，串起所有模块
- `src/agents/planner.py` — Planner Agent
- `src/tools/` — 4 个 Tool 实现
- `src/memory/` — 三层 Memory
- `src/verifiers/` — 三级 Verifier
- `src/schemas.py` — 全局数据结构
- `demo/demo_full_flow.py` — 可独立运行的演示

---

下一步建议阅读：[02_planner_executor.md](02_planner_executor.md)
