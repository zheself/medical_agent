# P0 VLLMBackend 实现及剩余问题修复报告

日期: 2026-05-29 ~ 2026-05-30

---

## 一、项目总体任务设计

### 1.1 项目背景

本项目是西北工业大学软件工程硕士的毕业设计——"多智能体协作与知识图谱驱动的医疗诊断推理系统"。当前状态为**代码脚手架 + PoC + 完整设计文档**，核心架构和 PoC 能在 Mock 后端下跑通。

### 1.2 架构概览

系统采用 Planner-Executor 模式，四大核心支柱：

1. **Planner-Executor**：一次 LLM 调用生成 DAG 计划，N 个无状态 Tool 并行执行，LLM 调用从 3.2 次/query 降到 1.4 次
2. **三层记忆**：Working（会话级）→ Episodic（用户级）→ Semantic（全局），写入方向单向
3. **分级反思**：L1 规则（<1ms，拦截 75%）→ L2 蒸馏小模型（~200ms，20%）→ L3 Reflexion（~2s，5%）
4. **GraphRAG**：三层社区检测 + Personalized PageRank（alpha=0.5）多跳推理

### 1.3 Mock → 真实组件的迁移路线

| 优先级 | 组件 | Mock → 真实 | 影响 |
|--------|------|------------|------|
| **P0 ✅** | LLM Backend | MockLLMBackend → VLLMBackend | Planner 整体质量 |
| P1 | KG Backend | MockKGBackend → SQLiteKGBackend → Neo4jBackend | 检索规模 |
| P1 | NER Tool | 词典匹配 → BertBiLSTMCRF | 实体提取精度 |
| P2 | Embedder | MockEmbedder（22维）→ bge-m3（1024维） | 检索质量 |
| P2 | Vector Store | MockCommunityVectorStore → Milvus | Global Search 质量 |
| P3 | L2 Verifier | MockSmallModel → 蒸馏 1.5B | 验证精度 |

### 1.4 工作约定

1. 每个大改动前列计划，确认后写代码
2. 每次改完跑 `python -m tests.run_all`，确保 38 个测试不被破坏
3. 切换真实组件时保留 Mock 实现和 factory 的 backend 开关
4. 改 `src/factory.py` 加新 backend，不删现有的
5. 遇到依赖/服务先告诉用户，由用户执行

---

## 二、Day 1 工作（2026-05-29）：P0 VLLMBackend 实现

### 2.1 环境确认

| 项目 | 详情 |
|------|------|
| GPU | 4× NVIDIA RTX A6000，单卡 48GB，总计 192GB |
| CUDA | 12.8，驱动 570.153.02 |
| vLLM | 0.11.0（cjz_opd conda 环境，Python 3.10.20） |
| 模型权重 | Qwen3-8B，5 个 safetensors 分片（符号链接指向 HF cache） |
| 模型参数 | 36 层，hidden 4096，32 heads，最大上下文 40960 tokens |

GPU 实际可用情况：GPU 0-2 被 ray 训练进程各占 ~1GB，GPU 3 被 sglang 占 40GB。最终使用 GPU 0-1 两张卡运行 vLLM。

### 2.2 VLLMBackend 实现

#### 代码改动

| 文件 | 改动 |
|------|------|
| `src/agents/planner.py` | +VLLMBackend 类、+think 标签剥离、PLANNER_SYSTEM_PROMPT 重写、SELF_CRITIQUE_SUFFIX 增强 |
| `src/factory.py` | +backend="vllm" 分支、+_load_dotenv、+_env |
| `src/orchestrator.py` | _synthesize_draft 以 thought 为主体、+正则提取 citations |
| `demo/demo_full_flow.py` | +--backend 参数、更新输出文案 |
| `.env.example` | 模型名改为 Qwen3-8B |
| `.env` | 新建，端口改为 8001 |

#### vLLM 启动命令

```bash
conda activate cjz_opd

python -m vllm.entrypoints.openai.api_server \
    --model /mnt/sdc/ubuntu/cjz_projects/OPD/Lightning-OPD/checkpoints/teachers/Qwen3-8B \
    --served-model-name Qwen3-8B \
    --host 0.0.0.0 \
    --port 8001 \
    --enable-prefix-caching \
    --max-model-len 8192 \
    --tensor-parallel-size 2
```

### 2.3 L1 CITATION 问题修复

**问题**：VLLMBackend 上线后，L1 规则校验持续报 `L1[CITATION]: 回答缺少 KG/文献引用支持`，导致触发 L3 Reflexion（额外 LLM 调用 +5s 延迟）。

**根因**：`_synthesize_draft` 完全依赖工具结果拼接 content 和 citations，工具返回空数据时 content 为空壳、citations 为空。

**修复方案（prompt B + A 组合）**：

1. **PLANNER_SYSTEM_PROMPT**：thought 字段从"简明扼要"改为"完整诊断推理"，新增强制引用格式 `[实体A] -[关系]-> [实体B]`
2. **`_synthesize_draft`**：以 Planner thought 作为回答主体，工具结果作为补充；从 thought 中正则提取 citations 兜底

**效果**：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Case 1 L1 | ❌ CITATION 失败 | ✅ 通过 |
| Case 1 L3 | 触发（+5.8s） | 未触发 |
| Case 1 延迟 | 21.8s | 17.0s |
| Case 2 延迟 | 9.1s | 5.3s |
| thought 质量 | 计划摘要 | 完整诊断推理 + 结构化引用 |

---

## 三、Day 2 工作（2026-05-30）：剩余问题修复

### 3.1 PPR 输出为空 → ✅ 已修复

**根因**：`data/medical_agent.db` 的 `kg_entities`、`kg_relations`、`kg_pagerank` 表全部为空。`seed_database.py` 有递归 bug（`log(*a)` 调用自身而非 `print(*a)`），导致脚本从未成功执行，KG 数据从未写入。

**修复**：
- `scripts/seed_database.py` 第 292 行：`log(*a)` → `print(*a)`
- 重新执行 `python scripts/seed_database.py --force`，写入 45 entities、52 relations、42 PageRank 节点

**验证**：db 模式下 PPR 正确输出 `脑膜炎: PPR=0.1661` 为最相关概念，和 Mock 模式行为一致。

### 3.2 Prefetch 命中率 0% → ✅ 已改善

**根因分析**（三个层面）：

1. **LLM 实体名与 KG 规范名不匹配**：LLM 可能输出"细菌性脑膜炎"、"流行性感冒"等变体名，和 KG 中的"脑膜炎"、"流感"精确字符串匹配失败
2. **produced_entities 提取遗漏**：只检查了 3 个 key（`facts[].target`、`relevant_concepts[].entity`、`candidate_entities[]`），漏了 NER `all_entities`、KG `entities_queried`、PPR `seed_entities`、社区 `core_entities`、facts `source`
3. **空成功膨胀统计**：KG 中找不到的实体仍返回 `success=True`（facts 为空），被计入 `total_prefetched` 但永远无法命中

**修复**：

1. `_reconcile_prefetch`：扩展 `produced_entities` 提取，新增 6 个数据来源
2. `_launch_prefetch`：只缓存 facts 非空的预取结果，过滤无效实体
3. 实体名归一化（LLM 变体 → KG 规范名）暂未实现，待后续

**效果**：Mock 模式下命中率从 67% → 100%（空命中过滤 + 更宽的匹配面）

### 3.3 thought 引用真实性 → ✅ 已修复

**问题**：thought 中的 `[脑膜炎] -[典型症状]-> [头痛]` 是 LLM 基于自身知识生成的，不是从 KG 检索的真实引用。

**修复**：`_synthesize_draft` 中 citations 分为两类：

- **verified**（`type: kg_fact` / `community_summary`）：来自工具返回的真实 KG 数据，或 thought 引用经 KG 事实集验证匹配
- **inferred**（`type: kg_inferred`）：thought 引用未匹配到真实 KG 数据，标注来源为 LLM 推断

构建流程：
1. 工具返回的 citations → verified（最高可信度）
2. 收集真实 KG 事实集 `{(source, rel, target)}` 用于验证
3. thought 中的 `[实体] -[关系]-> [实体]` 引用：匹配 KG → verified，不匹配 → inferred
4. 最终 citations = verified + inferred，verified 排前面

**验证**：db 模式下鉴别诊断查询 2 verified / 0 inferred，事实查询 3 verified / 0 inferred。

---

## 四、当前系统状态

### 4.1 三种 backend 模式对比

| 模式 | LLM | KG | PPR | Prefetch | Citations | 状态 |
|------|-----|----|-----|----------|-----------|------|
| mock | MockLLMBackend | MockKGBackend（硬编码） | ✅ 正常 | 100% 命中 | verified | ✅ 完整可用 |
| db | MockLLMBackend | SQLiteKGBackend（45实体） | ✅ 正常 | 100% 命中 | verified | ✅ 完整可用 |
| vllm | VLLMBackend（Qwen3-8B） | SQLiteKGBackend（45实体） | ✅ 正常 | 待真实 LLM 测试 | verified + inferred | ✅ 可用（需 vLLM 服务） |

### 4.2 测试覆盖

38/38 单元测试全部通过，覆盖 DAG 引用解析、PPR 算法、L1 规则、社区检测、数据库后端等核心逻辑。

---

## 五、后续计划

### P1：KG 扩展 + NER 升级

**目标**：将 KG 从 45 实体扩展到生产规模，NER 从词典匹配升级为模型

1. **Neo4jBackend**：骨架已写好，需要：
   - 运行 `build_kg.py` 从 Huatuo-26M 等数据集建图
   - NER + 关系抽取（当前是 NotImplementedError）
2. **NER Tool 升级**：BertBiLSTMCRF 骨架已预留
   - 需要训练模型或使用预训练 NER 模型
   - 7 类实体（疾病、症状、药物、检查、科室、体质、过敏原）
3. **实体名归一化**（Prefetch 命中率的最后一环）
   - 构建 LLM 输出变体名 → KG 规范名的映射表
   - 或在 NER 工具中增加别名识别

### P2：Embedder + Vector Store

1. MockEmbedder（22维 multi-hot）→ bge-m3（1024维）
2. MockCommunityVectorStore → Milvus
3. 重建 GraphRAG 索引（`build_graphrag_index.py`）

### P3：L2 Verifier 蒸馏

1. 教师标注（7B 评分 5000 三元组）
2. SFT 训练（1.5B 学生 + QLoRA）
3. Pearson 相关性 + AUC 评估

### P4（新增）：Prefetch 实体名归一化

- 构建 LLM 变体名 → KG 规范名映射
- 或在 `_reconcile_prefetch` 中加入 NER 词典归一化
- 预期将真实 LLM 下命中率从 0% 提升到 ~30-45%

---

## 六、完整文件改动清单

| 文件 | Day 1 改动 | Day 2 改动 |
|------|-----------|-----------|
| `src/agents/planner.py` | +VLLMBackend、+think 剥离、prompt 重写 | 无 |
| `src/factory.py` | +backend="vllm"、+_load_dotenv、+_env | 无 |
| `src/orchestrator.py` | _synthesize_draft 以 thought 为主体、+正则提取 citations | 扩展 produced_entities（6 个新来源）、过滤无效预取、citations verified/inferred 分类 |
| `demo/demo_full_flow.py` | +--backend 参数 | 无 |
| `.env.example` | 模型名改为 Qwen3-8B | 无 |
| `.env` | 新建，端口 8001 | 无 |
| `scripts/seed_database.py` | 无 | 修复 log 递归 bug（log → print） |