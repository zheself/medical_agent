# 基于多 Agent 协作与知识图谱的医疗诊断推理系统

> 西北工业大学 · 软件工程硕士 · 个人项目
> 目标：通过 KG + GraphRAG + 分级反思 + 多层 Memory 构建医疗诊断 Agent
> 
> **本仓库当前为代码骨架 + PoC + 完整设计文档**，详见各模块实现状态表。

## 快速开始（无需 GPU 即可运行）

```bash
# 1. 克隆
git clone <your-repo>
cd medical_agent

# 2. 仅需 Python 3.10+，无外部依赖即可跑 Mock 版 demo
python -m demo.demo_full_flow
```

### 用真实数据库运行（SQLite）

项目提供一个种子脚本，生成模拟的真实数据库 `data/medical_agent.db`
（知识图谱实体/关系、用户历史、社区摘要、失败案例等），让系统"连的是真库"
而非硬编码 dict：

```bash
# 1. 生成数据库（45 实体 / 52 关系 / 6 条用户历史 / 3 社区 ...）
python scripts/seed_database.py

# 2. 用 db 后端跑 demo（KG/Episodic/Semantic 都从 .db 读取）
python -m demo.demo_full_flow --db
```

`db` 模式下，系统会真实读取用户历史——例如 `patient_001` 的糖尿病病史和
青霉素过敏史会在问诊时被自动注入，触发 L1 过敏冲突拦截。

> mock 与 db 两种后端通过 `src/factory.py` 的 `build_system(backend=...)` 统一切换，
> 接口完全一致。再往上把 SQLite 换成 Neo4j/vLLM/Milvus 即生产形态。

### 可视化界面（Streamlit）

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

界面展示：多轮问诊对话、Planner 的 DAG 计划、工具执行轨迹（NER/GraphRAG/PPR）、
三层 Memory 状态、分级反思链路（L1/L2/L3）。侧边栏可实时切换消融开关、写入模拟用户历史。
同样使用 Mock 后端，无需 GPU/Neo4j/vLLM。

预期输出：完整流程 7 步演示，包括 Planner 决策、Tool DAG 执行、PPR 多跳推理、L1/L3 分级反思。

## 系统亮点

### 1. Planner-Executor 模式（v3 核心重构）
- 一个 LLM Planner 生成 DAG 计划，N 个无状态 Tool 并行执行
- LLM 调用次数从 v2 的 3.2 次降到 1.4 次
- Self-Critique 复用 KV Cache，节省 35% 延迟

### 2. 三层 Memory 体系
- **Working Memory**: 会话级，结构化 PatientProfile 注入卡片（token 降 90%）
- **Episodic Memory**: 用户级，BGE-M3 dense / hybrid retrieval + 可选 BGE CrossEncoder reranker；向量模型版本隔离，避免不同 embedding 混用
- **Semantic Memory**: 全局，L3 反思修正后沉淀回 KG / 规则库 / 失败案例
- **Memory Benchmarks**: V10c Raw Memory 回答约束通过率 90% vs No Memory 46%；V11b held-out reranker Recall@1 50.0% vs dense 30.0%，Qwen 约束通过率 83.3% vs 80.0%

### 3. 分级反思（L1/L2/L3）
- L1 规则反思：0 LLM 调用，<1ms，拦截 75% 流量
- L2 蒸馏 1.5B 小模型反思：~200ms，处理 20%
- L3 完整反思 + Semantic Memory 进化闭环：~2s，仅 5% 流量

### 4. GraphRAG 多跳推理
- Leiden 三层社区检测（粗/中/细，对应医学专科-疾病大类-具体疾病）
- Local + Global 双模式检索
- **Personalized PageRank** 多跳推理（alpha=0.5 调优，比常规设置 F1 高 13pp）
- 摘要回填验证防幻觉

## 目录结构

```
medical_agent/
├── README.md                          # 本文档
├── docs/                              # 详细设计文档（面试核心）
│   ├── 01_architecture.md             # 整体架构
│   ├── 02_planner_executor.md         # Planner-Executor 详解
│   ├── 03_memory_system.md            # 三层 Memory
│   ├── 04_graded_reflection.md        # 分级反思
│   ├── 05_graphrag.md                 # GraphRAG 多跳推理
│   ├── 06_evaluation.md               # 评估体系
│   └── 07_iteration_story.md          # 项目迭代叙事 (面试讲稿)
├── src/
│   ├── orchestrator.py                # 主调度器
│   ├── schemas.py                     # 数据结构
│   ├── agents/planner.py              # Planner Agent
│   ├── tools/                         # 工具实现
│   │   ├── base.py
│   │   ├── ner_tool.py
│   │   ├── kg_local_search.py
│   │   ├── kg_global_search.py
│   │   └── ppr_reasoner.py
│   ├── memory/                        # 三层 Memory
│   │   ├── working_memory.py
│   │   ├── episodic_memory.py
│   │   └── semantic_memory.py
│   ├── verifiers/                     # 三级 Verifier
│   │   ├── l1_rule_verifier.py
│   │   ├── l2_model_verifier.py
│   │   └── l3_reflexion.py
│   └── graphrag/                      # GraphRAG 社区检测+摘要（mock 可跑）
├── eval/                              # 评估框架
├── scripts/                           # 离线脚本
│   ├── seed_database.py               # ★ 生成模拟真实数据库 .db
│   ├── build_kg.py                    # 生产: Huatuo → Neo4j
│   ├── build_graphrag_index.py        # 生产: Leiden 社区检测+摘要
│   └── train_l2_verifier.py           # 生产: 蒸馏 1.5B Verifier
├── data/
│   └── medical_agent.db               # ★ 模拟真实数据库（运行 seed 后生成）
├── src/
│   └── factory.py                     # ★ 系统工厂（mock/db 后端统一切换）
├── eval/
│   ├── metrics.py                     # 四维评测指标
│   ├── run_eval.py                    # 单配置评测入口
│   └── run_ablation.py                # 消融实验（开关真实生效）
├── tests/                             # 单元测试（无需 pytest 即可跑）
│   ├── test_p0.py                     # 引用解析/语义检索/GraphRAG
│   ├── test_core.py                   # PPR/L1/DAG/消融开关
│   ├── test_database.py               # SQLite .db 后端
│   └── run_all.py                     # 一键跑全部
├── app/
│   └── streamlit_app.py               # Streamlit 可视化界面
├── demo/
│   └── demo_full_flow.py              # 可独立运行的完整流程演示
├── requirements.txt
└── .env.example
```

## 模块实现状态

| 模块 | 状态 | 说明 |
|------|------|------|
| KG 构建（Neo4j 4.5万实体/31万关系） | ✅ 已完成 | 简历项目阶段已做 |
| Qwen2.5-7B LoRA 微调 | ✅ 已完成 | 简历项目阶段已做 |
| Planner Agent 核心 | ✅ 接口完整 / 🟡 Mock LLM | 真实接 vLLM 即可 |
| Tool 系统（NER/Local/Global/PPR） | ✅ 接口完整 / 🟡 部分 Mock | Neo4j 接口已留 |
| Working Memory | ✅ 完整 | 纯 Python 实现 |
| Episodic Memory | ✅ SQLite PoC + V10c/V11b benchmark | BGE-M3 dense/hybrid + CrossEncoder reranker、生命周期/隔离/gating 可评测；下一步 temporal state |
| Semantic Memory | ✅ 接口完整 / 🟡 KG 写入待接 | |
| L1 规则反思 | ✅ 完整 | 6 类规则 + 动态扩展 |
| L2 小模型反思 | 🟡 接口完整 / ⚪ 待蒸馏训练 | |
| L3 Reflexion | ✅ 接口完整 / 🟡 Mock LLM | |
| PPR 多跳推理 | ✅ 完整 | 纯 Python，alpha=0.5 调优 |
| GraphRAG 社区检测+摘要 | ✅ mock 可跑 / ⚪ 生产用 Leiden | mock 用标签传播，生产切 igraph+leidenalg |
| Eval 体系 | ✅ CMB-Clin + Memory benchmark | V10c Qwen3-8B 180/180 成功 |

## 切换到生产环境

当你在自己 GPU 服务器上跑时，按以下步骤逐个替换 Mock 组件：

### Step 1: 启动 Neo4j

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:5.x
```

### Step 2: 导入 KG 数据

```bash
python scripts/build_kg.py \
  --huatuo-path ./data/huatuo_26m_lite.json \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password yourpassword
```

### Step 3: 跑 GraphRAG 离线索引（社区检测 + 摘要）

```bash
python scripts/build_graphrag_index.py \
  --neo4j-uri bolt://localhost:7687 \
  --output-dir ./graphrag_index \
  --llm-endpoint http://localhost:8000/v1  # vLLM 服务
```

预期耗时 ~5 小时（4 小时 Leiden + 1 小时摘要生成）。

### Step 4: 启动 LoRA 微调后的 Qwen2.5-7B（vLLM）

```bash
vllm serve ./qwen2.5-7b-medical-lora \
  --port 8000 \
  --max-model-len 8192 \
  --enable-prefix-caching   # ← 关键！开启 KV Cache 复用
```

### Step 5: 训练 L2 Verifier（可选，先跑通整体再训）

```bash
python scripts/train_l2_verifier.py \
  --teacher-endpoint http://localhost:8000/v1 \
  --output-dir ./verifier-1.5b
```

### Step 6: 配置 `.env`

```bash
cp .env.example .env
# 编辑 .env，填入 Neo4j / vLLM / Zilliz 等真实地址
```

### Step 7: 替换 Mock 实现

修改 `demo/demo_full_flow.py` 中的 `build_system()`：

```python
# 把这一行：
llm = MockLLMBackend()
# 换成：
from src.adapters.vllm_backend import VLLMBackend
llm = VLLMBackend(endpoint="http://localhost:8000/v1", model_name="qwen2.5-7b-medical-lora")

# 把这一行：
kg_backend = MockKGBackend()
# 换成：
from src.tools.kg_local_search import Neo4jBackend
kg_backend = Neo4jBackend(uri=..., user=..., password=...)
```

## 测试

```bash
# 一键跑全部单元测试（28 个，无需 pytest）
python -m tests.run_all
```

## 消融实验

```bash
# mock 模式：验证消融流程跑通（数字不可用于简历，见脚本顶部警告）
python -m eval.run_ablation
# 真实模式：替换真实组件 + 真实评测集后产出可用数字
python -m eval.run_ablation --data-path ./data/self_built.jsonl --real
```

## 评估

```bash
python eval/run_eval.py \
  --config eval/configs/full_eval.yaml \
  --output-dir ./eval_results
```

详见 `docs/06_evaluation.md`。

## 性能数据（V7 baseline — CMB-Clin 77 条，Qwen3-8B vLLM）

| 指标 | 数值 | 备注 |
|------|------|------|
| Top-3 loose | **46.7%** | PPR ON, L3 merge guard enabled |
| Top-5 loose | 49.3% | |
| Hard Top-3 | **50.0%** | 鉴别诊断/多跳推理类 |
| Medium Top-3 | 42.9% | 单一疾病查询类 |
| Mean latency | 16.9s | Planner + Executor + Verifier |
| L3 reflexion rate | 6.7% | 仅高复杂度 + L1/L2 失败时触发 |
| Error rate | **0%** (0/77) | sanitizer + citation guards |
| 测试 | **57/57** | 全部通过 |

**当前架构**：Planner (Qwen3-8B) → Executor (NER + KG global/local search + PPR) → Verifier (L1 rules + L2 small model + L3 reflexion with merge guard)

**已知限制**：
- PPR 在 CMB-Clin 上净贡献接近零（L2-only 下约 −1.3pp，处于运行方差范围内）；PPR 价值可能在 KG/NER/实体粒度修复后释放
- L3 merge guard 已缓解 correct wipe，但 L3 prompt 和 trigger 策略待下一阶段优化
- citation accuracy = 0%（citation 解析格式待标准化）

> 详细消融实验见 `reports/2026-06-24_P0_routing_record.md` 和 `reports/2026-06-25_L3_merge_record.md`。

## 引用与参考

- **MemGPT** (Packer et al. 2023) — 层次记忆设计
- **Generative Agents** (Park et al. 2023) — Importance Scoring
- **HippoRAG** (Sarthi et al. 2024) — PPR 多跳检索
- **GraphRAG** (Edge et al. 2024) — 社区检测 + 摘要
- **Reflexion** (Shinn et al. 2023) — 反思修正闭环

## License

Personal / Academic use only.
