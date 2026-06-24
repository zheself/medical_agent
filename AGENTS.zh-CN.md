# AGENTS.md 中文版

## 项目概览

当前工作区包含一个医疗诊断推理 Agent 原型。可运行项目位于 `medical_agent/`。

系统主要由以下部分组成：

- Planner-Executor 调度架构
- 知识图谱工具与 GraphRAG 检索
- Working、Episodic、Semantic 三层记忆
- L1 规则、L2 小模型、L3 Reflexion 组成的分级反思
- `mock`、`db`、`vllm` 三种后端路径

除非明确接入经过验证的生产级医学数据源并完成专业审查，否则本仓库中的医学内容都应视为研究/原型数据。不要把系统输出表述为医疗建议。

## 快速接手

新接手的 Agent 应优先阅读本节。

项目定位：这是一个医疗诊断推理 Agent，采用 Planner-Executor 架构、三层 Memory、分级反思和 GraphRAG。核心代码在 `medical_agent/src/`；架构文档在 `medical_agent/docs/`，建议先读 `medical_agent/docs/01_architecture.md`。

三种后端通过 `src.factory.build_system(backend=...)` 切换：

- `mock`：全内存硬编码数据，无外部依赖，主要用于测试。
- `db`：从 `data/medical_agent.db` 读取真实 SQLite KG，但 LLM 仍使用 `MockLLMBackend`。
- `vllm`：通过 vLLM 服务调用真实 Qwen3-8B。使用该模式前需要先启动 vLLM 服务。

当前进度：

- P0 真实 vLLM 接入已完成。
- P1 真实 KG 导入已完成。
- PPR IDF 边加权与类型过滤已完成。
- 评测管道已打通。
- CMB-Clin 评测集共有 77 条；适配脚本质量和 loose-match 规则已验证。
- vLLM 全量评测已跑两轮：V1 baseline 和 V2 memory 隔离后。
- 关键发现：PPR 调用率只有 6.5%。根因是 Planner 复杂度分类：93.5% 的条目被判为 `medium`，因此只触发 `local_search`。
- eval 中 episodic memory 跨 item 污染导致的噪声 bug 已修复。

接手后第一件事：

1. 确认环境：`conda activate cjz_opd`。工作报告记录该环境已安装 `vllm`、`igraph`、`leidenalg` 和 `networkx`。
2. 在 `medical_agent/` 下运行 `python tests/run_all.py`；历史基线期望是 38/38 通过。
3. 在 8001 端口启动 vLLM，然后运行 `python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl` 验证真实管道。
4. 在修改 Planner 路由或评测逻辑前，先阅读 `medical_agent/reports/2026-05-30_P1_report.md` 的 5e 和 5f 节。

硬性规则：

- 保留 `mock`、`db`、`vllm` 三种 backend。
- 除纯文档改动外，代码改动后运行 `tests/run_all.py`。
- 大改动先写计划，并等待用户确认。

## 当前项目状态

以 `medical_agent/reports/` 作为当前工作记录的准确信息源。它比 README 的部分内容更新。

已完成的重要里程碑：

- P0 vLLM 集成已完成：`backend="vllm"` 通过 OpenAI-compatible vLLM 服务使用 Qwen3-8B。
- P1 真实 KG 导入已完成：`data/medical_agent.db` 已替换为从 `data/medical_new_2.json` 导入的医学 KG。
- 当前 KG 规模约为 22,480 个实体、303,143 条关系、22,479 个 PageRank 节点，Leiden 社区为 L0=4、L1=14、L2=26。
- PPR IDF 边加权和 disease 类型过滤已实现。
- CMB-Clin 评测数据已适配到 `data/eval_cmb_clin.jsonl`，共 77 条。
- vLLM 全量评测已跑两轮：
  - V1 baseline：Top-3 loose 42.1%，平均延迟约 30.6s。
  - V2 memory 隔离和 PPR 工具描述清理后：Top-3 loose 仍为 42.1%，平均延迟约 24.7s，平均 token 降约 34%。
- 已修复一个重要 eval bug：每条 eval item 使用隔离的 `user_id=f"eval_{item_id}"`，避免 episodic memory 跨 item 污染。
- Planner 工具选择诊断已完成。PPR/global search 调用率只有 6.5%，因为 93.5% 的 CMB-Clin 条目被判为 `medium` 并路由到 `ner + kg_local_search`。

## 目录说明

- `medical_agent/README.md`：主项目文档和快速开始。
- `medical_agent/src/`：核心包。
- `medical_agent/src/factory.py`：推荐的系统组装入口。
- `medical_agent/demo/demo_full_flow.py`：端到端 demo。
- `medical_agent/tests/`：轻依赖测试，不依赖 pytest 也能运行。
- `medical_agent/eval/`：评测 runner 和指标。
- `medical_agent/scripts/`：离线数据、KG、GraphRAG、训练脚本。
- `medical_agent/app/streamlit_app.py`：可选 Streamlit UI。
- `medical_agent/data/`：本地 demo/评测数据和 SQLite 文件。
- `medical_agent/docs/`：架构和设计文档。
- `medical_agent/reports/`：按时间记录的工作报告，是当前状态的最佳信息源。
- 顶层 `factory.py` 和 `seed_database.py`：旧副本/shim；优先使用 `medical_agent/src/` 和 `medical_agent/scripts/` 下的版本。

## 常用命令

除非特别说明，从 `medical_agent/` 目录运行命令：

```bash
cd medical_agent
python -m demo.demo_full_flow
python scripts/seed_database.py --force
python -m demo.demo_full_flow --db
python -m tests.run_all
python -m eval.run_eval --backend mock --num-items 5
python scripts/diagnose_tool_usage.py
```

可选 UI：

```bash
cd medical_agent
streamlit run app/streamlit_app.py
```

生产向依赖列在 `medical_agent/requirements.txt`。mock demo 和核心测试设计上应可仅用 Python 标准库运行。

真实 vLLM 评测使用 reports 中记录的环境：

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

然后运行：

```bash
cd medical_agent
python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl --concurrency 1
```

## 开发规范

- 构造 Agent 时优先使用 `src.factory.build_system()`，backend 切换应集中在这里。
- 保留 mock 路径。它是无 GPU 服务时验证 orchestration、memory、tool call 和 verifier 行为的最快方式。
- SQLite 后端行为应继续兼容 `scripts/seed_database.py`。
- 不要假设 `data/medical_agent.db` 只是 45 实体的 seed 数据库。根据当前 reports，它已被完整导入 KG 替换。
- 不要把新的医学事实硬编码进调度逻辑。demo 数据应放在 seed 脚本、mock backend 或明确限定范围的测试 fixture 中。
- 保持 planner、tools、memory、verifiers、schemas 之间的接口稳定。修改契约时同步更新测试和文档。
- 已有 schema 时，使用 `src/schemas.py` 中的结构化数据，不要用松散 dict 替代。
- 除非用户明确要求保留可复现实验 fixture，否则不要提交生成的评测输出、本地数据库、模型 checkpoint 或 secrets。
- `.env` 保持本地使用。用 `.env.example` 记录配置项。
- 修改 eval 逻辑时，必须保留每条 item 独立的 user id。不要重新引入共享 `eval_user`。

## 测试要求

交付代码改动前至少运行：

```bash
cd medical_agent
python -m tests.run_all
```

涉及评测逻辑的改动，运行：

```bash
cd medical_agent
python -m eval.run_eval --backend mock --num-items 5
```

涉及 SQLite 后端或 seed 数据的改动，运行：

```bash
cd medical_agent
python scripts/seed_database.py --force
python -m tests.run_all
python -m demo.demo_full_flow --db
```

如果某个命令依赖当前不可用的服务，例如 Neo4j、Milvus、vLLM 或 GPU，要说明没有运行，并写清楚缺少什么服务。

## 医疗安全边界

- 本仓库是研究/原型系统。生成回答时应保留适当的不确定性和就医升级提示。
- 不要随意移除 L1 安全检查。药物过敏、禁忌症、急症症状、儿童、妊娠等高风险检查应尽可能保持低成本、确定性。
- 对高热伴颈强直、胸痛、卒中样症状、严重过敏反应、自杀意念等红旗信号，行为应保守。
- 任何生产化计划都必须包含经过验证的医学数据源、临床专家审查、审计日志、隐私评审和监管评估。

## 当前优先级计划

1. P0：调整 Planner complexity/routing，让临床诊断和鉴别诊断条目更充分地判为 `high`，从而触发 `kg_global_search` 和 `ppr_reasoner`。
2. P0：增加结构化诊断候选数量。当前全量评测平均只有约 2 个候选，因此 Top-3 和 Top-5 相同。
3. P0：路由改动后先重跑 `scripts/diagnose_tool_usage.py`。在跑昂贵的全量评测前，目标是让 PPR/global-search 调用率从当前 6.5% 有实质提升。
4. P0：重跑 CMB-Clin vLLM 评测；当工具使用率实际发生变化后，再跑消融实验。
5. P1：实现 `SQLiteCommunityStore`，让 global search 使用导入的 Leiden 社区，而不是 mock 社区数据。
6. P2：改进 PPR 和 local search 的 KG 检索深度与性能，重点关注 `SQLiteKGBackend.query_neighbors`。
7. P2：NER 先从硬编码匹配升级到 Trie/词典归一化，再考虑模型。真实 KG 的实体变体当前是召回瓶颈。
8. P2：评估实体归一化做到什么程度才值得。reports 已证明同义词/粒度不一致是真问题，但完整归一化容易失控。
