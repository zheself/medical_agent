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
- 当前基线：**V8** — L3 Phase 2（prompt 改写 + trigger guard）在 Phase 1（merge guard）之上。
  - Top-3 loose **45.5%**，Top-5 49.4%，Hard Top-3 47.5%，0 errors (77/77)。
  - V7 保持历史最佳：Top-3 46.7%，Hard Top-3 50.0%。
  - L3 reflexion 率 9.1% — 全部为 `empty_diagnoses` 触发，correct wipe = 0。
  - L3 merge guard + trigger guard 已默认启用。`verification_meta` 字段可观测 trigger/skip reason。
  - 诊断基线仍为 V8；V11b 工程改动后当前为 **103/103 测试通过**。
- V9 完成 DAG 工具并行可观测性，sleep benchmark 并行倍率 2.98x。
- V10a-c 完成 Memory 可观测性、规则门控、写入生命周期和 60 条多会话 benchmark。
  - Qwen3-8B V10c：Raw Memory 将回答约束通过率从 46% 提升到 90%，但每题注入 2.0 条无关记录。
  - Rule Gate 将 episodic 上下文压缩 86.3%，但 injection recall 降至 28%；critical profile recall 保持 100%。
  - 三配置跨用户泄漏率均为 0%。
- P0 消融关键发现：PPR 在 CMB-Clin 上净贡献接近零（L2-only 下约 −1.3pp，运行方差范围内）。
  PPR OFF 的 Top-3 优势（+6.1pp）主要来自 L3 补偿效应，而非 PPR 质量。
- V11b 已完成 hard-negative reranker 与 24/36 dev/test 隔离。held-out test 上 hybrid rerank 将 Recall@1 从 30.0% 提升到 50.0%，temporal forbidden@5 从 100% 降到 16.7%，但 Recall@3/5 下降；Qwen 回答约束通过率从 80.0% 到 83.3%，因此 reranker 保持可选。

接手后第一件事：

1. 确认环境：`conda activate cjz_opd`。工作报告记录该环境已安装 `vllm`、`igraph`、`leidenalg` 和 `networkx`。
2. 在 `medical_agent/` 下运行 `python tests/run_all.py`；当前期望是 **103/103 通过**。
3. 在 8001 端口启动 vLLM，然后运行 `python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl --concurrency 1` 验证真实管道。
4. 阅读最新工作报告：`medical_agent/reports/2026-07-17_V11b_reranking_record.md`、`medical_agent/reports/2026-07-16_V11a_bge_retrieval_record.md` 和 `medical_agent/reports/2026-07-15_V10c_memory_benchmark_record.md`。

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
- **V8 诊断基线**：Top-3 loose 45.5%，Hard Top-3 47.5%，0 errors；V7 保持历史最佳 Top-3 46.7%。当前测试基线为 103/103。
- **V10c Memory benchmark**：60 条场景，Qwen3-8B，180/180 成功。Raw Memory 回答约束通过率 90%，Rule Gate 74%，No Memory 46%，跨用户泄漏 0%。
- Planner sanitizer (`_sanitize_plan`)：规范化 depends_on 类型、去重重复工具调用、清理非法依赖。
- Routing guard (`_route_complexity`)：规则层修正 LLM 复杂度判断。
- L3 merge guard：防止 L3 reflexion 覆盖 L2 正确诊断。
- PPR OFF 消融开关和 `--max-verifier-level` flag 已在 `eval/run_eval.py` 中可用。
- P0 关键发现：PPR 在 CMB-Clin 上净贡献接近零（L2-only 约 −1.3pp）。L3 reflexion 是更强的 lever。
- **V11a BGE 检索**：真实 BGE-M3 dense Recall@1 72%、MRR 0.850；hybrid 将 temporal forbidden@1 从 80% 降至 0%，但整体 MRR 降至 0.793；跨用户泄漏仍为 0%。
- **V11b Reranker**：12 候选 hard-negative benchmark，dev 冻结调参。held-out Recall@1 50.0% vs dense 30.0%，temporal forbidden@5 16.7% vs 100%，Qwen 约束通过率 83.3% vs 80.0%；由于 Recall@3/5 回退，当前保持 opt-in。
- 最新工作记录：`reports/2026-07-17_V11b_reranking_record.md`、`reports/2026-07-16_V11a_bge_retrieval_record.md`、`reports/2026-07-15_V10c_memory_benchmark_record.md`，以及此前的 P0/L3 报告。

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

1. **Temporal Memory**：加入结构化状态、来源和 supersede，处理新旧事实冲突。
2. **按记忆类型的策略**：避免 medication/critical memory 共用一个全局 threshold，在保留 abstention 的同时恢复召回。
3. **真实 L2 verifier**：替换 MockSmallModel，并重新标定 L3 trigger。
4. **KG retrieval**：实现 `SQLiteCommunityStore`，继续实体归一化和 NER。
