# CLAUDE.md — Medical Agent 项目文档

## 项目概述

基于多 Agent 协作与知识图谱的医疗诊断推理系统。架构：Planner (Qwen3-8B) → Executor (NER + KG global/local search + PPR) → Graded Verifier (L1 rules + L2 small model + L3 reflexion)。

## 当前基线：V8

| 指标 | 数值 | 备注 |
|------|------|------|
| 评测集 | CMB-Clin 77 条 |
| Top-3 loose | **45.5%** | V7 46.8% 为历史最佳单次 |
| Top-5 loose | 49.4% |
| Hard Top-3 | 47.5% | V7 50.0% 为历史最佳 |
| Medium Top-3 | 43.2% |
| Errors | **0 / 77** |
| Mean latency | 17.2s |
| L3 reflexion rate | 9.1% | 全部为 empty_diagnoses 触发 |
| L3 merge guard | 已默认启用 | correct wipe = 0 |
| L3 trigger guard | 已默认启用 | trigger_reason 可观测 |
| 测试 | 103/103 通过 |

## 关键工程组件

- **Planner sanitizer** (`src/agents/planner.py`): `_sanitize_plan()` — depends_on 规范化 + 重复工具去重
- **Routing guard** (`src/agents/planner.py`): `_route_complexity()` — 规则层修正 LLM 复杂度判断
- **PPR OFF switch** (`src/orchestrator.py`): `enable_ppr=False` — 消融开关，过滤 PPR 步骤并清理依赖
- **L3 merge guard** (`src/verifiers/l3_reflexion.py`): `merge_l3_diagnoses()` — 防止 L3 覆盖 L2 正确诊断
- **Citation guards**: L1/L3/orchestrator 多处 dict 类型防御
- **Memory observability/gating** (`src/memory/memory_gate.py`): episode IDs、critical bypass、retrieved/injected/filtered metadata
- **Memory benchmark** (`eval/run_memory_eval.py`): 60 条跨会话场景，支持 no-memory/raw/rule-gate 配对评测
- **BGE retrieval** (`src/memory/embedders.py`, `eval/run_memory_retrieval_eval.py`): BGE-M3 dense/hybrid、embedding 版本隔离、检索排序可观测
- **Memory reranking** (`src/memory/rerankers.py`, `eval/run_memory_reranking_eval.py`): BGE CrossEncoder、dev/test 隔离、阈值 abstention、可配置 hybrid 权重

## 已知发现（P0 消融）

- PPR 在 CMB-Clin 上净贡献接近零（L2-only 下约 −1.3pp，运行方差范围内）
- PPR OFF 的 Top-3 优势（+6.1pp）主要来自 L3 补偿效应（~+4.5pp）
- L3 reflexion 是比 PPR 更强的性能杠杆
- L3 当前行为是"重写而非修正"→ merge guard 已缓解 correct wipe

## 评测入口

```bash
# 全量评测（默认 PPR ON, L3 allowed）
python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl --concurrency 1 --output-dir eval_results/cmb_clin_vllm_v7

# 消融选项
--no-ppr              # PPR OFF
--max-verifier-level L2  # 限制反思最高级别

# 工具诊断
python scripts/diagnose_tool_usage.py --backend vllm --output-dir eval_results/tool_diagnosis

# 误差分析
python scripts/analyze_errors.py --predictions <raw_predictions.jsonl> --tool-diagnosis <item_tool_details.json> --output-dir <dir>

# L3 case 分析
python scripts/analyze_l3_cases.py --v6-l3allowed <predictions> --v6-l2only <predictions> --output-dir <dir>

# V10c Memory benchmark
NO_PROXY=127.0.0.1,localhost python -m eval.run_memory_eval \
  --backend vllm --llm-endpoint http://127.0.0.1:8001/v1 \
  --llm-model Qwen3-8B --output-dir eval_results/memory_v10c_vllm
```

## 目录结构

- `src/agents/planner.py` — Planner Agent（核心 prompt + sanitizer + routing guard）
- `src/orchestrator.py` — MedicalAgentOrchestrator（DAG 执行 + 消融开关）
- `src/verifiers/l3_reflexion.py` — L3 reflexion + merge guard
- `eval/run_eval.py` — 全量评测入口
- `eval/run_ablation.py` — 消融评测入口
- `eval/metrics.py` — 评测指标
- `eval/memory_metrics.py` / `eval/run_memory_eval.py` — Memory 专用指标与评测入口
- `scripts/` — 工具诊断、误差分析、L3 分析脚本
- `reports/` — 工作记录文档
- `eval_results/` — 评测数据（不追踪）
- `data/eval_cmb_clin.jsonl` — CMB-Clin 评测集（77 条）

## 下一阶段

V10c Memory benchmark 已收口：Raw Memory 回答约束通过率 90%，Rule Gate 74%，No Memory 46%。V11b 已完成 12 候选 hard-negative reranking：held-out hybrid rerank Recall@1 50.0% vs dense 30.0%，temporal forbidden@5 16.7% vs 100%；Qwen 约束通过率 83.3% vs 80.0%。由于 Recall@3/5 与 medication 召回下降，reranker 当前保持可选。

建议优先级：
1. **Temporal conflict resolution**：结构化状态、provenance、supersede
2. **Type-aware memory policy**：对 medication/critical memory 使用独立 threshold 或保底召回
3. **真实 L2 verifier**，再重新评估 `_should_trigger_l3` 阈值
4. **实体归一化 / KG retrieval**
