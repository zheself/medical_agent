# AGENTS.md

## Project Overview

This workspace contains a Python medical diagnosis reasoning Agent prototype. The runnable project lives under `medical_agent/`.

The system combines:

- Planner-Executor orchestration
- Knowledge graph tools and GraphRAG-style retrieval
- Working, episodic, and semantic memory
- Graded verification with L1 rules, L2 model verifier stubs, and L3 reflexion
- Mock, SQLite database, and vLLM backend paths

Treat all medical content in this repository as research/prototype data unless it is explicitly wired to validated production sources and reviewed. Do not present outputs as medical advice.

## Quick Handoff

New agents should read this section first.

Project summary: this is a medical diagnosis reasoning Agent with Planner-Executor orchestration, three-layer memory, graded reflection, and GraphRAG. Core code is in `medical_agent/src/`; architecture docs are in `medical_agent/docs/`, starting with `medical_agent/docs/01_architecture.md`.

Backend modes are switched through `src.factory.build_system(backend=...)`:

- `mock`: in-memory hard-coded data, zero external dependencies, primarily for tests.
- `db`: reads the real SQLite KG from `data/medical_agent.db`, while the LLM remains `MockLLMBackend`.
- `vllm`: uses real Qwen3-8B through a vLLM service. Start the vLLM service before using this mode.

Current progress:

- P0 real vLLM integration is complete.
- P1 real KG import is complete.
- PPR IDF edge weighting and type filtering are complete.
- The evaluation pipeline is working.
- CMB-Clin evaluation data has 77 items; the adapter quality and loose-match rules have been validated.
- Current baseline: **V8** — L3 Phase 2 (prompt rewrite + trigger guard) on top of Phase 1 (merge guard).
  - Top-3 loose **45.5%**, Top-5 49.4%, Hard Top-3 47.5%, 0 errors (77/77).
  - V7 holds historical best: Top-3 46.7%, Hard Top-3 50.0%.
  - L3 reflexion rate 9.1% — all triggers are `empty_diagnoses`, 0 correct wipes.
  - L3 merge guard + trigger guard enabled. `verification_meta` field tracks trigger/skip reasons.
  - Diagnosis baseline remains V8; **103/103 tests passing** after V11b engineering work.
- V9 hardened DAG tool parallelism and added `execution_meta`; the synthetic sleep benchmark reached 2.98x parallelism.
- V10a-c completed Memory observability, rule gating, lifecycle correctness, and a 60-scenario multi-session benchmark.
  - Qwen3-8B V10c: Raw Memory raised answer constraint pass from 46% to 90%, but injected 2.0 irrelevant memories/item.
  - Rule Gate reduced episodic context by 86.3%, but episodic injection recall fell to 28%; critical-profile recall remained 100%.
  - Cross-user leakage was 0% for all configurations.
- Key finding from P0 ablation: PPR net contribution on CMB-Clin is near zero (≈−1.3pp @ L2-only, within run variance).
  The PPR OFF Top-3 advantage (+6.1pp) was mostly L3 reflexion compensation, not PPR quality.
- V11b completed hard-negative reranking with a 24/36 dev/test split. On held-out test, hybrid reranking raises Recall@1 from 30.0% to 50.0% and reduces temporal forbidden@5 from 100% to 16.7%, but lowers Recall@3/5. Qwen answer constraint pass changes from 80.0% to 83.3%, so reranking remains optional.

First things to do when taking over:

1. Confirm the environment: `conda activate cjz_opd`. The reports record this environment as having `vllm`, `igraph`, `leidenalg`, and `networkx`.
2. Run `python tests/run_all.py` from `medical_agent/`; the expected baseline is **103/103 passing**.
3. Start vLLM on port 8001 and run `python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl --concurrency 1` to validate the real pipeline.
4. Read `medical_agent/reports/2026-07-17_V11b_reranking_record.md`, `medical_agent/reports/2026-07-16_V11a_bge_retrieval_record.md`, and `medical_agent/reports/2026-07-15_V10c_memory_benchmark_record.md` for the latest work records.

Hard rules:

- Keep all three backend modes: `mock`, `db`, and `vllm`.
- After code changes, run `tests/run_all.py` unless the change is documentation-only.
- For major changes, write the plan first and get user confirmation.

## Current Project State

Use `medical_agent/reports/` as the most current work log. It is more up to date than parts of the README.

Important completed milestones:

- P0 vLLM integration is complete: `backend="vllm"` uses Qwen3-8B through a vLLM OpenAI-compatible service.
- P1 real KG import is complete: `data/medical_agent.db` has been replaced with the imported medical KG from `data/medical_new_2.json`.
- Current KG scale is approximately 22,480 entities, 303,143 relations, 22,479 PageRank nodes, and Leiden communities L0=4, L1=14, L2=26.
- PPR IDF edge weighting and disease-type filtering have been implemented.
- CMB-Clin evaluation data has been adapted to `data/eval_cmb_clin.jsonl` with 77 items.
- **V8 diagnosis baseline**: Top-3 loose 45.5%, Hard Top-3 47.5%, 0 errors. V7 holds historical best Top-3 (46.7%). Current test baseline is 103/103.
- **V10c Memory benchmark**: 60 scenarios, Qwen3-8B, 180/180 successful. Raw Memory answer constraint pass 90%; Rule Gate 74%; no-memory 46%; cross-user leakage 0%.
- Planner sanitizer (`_sanitize_plan`) normalizes depends_on types, removes duplicate tool calls, and cleans invalid dependencies.
- Routing guard (`_route_complexity`) applies rules-based complexity correction on top of LLM judgment.
- L3 merge guard prevents L3 reflexion from overwriting correct L2 diagnoses.
- PPR OFF ablation switch and `--max-verifier-level` flag available in `eval/run_eval.py`.
- Key P0 finding: PPR net contribution on CMB-Clin is near zero (≈−1.3pp @ L2-only). L3 reflexion is a stronger lever.
- **V11a BGE retrieval**: real BGE-M3 dense Recall@1 72%, MRR 0.850; hybrid reduces temporal forbidden@1 from 80% to 0% but reduces overall MRR to 0.793. Cross-user leakage remains 0%.
- **V11b reranking**: 12-candidate hard-negative benchmark with frozen dev tuning. Held-out Recall@1 is 50.0% vs dense 30.0%; temporal forbidden@5 is 16.7% vs 100%; Qwen constraint pass is 83.3% vs 80.0%. Recall@3/5 regressions keep it opt-in.
- Detailed work records: `reports/2026-07-17_V11b_reranking_record.md`, `reports/2026-07-16_V11a_bge_retrieval_record.md`, `reports/2026-07-15_V10c_memory_benchmark_record.md`, and the earlier P0/L3 reports.

## Directory Map

- `medical_agent/README.md`: main project documentation and quick start.
- `medical_agent/src/`: core package.
- `medical_agent/src/factory.py`: preferred system assembly entry point.
- `medical_agent/demo/demo_full_flow.py`: end-to-end demo.
- `medical_agent/tests/`: dependency-light tests, runnable without pytest.
- `medical_agent/eval/`: evaluation runners and metrics.
- `medical_agent/scripts/`: offline data, KG, GraphRAG, and training scripts.
- `medical_agent/app/streamlit_app.py`: optional Streamlit UI.
- `medical_agent/data/`: local demo/evaluation data and SQLite files.
- `medical_agent/docs/`: architecture and design notes.
- `medical_agent/reports/`: chronological work records and the best source for current status.
- Top-level `factory.py` and `seed_database.py`: older copies/shims; prefer the versions under `medical_agent/src/` and `medical_agent/scripts/`.

## Setup And Commands

Run commands from `medical_agent/` unless noted otherwise:

```bash
cd medical_agent
python -m demo.demo_full_flow
python scripts/seed_database.py --force
python -m demo.demo_full_flow --db
python -m tests.run_all
python -m eval.run_eval --backend mock --num-items 5
python scripts/diagnose_tool_usage.py
```

Optional UI:

```bash
cd medical_agent
streamlit run app/streamlit_app.py
```

Production-oriented dependencies are listed in `medical_agent/requirements.txt`, but the mock demo and the core tests are intended to run with the Python standard library.

For real vLLM evaluation, use the project environment recorded in the reports:

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

Then run:

```bash
cd medical_agent
python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl --concurrency 1
```

## Development Guidelines

- Prefer `src.factory.build_system()` for constructing the Agent. Keep backend switching centralized there.
- Preserve the mock path. It is the fastest way to validate orchestration, memory, tool calls, and verifier behavior without GPU services.
- Keep SQLite-backed behavior compatible with `scripts/seed_database.py`.
- Do not assume `data/medical_agent.db` is only the 45-entity seed database. Current reports indicate it has been replaced by the full imported KG.
- Avoid hard-coding new medical facts directly into orchestration logic. Put demo data in seed scripts, mock backends, or clearly scoped test fixtures.
- Keep interfaces between planner, tools, memory, verifiers, and schemas stable. When changing contracts, update tests and docs together.
- Use structured data from `src/schemas.py` instead of loose dictionaries where a schema already exists.
- Do not commit generated evaluation outputs, local databases, model checkpoints, or secrets unless the user explicitly asks for a reproducible fixture.
- Keep `.env` local. Use `.env.example` for documented configuration.
- For eval changes, preserve isolated per-item user IDs. Do not reintroduce a shared `eval_user`.

## Testing Expectations

Before handing off code changes, run at least:

```bash
cd medical_agent
python -m tests.run_all
```

For changes touching evaluation:

```bash
cd medical_agent
python -m eval.run_eval --backend mock --num-items 5
```

For changes touching SQLite backends or seed data:

```bash
cd medical_agent
python scripts/seed_database.py --force
python -m tests.run_all
python -m demo.demo_full_flow --db
```

If a command depends on unavailable services such as Neo4j, Milvus, vLLM, or a GPU, document that it was not run and explain what service is required.

## Medical Safety Notes

- This repository is a research/prototype system. Generated answers must include appropriate uncertainty and escalation language where relevant.
- Do not remove L1 safety checks casually. Drug allergy, contraindication, emergency symptom, pediatric, pregnancy, and other high-risk checks should remain cheap and deterministic where possible.
- Prefer conservative behavior for red flags such as high fever with neck stiffness, chest pain, stroke-like symptoms, severe allergic reaction, or suicidal ideation.
- Any productionization plan must include validated medical data sources, clinician review, audit logs, privacy review, and regulatory assessment.

## Current Priority Plan

1. **Temporal Memory**: add structured status, provenance, and supersede handling for conflicting old/new facts.
2. **Type-aware memory policy**: avoid one global threshold for medication and critical memories; recover recall while retaining abstention.
3. **Real L2 verifier**: replace MockSmallModel and recalibrate the L3 trigger thresholds.
4. **KG retrieval**: implement `SQLiteCommunityStore` and improve entity normalization / NER.
