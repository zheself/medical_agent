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
- Full vLLM evaluation has been run twice: V1 baseline and V2 after memory isolation.
- Key finding: PPR call rate is only 6.5%. The root cause is Planner complexity classification: 93.5% of items are classified as `medium`, so they only trigger `local_search`.
- The episodic memory noise bug from cross-item eval pollution has been fixed.

First things to do when taking over:

1. Confirm the environment: `conda activate cjz_opd`. The reports record this environment as having `vllm`, `igraph`, `leidenalg`, and `networkx`.
2. Run `python tests/run_all.py` from `medical_agent/`; the expected historical baseline is 38/38 passing.
3. Start vLLM on port 8001 and run `python -m eval.run_eval --backend vllm --data-path data/eval_cmb_clin.jsonl` to validate the real pipeline.
4. Read sections 5e and 5f of `medical_agent/reports/2026-05-30_P1_report.md` before changing Planner routing or evaluation logic.

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
- Full vLLM evaluation has been run twice:
  - V1 baseline: Top-3 loose 42.1%, mean latency about 30.6s.
  - V2 after memory isolation and PPR tool-description cleanup: Top-3 loose still 42.1%, mean latency about 24.7s, mean tokens down about 34%.
- A major eval bug was fixed: each eval item now uses an isolated `user_id=f"eval_{item_id}"` to avoid episodic memory cross-item pollution.
- Planner tool diagnosis has been run. PPR/global search call rate is only 6.5% because 93.5% of CMB-Clin items are classified as `medium` and routed to `ner + kg_local_search`.

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

1. P0: adjust Planner complexity/routing so clinical diagnosis and differential-diagnosis items are classified as `high` often enough to trigger `kg_global_search` and `ppr_reasoner`.
2. P0: increase the number of structured diagnosis candidates. Current full eval averages about two candidates, so Top-3 and Top-5 are identical.
3. P0: rerun `scripts/diagnose_tool_usage.py` after routing changes. Target a meaningful increase from the current 6.5% PPR/global-search call rate before running expensive full eval.
4. P0: rerun CMB-Clin vLLM evaluation and then run ablation once routing actually changes tool usage.
5. P1: implement `SQLiteCommunityStore` so global search uses imported Leiden communities rather than mock community data.
6. P2: improve KG retrieval depth/performance for PPR and local search, especially around `SQLiteKGBackend.query_neighbors`.
7. P2: upgrade NER from hard-coded matching to Trie/dictionary normalization first, then consider a model. Real KG entity variants are currently a major recall bottleneck.
8. P2: decide how much entity normalization is worth doing. Reports show synonym/granularity mismatch is real, but full normalization can become unbounded.
