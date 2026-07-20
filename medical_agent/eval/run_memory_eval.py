"""Run controlled, multi-session episodic-memory ablations."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval.memory_metrics import aggregate_memory_results, score_memory_result
from src.factory import build_system
from src.memory.embedders import create_embedder
from src.memory.episodic_memory import EpisodicMemory, MockEmbedder, SQLiteEpisodicBackend
from src.memory.rerankers import create_reranker
from src.memory.semantic_memory import SemanticMemory
from src.schemas import Episode, UserQuery


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "no_long_term": {"injection": False, "gating": False, "temporal_lifecycle": False},
    "long_term_raw": {"injection": True, "gating": False, "temporal_lifecycle": False},
    "long_term_rule_gate": {"injection": True, "gating": True, "temporal_lifecycle": False},
    "long_term_temporal": {"injection": True, "gating": False, "temporal_lifecycle": True},
}
DEFAULT_CONFIGS = ["no_long_term", "long_term_raw", "long_term_rule_gate"]


def _parse_weights(value: str) -> Tuple[float, float, float, float]:
    weights = tuple(float(part.strip()) for part in value.split(","))
    if len(weights) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated weights")
    return weights


def load_scenarios(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_existing(
    output_dir: Path,
    config_names: List[str],
    scenarios: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    labels = {scenario["scenario_id"]: scenario for scenario in (scenarios or [])}
    reports = {}
    for config_name in config_names:
        rows_path = output_dir / f"{config_name}_rows.jsonl"
        with rows_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for row in rows:
            scenario = labels.get(row.get("scenario_id"), {})
            for key in (
                "expected_memory_ids", "forbidden_memory_ids", "must_include",
                "must_not_include", "expected_critical_facts",
            ):
                if key in scenario:
                    row[key] = scenario[key]
        reports[config_name] = aggregate_memory_results(rows)
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    return reports


def _make_episode(memory: EpisodicMemory, user_id: str, spec: Dict[str, Any]) -> Episode:
    episode = Episode(
        episode_id=spec["episode_id"],
        user_id=user_id,
        timestamp=datetime.now() - timedelta(days=float(spec.get("days_ago", 0))),
        episode_type=spec.get("episode_type", "consultation"),
        diagnoses=list(spec.get("diagnoses") or []),
        medications=list(spec.get("medications") or []),
        symptoms=list(spec.get("symptoms") or []),
        summary=spec["summary"],
        status=spec.get("status", "active"),
        provenance=dict(spec.get("provenance") or {"source_type": "evaluation_fixture"}),
        superseded_by=spec.get("superseded_by"),
        importance_score=float(spec.get("importance_score", 0.8)),
    )
    memory._set_embedding(episode, memory.embedder.embed(episode.summary))
    return episode


def _seed_episode(memory: EpisodicMemory, user_id: str, spec: Dict[str, Any]) -> None:
    episode = _make_episode(memory, user_id, spec)
    memory.backend.insert(episode)


def seed_scenario(
    memory: EpisodicMemory,
    scenario: Dict[str, Any],
    temporal_lifecycle: bool = False,
) -> tuple[str, str]:
    target_user = f"memory_eval::{scenario['scenario_id']}::target"
    other_user = f"memory_eval::{scenario['scenario_id']}::other"
    memories = list(scenario.get("memories", []))
    expected = set(scenario.get("expected_memory_ids", []))
    forbidden = set(scenario.get("forbidden_memory_ids", []))
    use_lifecycle = temporal_lifecycle and scenario.get("category") == "temporal"
    for spec in memories:
        scope = spec.get("scope", "target")
        if use_lifecycle and scope == "target" and spec["episode_id"] in expected:
            continue
        _seed_episode(memory, target_user if scope == "target" else other_user, spec)
    if use_lifecycle:
        old_ids = [
            spec["episode_id"] for spec in memories
            if spec.get("scope", "target") == "target" and spec["episode_id"] in forbidden
        ]
        for spec in memories:
            if spec.get("scope", "target") != "target" or spec["episode_id"] not in expected:
                continue
            memory.backend.insert_superseding(_make_episode(memory, target_user, spec), old_ids)
    return target_user, other_user


async def run_config(
    config_name: str,
    scenarios: List[Dict[str, Any]],
    backend: str,
    gate_threshold: float,
    llm_endpoint: str | None,
    llm_model: str | None,
    memory_embedder: str,
    memory_embedding_model: str,
    memory_embedding_device: str | None,
    memory_retrieval_mode: str,
    memory_reranker: str,
    memory_reranker_model: str,
    memory_reranker_device: str | None,
    memory_reranker_candidate_k: int,
    memory_reranker_threshold: float | None,
    memory_retrieval_weights: Tuple[float, float, float, float],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = CONFIGS[config_name]
    agent, _ = build_system(
        backend=backend,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
        enable_memory_gating=cfg["gating"],
        memory_gate_threshold=gate_threshold,
    )
    embedder = create_embedder(
        memory_embedder,
        model_name=memory_embedding_model,
        device=memory_embedding_device,
    )
    memory = EpisodicMemory(
        SQLiteEpisodicBackend(":memory:"), embedder,
        retrieval_mode=memory_retrieval_mode,
        reranker=create_reranker(
            memory_reranker,
            model_name=memory_reranker_model,
            device=memory_reranker_device,
        ),
        reranker_candidate_k=memory_reranker_candidate_k,
        reranker_threshold=memory_reranker_threshold,
        retrieval_weights=memory_retrieval_weights,
    )
    agent.episodic = memory
    isolated_semantic = SemanticMemory(kg_backend=None, embedder=MockEmbedder())
    agent.semantic = isolated_semantic
    if hasattr(agent.verifier, "l3"):
        agent.verifier.l3.semantic_memory = isolated_semantic
    agent.enable_memory_injection = cfg["injection"]
    agent.enable_memory_gating = cfg["gating"]

    rows = []
    for index, scenario in enumerate(scenarios, start=1):
        target_user, _ = seed_scenario(
            memory, scenario, temporal_lifecycle=cfg["temporal_lifecycle"],
        )
        started = time.perf_counter()
        try:
            answer = await agent.answer_async(
                UserQuery(user_id=target_user, text=scenario["query"]),
                session_id=f"memory_eval::{config_name}::{scenario['scenario_id']}::test",
            )
            await agent.flush_memory_writes()
            meta = answer.memory_meta
            episodic = meta["episodic"]
            row = {
                "scenario_id": scenario["scenario_id"],
                "category": scenario["category"],
                "success": True,
                "query": scenario["query"],
                "answer": answer.content,
                "expected_memory_ids": scenario.get("expected_memory_ids", []),
                "forbidden_memory_ids": scenario.get("forbidden_memory_ids", []),
                "must_include": scenario.get("must_include", []),
                "must_not_include": scenario.get("must_not_include", []),
                "expected_critical_facts": scenario.get("expected_critical_facts", []),
                "injected_critical_facts": (
                    meta.get("critical", {}).get("allergies", [])
                    + meta.get("critical", {}).get("chronic_diseases", [])
                ),
                "retrieved_episode_ids": episodic.get("retrieved_episode_ids", []),
                "injected_episode_ids": episodic.get("injected_episode_ids", []),
                "retrieved_count": episodic.get("retrieved_count", 0),
                "injected_count": episodic.get("injected_count", 0),
                "filtered_count": episodic.get("filtered_count", 0),
                "gate_records": episodic.get("gate_records", []),
                "retrieval_records": episodic.get("retrieval_records", []),
                "embedding_model": episodic.get("embedding_model", ""),
                "retrieval_mode": episodic.get("retrieval_mode", ""),
                "retrieval_weights": episodic.get("retrieval_weights", []),
                "reranker_model": episodic.get("reranker_model", ""),
                "reranker_candidate_k": episodic.get("reranker_candidate_k", 0),
                "reranker_threshold": episodic.get("reranker_threshold"),
                "active_only": episodic.get("active_only", False),
                "lifecycle_counts": episodic.get("lifecycle_counts", {}),
                "long_term_context_chars": meta["injection"].get("long_term_context_chars", 0),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
            row["scores"] = score_memory_result(row)
        except Exception as exc:
            row = {
                "scenario_id": scenario["scenario_id"],
                "category": scenario["category"],
                "success": False,
                "error": str(exc),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        rows.append(row)
        print(f"[{config_name}] {index}/{len(scenarios)} {scenario['scenario_id']} success={row['success']}")
    return rows, aggregate_memory_results(rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mock", "db", "vllm"], default="mock")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "eval_memory_scenarios.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "eval_results" / "memory_v10c"))
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=DEFAULT_CONFIGS)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="只运行指定 scenario category")
    parser.add_argument("--num-scenarios", type=int)
    parser.add_argument("--memory-gate-threshold", type=float, default=0.2)
    parser.add_argument("--llm-endpoint", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--memory-embedder", choices=["mock", "bge-m3"], default="mock")
    parser.add_argument("--memory-embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--memory-embedding-device", default=None)
    parser.add_argument("--memory-retrieval-mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--memory-reranker", choices=["none", "identity", "bge-reranker"], default="none")
    parser.add_argument("--memory-reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--memory-reranker-device", default=None)
    parser.add_argument("--memory-reranker-candidate-k", type=int, default=8)
    parser.add_argument("--memory-reranker-threshold", type=float, default=None)
    parser.add_argument(
        "--memory-retrieval-weights", type=_parse_weights,
        default=(0.5, 0.3, 0.15, 0.05),
        help="similarity,importance,time,frequency weights",
    )
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--summarize-only", action="store_true",
                        help="从已有 *_rows.jsonl 重新计算 report.json，不调用 Agent")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        scenarios = load_scenarios(Path(args.data_path))
        reports = summarize_existing(output_dir, args.configs, scenarios=scenarios)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    scenarios = load_scenarios(Path(args.data_path))
    if args.categories:
        categories = set(args.categories)
        scenarios = [scenario for scenario in scenarios if scenario.get("category") in categories]
    if args.split != "all":
        scenarios = [scenario for scenario in scenarios if scenario.get("split") == args.split]
    if args.num_scenarios:
        scenarios = scenarios[:args.num_scenarios]

    reports = {}
    for config_name in args.configs:
        rows, report = await run_config(
            config_name, scenarios, args.backend, args.memory_gate_threshold,
            args.llm_endpoint, args.llm_model,
            args.memory_embedder, args.memory_embedding_model,
            args.memory_embedding_device, args.memory_retrieval_mode,
            args.memory_reranker, args.memory_reranker_model,
            args.memory_reranker_device, args.memory_reranker_candidate_k,
            args.memory_reranker_threshold, args.memory_retrieval_weights,
        )
        reports[config_name] = report
        with (output_dir / f"{config_name}_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
