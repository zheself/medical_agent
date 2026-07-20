"""Evaluate V11c structured temporal lifecycle without requiring an LLM."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.memory.embedders import create_embedder
from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
from src.schemas import Episode


ROOT = Path(__file__).resolve().parent.parent


def load_temporal_scenarios(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [
            row for line in handle if line.strip()
            for row in [json.loads(line)] if row.get("category") == "temporal"
        ]


def make_episode(spec: Dict[str, Any], user_id: str) -> Episode:
    return Episode(
        episode_id=spec["episode_id"],
        user_id=user_id,
        timestamp=datetime.now() - timedelta(days=float(spec.get("days_ago", 0))),
        episode_type=spec.get("episode_type", "consultation"),
        diagnoses=list(spec.get("diagnoses") or []),
        medications=list(spec.get("medications") or []),
        symptoms=list(spec.get("symptoms") or []),
        summary=spec["summary"],
        importance_score=float(spec.get("importance_score", 0.8)),
        provenance={
            "source_type": "evaluation_fixture",
            "source_id": spec["episode_id"],
        },
    )


def seed_scenario(
    memory: EpisodicMemory, scenario: Dict[str, Any], structured: bool,
) -> str:
    user_id = f"v11c::{scenario['scenario_id']}"
    target_specs = [spec for spec in scenario["memories"] if spec.get("scope", "target") == "target"]
    forbidden = set(scenario.get("forbidden_memory_ids", []))
    expected = set(scenario.get("expected_memory_ids", []))

    for spec in target_specs:
        if structured and spec["episode_id"] in expected:
            continue
        episode = make_episode(spec, user_id)
        memory._set_embedding(episode, memory.embedder.embed(episode.summary))
        memory.backend.insert(episode)

    if structured:
        old_ids = [spec["episode_id"] for spec in target_specs if spec["episode_id"] in forbidden]
        for spec in target_specs:
            if spec["episode_id"] not in expected:
                continue
            episode = make_episode(spec, user_id)
            memory._set_embedding(episode, memory.embedder.embed(episode.summary))
            memory.backend.insert_superseding(episode, old_ids)
    return user_id


def evaluate_config(
    scenarios: List[Dict[str, Any]], embedder: Any, structured: bool,
    retrieval_mode: str,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    memory = EpisodicMemory(
        SQLiteEpisodicBackend(":memory:"), embedder, retrieval_mode=retrieval_mode,
    )
    rows = []
    for scenario in scenarios:
        user_id = seed_scenario(memory, scenario, structured)
        retrieved = memory.retrieve(user_id, scenario["query"], top_k=5)
        ids = [episode.episode_id for episode in retrieved]
        expected = set(scenario.get("expected_memory_ids", []))
        forbidden = set(scenario.get("forbidden_memory_ids", []))
        critical = memory.retrieve_critical_facts(user_id)
        row = {
            "scenario_id": scenario["scenario_id"],
            "retrieved_episode_ids": ids,
            "expected_at_1": float(bool(ids) and ids[0] in expected),
            "expected_at_5": float(bool(expected.intersection(ids[:5]))),
            "forbidden_at_1": float(bool(ids) and ids[0] in forbidden),
            "forbidden_at_5": float(bool(forbidden.intersection(ids[:5]))),
            "lifecycle_counts": memory.lifecycle_counts(user_id),
            "critical_facts": critical,
        }
        rows.append(row)

    def avg(key: str) -> float:
        return mean(float(row[key]) for row in rows) if rows else 0.0

    report = {
        "scenario_count": len(rows),
        "expected_at_1_rate": avg("expected_at_1"),
        "expected_at_5_rate": avg("expected_at_5"),
        "forbidden_at_1_rate": avg("forbidden_at_1"),
        "forbidden_at_5_rate": avg("forbidden_at_5"),
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", default=str(ROOT / "data/eval_memory_scenarios.jsonl"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "eval_results/memory_v11c_temporal"),
    )
    parser.add_argument("--embedder", choices=["mock", "bge-m3"], default="mock")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="hybrid")
    args = parser.parse_args()

    scenarios = load_temporal_scenarios(Path(args.data_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = create_embedder(
        args.embedder, model_name=args.embedding_model, device=args.embedding_device,
    )
    reports = {}
    for name, structured in (("unstructured", False), ("structured_lifecycle", True)):
        rows, report = evaluate_config(scenarios, embedder, structured, args.retrieval_mode)
        reports[name] = report
        with (output_dir / f"{name}_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
