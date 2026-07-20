"""Evaluate episodic retrieval without invoking the diagnosis LLM."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.memory.embedders import create_embedder
from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
from src.schemas import Episode


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "mock_dense": ("mock", "dense"),
    "mock_hybrid": ("mock", "hybrid"),
    "bge_dense": ("bge-m3", "dense"),
    "bge_hybrid": ("bge-m3", "hybrid"),
}


def _load(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _seed_all(memory: EpisodicMemory, scenarios: List[Dict[str, Any]]) -> None:
    pending = []
    for scenario in scenarios:
        for spec in scenario.get("memories", []):
            scope = spec.get("scope", "target")
            user_id = f"retrieval::{scenario['scenario_id']}::{scope}"
            pending.append((user_id, spec))
    vectors = memory.embedder.embed_many([spec["summary"] for _, spec in pending])
    for (user_id, spec), vector in zip(pending, vectors):
        episode = Episode(
            episode_id=spec["episode_id"],
            user_id=user_id,
            timestamp=datetime.now() - timedelta(days=float(spec.get("days_ago", 0))),
            episode_type=spec.get("episode_type", "consultation"),
            diagnoses=list(spec.get("diagnoses") or []),
            medications=list(spec.get("medications") or []),
            symptoms=list(spec.get("symptoms") or []),
            summary=spec["summary"],
            importance_score=float(spec.get("importance_score", 0.8)),
        )
        memory._set_embedding(episode, vector)
        memory.backend.insert(episode)


def _reciprocal_rank(ids: List[str], expected: set[str]) -> float:
    for rank, episode_id in enumerate(ids, start=1):
        if episode_id in expected:
            return 1.0 / rank
    return 0.0


def _ndcg(ids: List[str], expected: set[str], k: int) -> float:
    if not expected:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, episode_id in enumerate(ids[:k], start=1)
        if episode_id in expected
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(k, len(expected)) + 1)
    )
    return dcg / ideal if ideal else 0.0


def _aggregate(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    relevant = [row for row in rows if row["expected_memory_ids"]]

    def mean(key: str, source: List[Dict[str, Any]] = relevant) -> float:
        return sum(float(row[key]) for row in source) / len(source) if source else 0.0

    report = {
        "scenario_count": len(rows),
        "recall_at_1": mean("recall_at_1"),
        "recall_at_2": mean("recall_at_2"),
        "recall_at_5": mean("recall_at_5"),
        "mrr": mean("mrr"),
        "ndcg_at_5": mean("ndcg_at_5"),
        "forbidden_at_1_rate": sum(row["forbidden_at_1"] for row in rows) / len(rows),
        "forbidden_at_5_rate": sum(row["forbidden_at_5"] for row in rows) / len(rows),
        "cross_user_leakage_rate": sum(row["cross_user_leakage"] for row in rows) / len(rows),
        "avg_query_ms": mean("elapsed_ms", rows),
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    report["by_category"] = {
        category: {
            "count": len(items),
            "recall_at_1": mean("recall_at_1", [r for r in items if r["expected_memory_ids"]]),
            "mrr": mean("mrr", [r for r in items if r["expected_memory_ids"]]),
            "forbidden_at_1_rate": mean("forbidden_at_1", items),
            "forbidden_at_5_rate": mean("forbidden_at_5", items),
        }
        for category, items in sorted(grouped.items())
    }
    return report


def run_config(
    config_name: str,
    scenarios: List[Dict[str, Any]],
    embedder,
    top_k: int = 5,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _, retrieval_mode = CONFIGS[config_name]
    memory = EpisodicMemory(
        SQLiteEpisodicBackend(":memory:"), embedder, retrieval_mode=retrieval_mode,
    )
    _seed_all(memory, scenarios)
    rows = []
    for scenario in scenarios:
        started = time.perf_counter()
        episodes = memory.retrieve(
            f"retrieval::{scenario['scenario_id']}::target",
            scenario["query"],
            top_k=top_k,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        ids = [episode.episode_id for episode in episodes]
        expected = set(scenario.get("expected_memory_ids") or [])
        forbidden = set(scenario.get("forbidden_memory_ids") or [])
        row = {
            "scenario_id": scenario["scenario_id"],
            "category": scenario["category"],
            "query": scenario["query"],
            "expected_memory_ids": sorted(expected),
            "forbidden_memory_ids": sorted(forbidden),
            "retrieved_episode_ids": ids,
            "retrieval_records": [
                {
                    "episode_id": episode.episode_id,
                    "score": episode.retrieval_score,
                    "components": episode.retrieval_components,
                }
                for episode in episodes
            ],
            "recall_at_1": float(bool(expected.intersection(ids[:1]))) if expected else 1.0,
            "recall_at_2": float(bool(expected.intersection(ids[:2]))) if expected else 1.0,
            "recall_at_5": float(bool(expected.intersection(ids[:5]))) if expected else 1.0,
            "mrr": _reciprocal_rank(ids, expected) if expected else 1.0,
            "ndcg_at_5": _ndcg(ids, expected, 5),
            "forbidden_at_1": float(bool(forbidden.intersection(ids[:1]))),
            "forbidden_at_5": float(bool(forbidden.intersection(ids[:5]))),
            "cross_user_leakage": float(any(eid.startswith("other::") for eid in ids)),
            "elapsed_ms": elapsed_ms,
        }
        rows.append(row)
    return rows, _aggregate(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "data" / "eval_memory_scenarios.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "eval_results" / "memory_v11_retrieval"))
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    scenarios = _load(Path(args.data_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedders = {}
    reports = {}
    for config_name in args.configs:
        embedder_name, _ = CONFIGS[config_name]
        if embedder_name not in embedders:
            embedders[embedder_name] = create_embedder(
                embedder_name,
                model_name=args.model_name,
                device=args.device,
                batch_size=args.batch_size,
                local_files_only=args.local_files_only,
            )
        rows, report = run_config(config_name, scenarios, embedders[embedder_name], args.top_k)
        reports[config_name] = report
        with (output_dir / f"{config_name}_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{config_name}: {json.dumps(report, ensure_ascii=False)}")
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
