"""V11b dev-tuned episodic-memory reranking evaluation."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.run_memory_retrieval_eval import _load, _seed_all
from src.memory.embedders import create_embedder
from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend
from src.memory.rerankers import create_reranker


ROOT = Path(__file__).resolve().parent.parent
INJECTION_K = 5


def _mrr(ids: List[str], expected: set[str]) -> float:
    for rank, episode_id in enumerate(ids, 1):
        if episode_id in expected:
            return 1.0 / rank
    return 0.0


def _ndcg(ids: List[str], expected: set[str], k: int = 5) -> float:
    if not expected:
        return 1.0
    dcg = sum(1 / math.log2(i + 1) for i, eid in enumerate(ids[:k], 1) if eid in expected)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(expected)) + 1))
    return dcg / ideal if ideal else 0.0


def collect_candidates(scenarios, embedder, reranker=None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    memory = EpisodicMemory(SQLiteEpisodicBackend(":memory:"), embedder, retrieval_mode="dense")
    _seed_all(memory, scenarios)
    reranker_warmup_ms = 0.0
    if reranker:
        started = time.perf_counter()
        reranker.score("warmup", ["warmup"])
        reranker_warmup_ms = (time.perf_counter() - started) * 1000
    rows = []
    for scenario in scenarios:
        count = int(scenario.get("candidate_count", 20))
        started = time.perf_counter()
        episodes = memory.retrieve(
            f"retrieval::{scenario['scenario_id']}::target", scenario["query"], top_k=count,
        )
        dense_ms = (time.perf_counter() - started) * 1000
        reranker_scores = None
        reranker_ms = 0.0
        if reranker:
            started = time.perf_counter()
            reranker_scores = reranker.score(scenario["query"], [ep.summary for ep in episodes])
            reranker_ms = (time.perf_counter() - started) * 1000
        candidates = []
        for index, episode in enumerate(episodes):
            candidates.append({
                "episode_id": episode.episode_id,
                "summary": episode.summary,
                **episode.retrieval_components,
                "reranker_score": reranker_scores[index] if reranker_scores is not None else None,
            })
        rows.append({
            "scenario_id": scenario["scenario_id"], "category": scenario["category"],
            "split": scenario["split"], "query": scenario["query"],
            "expected_memory_ids": scenario.get("expected_memory_ids", []),
            "forbidden_memory_ids": scenario.get("forbidden_memory_ids", []),
            "candidates": candidates, "dense_ms": dense_ms, "reranker_ms": reranker_ms,
        })
    return rows, {"reranker_warmup_ms": reranker_warmup_ms}


def rank_row(
    row: Dict[str, Any], mode: str, weights: Tuple[float, float, float, float],
    candidate_k: int, threshold: Optional[float] = None,
) -> Dict[str, Any]:
    ws, wi, wt, wf = weights
    candidates = []
    for candidate in row["candidates"]:
        base_score = (
            ws * candidate["similarity"] + wi * candidate["importance"]
            + wt * candidate["time_decay"] + wf * candidate["frequency"]
        )
        item = dict(candidate, base_score=base_score)
        candidates.append(item)
    candidates.sort(key=lambda item: item["base_score"], reverse=True)
    if mode.endswith("rerank"):
        candidates = candidates[:candidate_k]
        candidates.sort(key=lambda item: float(item["reranker_score"]), reverse=True)
        if threshold is not None:
            candidates = [item for item in candidates if float(item["reranker_score"]) >= threshold]
    ids = [item["episode_id"] for item in candidates]
    expected, forbidden = set(row["expected_memory_ids"]), set(row["forbidden_memory_ids"])
    reranker_ms = row["reranker_ms"] if mode.endswith("rerank") else 0.0
    return {
        **{key: row[key] for key in ("scenario_id", "category", "split")},
        "retrieved_episode_ids": ids,
        "recall_at_1": float(bool(expected.intersection(ids[:1]))) if expected else 1.0,
        "recall_at_3": float(bool(expected.intersection(ids[:3]))) if expected else 1.0,
        "recall_at_5": float(bool(expected.intersection(ids[:5]))) if expected else 1.0,
        "mrr": _mrr(ids, expected) if expected else 1.0,
        "ndcg_at_5": _ndcg(ids, expected),
        "forbidden_at_1": float(bool(forbidden.intersection(ids[:1]))),
        "forbidden_at_5": float(bool(forbidden.intersection(ids[:INJECTION_K]))),
        "irrelevant_injection": float(not expected and bool(ids)),
        "cross_user_leakage": float(any(eid.startswith("other::") for eid in ids)),
        "abstained": float(not ids),
        "dense_ms": row["dense_ms"], "reranker_ms": reranker_ms,
        "total_retrieval_ms": row["dense_ms"] + reranker_ms,
        "ranking": candidates,
    }


def report(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    relevant = [row for row in rows if not row["category"] == "irrelevant"]
    temporal = [row for row in rows if row["category"] == "temporal"]
    irrelevant = [row for row in rows if row["category"] == "irrelevant"]
    mean = lambda key, data: sum(row[key] for row in data) / len(data) if data else 0.0
    percentile = lambda key, data, q: (
        sorted(float(row[key]) for row in data)[min(len(data) - 1, math.ceil(q * len(data)) - 1)]
        if data else 0.0
    )
    result = {
        "count": len(rows), "recall_at_1": mean("recall_at_1", relevant),
        "recall_at_3": mean("recall_at_3", relevant), "recall_at_5": mean("recall_at_5", relevant),
        "mrr": mean("mrr", relevant), "ndcg_at_5": mean("ndcg_at_5", relevant),
        "forbidden_at_1_rate": mean("forbidden_at_1", temporal),
        "forbidden_at_5_rate": mean("forbidden_at_5", temporal),
        "irrelevant_injection_rate": mean("irrelevant_injection", irrelevant),
        "cross_user_leakage_rate": mean("cross_user_leakage", rows),
        "abstention_rate": mean("abstained", rows),
        "relevant_abstention_rate": mean("abstained", relevant),
        "avg_dense_ms": mean("dense_ms", rows), "avg_reranker_ms": mean("reranker_ms", rows),
        "p50_reranker_ms": percentile("reranker_ms", rows, 0.50),
        "p95_reranker_ms": percentile("reranker_ms", rows, 0.95),
        "avg_total_retrieval_ms": mean("total_retrieval_ms", rows),
        "p95_total_retrieval_ms": percentile("total_retrieval_ms", rows, 0.95),
    }
    result["selection_score"] = (
        result["mrr"] - 0.25 * result["forbidden_at_5_rate"]
        - 0.20 * result["irrelevant_injection_rate"]
    )
    return result


def evaluate(rows, mode, weights, candidate_k, threshold=None):
    ranked = [rank_row(row, mode, weights, candidate_k, threshold) for row in rows]
    return ranked, report(ranked)


def tune_hybrid(dev_rows, candidate_k):
    best = None
    for similarity, importance, time_weight in itertools.product(
        (0.5, 0.65, 0.8, 0.9), (0.0, 0.1, 0.2), (0.0, 0.1, 0.2, 0.3),
    ):
        weights = (similarity, importance, time_weight, 0.0)
        _, metrics = evaluate(dev_rows, "hybrid", weights, candidate_k)
        if best is None or metrics["selection_score"] > best[1]["selection_score"]:
            best = (weights, metrics)
    return best


def tune_threshold(dev_rows, mode, weights, candidate_k):
    unfiltered, _ = evaluate(dev_rows, mode, weights, candidate_k, threshold=None)
    scores = sorted({
        float(candidate["reranker_score"])
        for row in unfiltered for candidate in row["ranking"]
    })
    # Include an explicit abstain-all option. Using only observed scores cannot
    # reject a candidate whose score equals the largest development score.
    thresholds = [None] + scores
    if scores:
        thresholds.append(math.nextafter(scores[-1], math.inf))
    best = None
    for threshold in thresholds:
        _, metrics = evaluate(dev_rows, mode, weights, candidate_k, threshold)
        if best is None or metrics["selection_score"] > best[1]["selection_score"]:
            best = (threshold, metrics)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ROOT / "data/eval_memory_reranking_v11b.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "eval_results/memory_v11b_reranking"))
    parser.add_argument("--embedder", choices=["mock", "bge-m3"], default="mock")
    parser.add_argument("--embedding-model", default="data/models/bge-m3")
    parser.add_argument("--reranker", choices=["identity", "bge-reranker"], default="identity")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=8)
    args = parser.parse_args()

    scenarios = _load(Path(args.data_path))
    embedder = create_embedder(args.embedder, model_name=args.embedding_model, device=args.device,
                               batch_size=args.batch_size)
    reranker = create_reranker(args.reranker, model_name=args.reranker_model,
                               device=args.device, batch_size=args.batch_size)
    raw_rows, timing = collect_candidates(scenarios, embedder, reranker)
    dev, test = ([r for r in raw_rows if r["split"] == split] for split in ("dev", "test"))
    dense_weights = (1.0, 0.0, 0.0, 0.0)
    hybrid_weights, hybrid_dev = tune_hybrid(dev, args.candidate_k)
    dense_threshold, dense_rerank_dev = tune_threshold(
        dev, "dense_rerank", dense_weights, args.candidate_k,
    )
    hybrid_threshold, hybrid_rerank_dev = tune_threshold(
        dev, "hybrid_rerank", hybrid_weights, args.candidate_k,
    )
    configs = {
        "dense": ("dense", dense_weights, None),
        "hybrid_tuned": ("hybrid", hybrid_weights, None),
        "dense_rerank": ("dense_rerank", dense_weights, dense_threshold),
        "hybrid_rerank": ("hybrid_rerank", hybrid_weights, hybrid_threshold),
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    result = {"runtime": timing, "tuning": {
        "hybrid_weights": hybrid_weights, "hybrid_dev": hybrid_dev,
        "dense_rerank_threshold": dense_threshold, "dense_rerank_dev": dense_rerank_dev,
        "hybrid_rerank_threshold": hybrid_threshold, "hybrid_rerank_dev": hybrid_rerank_dev,
    }, "test": {}}
    for name, (mode, weights, threshold) in configs.items():
        ranked, metrics = evaluate(test, mode, weights, args.candidate_k, threshold)
        result["test"][name] = metrics
        with (output / f"{name}_test_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in ranked:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
