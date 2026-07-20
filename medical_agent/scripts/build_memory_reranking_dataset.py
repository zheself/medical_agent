"""Build the V11b hard-negative reranking benchmark from V10c scenarios."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

try:
    from scripts.build_memory_eval_dataset import build_scenarios
except ModuleNotFoundError:
    from build_memory_eval_dataset import build_scenarios


ROOT = Path(__file__).resolve().parent.parent
DEV_INDICES = {1, 4, 7, 10}
TARGET_CANDIDATES = 12


def _primary_memory(scenario: Dict[str, Any]) -> Dict[str, Any] | None:
    expected = set(scenario.get("expected_memory_ids") or [])
    return next((m for m in scenario["memories"] if m["episode_id"] in expected), None)


def build_reranking_scenarios() -> List[Dict[str, Any]]:
    base = build_scenarios()
    primaries = [(scenario, _primary_memory(scenario)) for scenario in base]
    output = []
    for scenario in base:
        row = deepcopy(scenario)
        index = int(row["scenario_id"].rsplit("_", 1)[1])
        row["split"] = "dev" if index in DEV_INDICES else "test"
        own_summaries = {m["summary"] for m in row["memories"] if m["scope"] == "target"}
        pool = [
            (source, memory) for source, memory in primaries
            if memory and source["scenario_id"] != row["scenario_id"]
        ]
        pool.sort(key=lambda item: (item[0]["category"] != row["category"], item[0]["scenario_id"]))
        added = 0
        for source, memory in pool:
            if len([m for m in row["memories"] if m["scope"] == "target"]) >= TARGET_CANDIDATES:
                break
            clone = deepcopy(memory)
            clone["scope"] = "target"
            clone["episode_id"] = f"hard::{row['scenario_id']}::{source['scenario_id']}"
            # Exact duplicates make ranking artificially easy and are not hard negatives.
            if clone["summary"] in own_summaries:
                continue
            row["memories"].append(clone)
            own_summaries.add(clone["summary"])
            added += 1
        if row["category"] == "irrelevant":
            row["forbidden_memory_ids"] = [
                m["episode_id"] for m in row["memories"] if m["scope"] == "target"
            ]
        row["candidate_count"] = len([m for m in row["memories"] if m["scope"] == "target"])
        assert row["candidate_count"] >= 10 and added > 0
        output.append(row)
    assert len(output) == 60
    assert sum(row["split"] == "dev" for row in output) == 24
    return output


def main() -> None:
    output = ROOT / "data" / "eval_memory_reranking_v11b.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for scenario in build_reranking_scenarios():
            handle.write(json.dumps(scenario, ensure_ascii=False) + "\n")
    print(f"Wrote 60 V11b scenarios to {output}")


if __name__ == "__main__":
    main()
