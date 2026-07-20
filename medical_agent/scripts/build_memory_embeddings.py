"""Rebuild episodic-memory embeddings in a copied or explicitly mutable DB."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.memory.embedders import create_embedder
from src.memory.episodic_memory import EpisodicMemory, SQLiteEpisodicBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True, help="source SQLite database")
    parser.add_argument("--output-db", help="copy source here before indexing")
    parser.add_argument("--in-place", action="store_true", help="explicitly modify --db-path")
    parser.add_argument("--embedder", choices=["mock", "bge-m3"], default="bge-m3")
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    source = Path(args.db_path).resolve()
    if args.in_place == bool(args.output_db):
        parser.error("choose exactly one of --output-db or --in-place")
    target = source if args.in_place else Path(args.output_db).resolve()
    if not source.exists():
        parser.error(f"database does not exist: {source}")
    if not args.in_place:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    embedder = create_embedder(
        args.embedder,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    memory = EpisodicMemory(SQLiteEpisodicBackend(str(target)), embedder)
    count = memory.reindex(batch_size=args.batch_size)
    print(f"indexed={count} model={memory.embedding_model_id} db={target}")


if __name__ == "__main__":
    main()
