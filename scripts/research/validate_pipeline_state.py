#!/usr/bin/env python3
"""
Validate pipeline state before running later stages or paper reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pipeline state and fail fast on blockers.")
    parser.add_argument("--data-base", type=Path, default=Path("data/output"))
    parser.add_argument("--strict", action="store_true", help="Return non-zero on any warning")
    args = parser.parse_args()

    interim = args.data_base / "interim"
    processed = args.data_base / "processed"

    issues = []
    warns = []

    contexts_p = interim / "multimodal_contexts.json"
    if not contexts_p.exists():
        issues.append(f"Missing required file: {contexts_p}")
        contexts = []
    else:
        contexts = _load_json(contexts_p)
        if not isinstance(contexts, list) or len(contexts) == 0:
            issues.append("multimodal_contexts.json is empty or invalid")

    raw_qa_p = interim / "raw_qa_pairs.json"
    raw_qa = []
    if raw_qa_p.exists():
        try:
            d = _load_json(raw_qa_p)
            if isinstance(d, list):
                raw_qa = d
        except Exception:
            issues.append("raw_qa_pairs.json is not valid JSON list")
    else:
        warns.append("raw_qa_pairs.json does not exist yet (Stage 3 not finalized)")

    qa_chunks_dir = interim / "qa_chunks"
    qa_chunk_files = list(qa_chunks_dir.glob("*.json")) if qa_chunks_dir.exists() else []
    if len(raw_qa) == 0 and len(qa_chunk_files) == 0:
        issues.append("No QA artifacts found: both raw_qa_pairs and qa_chunks are empty")
    elif len(raw_qa) == 0 and len(qa_chunk_files) > 0:
        warns.append("raw_qa_pairs is empty; results currently depend on qa_chunks fallback only")

    filtered_p = interim / "filtered_qa_pairs.json"
    if not filtered_p.exists():
        warns.append("filtered_qa_pairs.json missing (Stage 4 not completed)")

    dataset_p = processed / "dataset.jsonl"
    if not dataset_p.exists():
        warns.append("dataset.jsonl missing (final output not completed)")

    print("=== PIPELINE VALIDATION ===")
    print(f"contexts_count={len(contexts) if isinstance(contexts, list) else 0}")
    print(f"raw_qa_count={len(raw_qa)}")
    print(f"qa_chunk_files={len(qa_chunk_files)}")
    print(f"has_filtered={filtered_p.exists()}")
    print(f"has_dataset={dataset_p.exists()}")

    for m in warns:
        print(f"[WARN] {m}")
    for m in issues:
        print(f"[ERROR] {m}")

    if issues:
        sys.exit(2)
    if args.strict and warns:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

