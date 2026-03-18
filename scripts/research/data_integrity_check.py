#!/usr/bin/env python3
"""
Data integrity checks for QA datasets and split manifests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_QA_KEYS = {
    "qa_id",
    "doc_id",
    "bloom_level",
    "question_content",
    "candidate_answers",
    "ground_truth",
    "legal_rationale",
    "is_multimodal",
}


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    data = _read_json(path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _check_records(records: list[dict]) -> dict:
    issues = {
        "missing_required_fields": 0,
        "empty_question": 0,
        "invalid_candidate_answers": 0,
        "duplicate_qa_id": 0,
        "duplicate_question_ground_truth": 0,
    }
    qa_ids = Counter()
    sigs = Counter()

    for r in records:
        missing = REQUIRED_QA_KEYS - set(r.keys())
        if missing:
            issues["missing_required_fields"] += 1
        if not str(r.get("question_content", "")).strip():
            issues["empty_question"] += 1
        ca = r.get("candidate_answers", [])
        if not isinstance(ca, list) or len(ca) < 2:
            issues["invalid_candidate_answers"] += 1
        qa_ids[str(r.get("qa_id", ""))] += 1

        sig = (
            " ".join(str(r.get("question_content", "")).strip().lower().split()),
            " ".join(str(r.get("ground_truth", "")).strip().lower().split()),
        )
        sigs[sig] += 1

    issues["duplicate_qa_id"] = sum(v - 1 for v in qa_ids.values() if v > 1)
    issues["duplicate_question_ground_truth"] = sum(v - 1 for v in sigs.values() if v > 1)
    return issues


def _check_split_leakage(records: list[dict], manifest_path: Path | None) -> dict:
    if not manifest_path or not manifest_path.exists():
        return {"checked": False, "leaky_doc_count": 0, "details": {}}

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, list):
        return {"checked": False, "leaky_doc_count": 0, "details": {}}

    doc_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if doc_id and split:
            doc_splits[doc_id].add(split)

    leak = {doc: sorted(list(s)) for doc, s in doc_splits.items() if len(s) > 1}
    return {"checked": True, "leaky_doc_count": len(leak), "details": leak}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run integrity checks for dataset files.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to .json or .jsonl dataset")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional split manifest JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = _load_records(args.dataset)
    checks = _check_records(records)
    leakage = _check_split_leakage(records, args.manifest)

    report = {
        "dataset_path": str(args.dataset),
        "n_records": len(records),
        "record_checks": checks,
        "split_leakage": leakage,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote integrity report: {args.out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

