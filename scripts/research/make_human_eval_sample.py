#!/usr/bin/env python3
"""
Create a stratified human-evaluation sample from QA records.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def _read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _load_from_qa_chunks(qa_chunks_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not qa_chunks_dir.exists():
        return rows
    for p in sorted(qa_chunks_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows.extend(x for x in data if isinstance(x, dict))
            elif isinstance(data, dict):
                rows.append(data)
        except Exception:
            continue
    return rows


def _qa_to_row(qa: dict, idx: int) -> dict:
    context_text = str(qa.get("context_text", ""))
    if not context_text and isinstance(qa.get("context_payload"), dict):
        context_text = str(qa["context_payload"].get("text", ""))

    return {
        "item_id": f"item_{idx:04d}",
        "qa_id": str(qa.get("qa_id", "")),
        "doc_id": str(qa.get("doc_id", "")),
        "bloom_level": str(qa.get("bloom_level", "")),
        "is_multimodal": int(bool(qa.get("is_multimodal", False))),
        "question_content": str(qa.get("question_content", "")),
        "candidate_answers": " || ".join(qa.get("candidate_answers", []) if isinstance(qa.get("candidate_answers", []), list) else []),
        "ground_truth": str(qa.get("ground_truth", "")),
        "legal_rationale": str(qa.get("legal_rationale", "")),
        "context_excerpt": context_text[:600],
        "ann_legal_correctness": "",
        "ann_groundedness": "",
        "ann_question_quality": "",
        "ann_distractor_quality": "",
        "ann_visual_alignment": "",
        "ann_comments": "",
    }


def _stratified_sample(records: list[dict], target_n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    strata: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        key = (str(r.get("bloom_level", "Unknown")), int(bool(r.get("is_multimodal", False))))
        strata[key].append(r)

    keys = sorted(strata.keys())
    if not keys:
        return []

    per = max(1, target_n // len(keys))
    sample: list[dict] = []
    for k in keys:
        bucket = strata[k]
        rng.shuffle(bucket)
        sample.extend(bucket[: min(per, len(bucket))])

    if len(sample) < target_n:
        remaining = [x for x in records if x not in sample]
        rng.shuffle(remaining)
        sample.extend(remaining[: max(0, target_n - len(sample))])

    return sample[:target_n]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create human evaluation sheets.")
    parser.add_argument("--input", type=Path, required=True, help="QA JSON/JSONL source")
    parser.add_argument("--n", type=int, default=240, help="Target sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qa-chunks-dir", type=Path, default=None)
    args = parser.parse_args()

    records = [r for r in _read_records(args.input) if isinstance(r, dict)]
    if not records and args.qa_chunks_dir is not None:
        records = _load_from_qa_chunks(args.qa_chunks_dir)
    if not records:
        raise ValueError("No records loaded from input (and qa_chunks fallback if provided)")

    sample = _stratified_sample(records, target_n=args.n, seed=args.seed)
    rows = [_qa_to_row(qa, idx=i + 1) for i, qa in enumerate(sample)]

    master = args.out_dir / "human_eval_master.csv"
    ann1 = args.out_dir / "annotator_1.csv"
    ann2 = args.out_dir / "annotator_2.csv"
    _write_csv(master, rows)
    _write_csv(ann1, rows)
    _write_csv(ann2, rows)

    meta = {
        "input": str(args.input),
        "sample_size": len(rows),
        "seed": args.seed,
        "files": [str(master), str(ann1), str(ann2)],
    }
    with open(args.out_dir / "sample_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote human-eval sheets to: {args.out_dir}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
