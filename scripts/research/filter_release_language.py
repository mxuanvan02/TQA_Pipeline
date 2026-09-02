#!/usr/bin/env python3
"""
Sanitize and filter release JSONL files for conservative Vietnamese-only export.

This script removes harmless English scaffolding such as "(full correct answer
text)" and drops rows that still contain foreign-language or prompt-leak
artifacts after sanitation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.language_sanity import qa_artifact_reasons, sanitize_generated_text


TEXT_FIELDS = ("question_content", "ground_truth", "legal_rationale")
LIST_FIELDS = ("candidate_answers", "candidate_answers_raw")


def _sanitize_row(row: dict) -> tuple[dict, bool]:
    updated = dict(row)
    changed = False

    for field in TEXT_FIELDS:
        if field in updated:
            cleaned = sanitize_generated_text(updated.get(field, ""))
            changed = changed or cleaned != updated.get(field, "")
            updated[field] = cleaned

    for field in LIST_FIELDS:
        value = updated.get(field)
        if isinstance(value, list):
            cleaned_list = [sanitize_generated_text(x) for x in value]
            changed = changed or cleaned_list != value
            updated[field] = cleaned_list

    if "ground_truth_raw" in updated:
        cleaned = sanitize_generated_text(updated.get("ground_truth_raw", ""))
        changed = changed or cleaned != updated.get("ground_truth_raw", "")
        updated["ground_truth_raw"] = cleaned

    return updated, changed


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize and filter release JSONL files.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--out", type=Path, required=True, help="Output cleaned JSONL file")
    parser.add_argument("--report-out", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--split-dir", type=Path, default=None, help="Optional output dir for split JSONL files")
    args = parser.parse_args()

    rows_out: list[dict] = []
    split_rows: dict[str, list[dict]] = {}
    reason_counts = Counter()
    split_kept = Counter()
    split_removed = Counter()
    removed_examples: list[dict] = []
    total = 0
    sanitized = 0

    with args.input.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            cleaned_row, changed = _sanitize_row(row)
            if changed:
                sanitized += 1

            reasons = qa_artifact_reasons(
                cleaned_row.get("question_content", ""),
                cleaned_row.get("candidate_answers", []),
                cleaned_row.get("ground_truth", ""),
                cleaned_row.get("legal_rationale", ""),
            )
            split = str(cleaned_row.get("split", "unknown")).strip() or "unknown"
            if reasons:
                reason_counts.update(reasons)
                split_removed[split] += 1
                if len(removed_examples) < 25:
                    removed_examples.append(
                        {
                            "qa_id": cleaned_row.get("qa_id"),
                            "split": split,
                            "reasons": reasons,
                        }
                    )
                continue

            rows_out.append(cleaned_row)
            split_kept[split] += 1
            if args.split_dir is not None:
                split_rows.setdefault(split, []).append(cleaned_row)

    _write_jsonl(args.out, rows_out)
    if args.split_dir is not None:
        for split, split_rows_list in split_rows.items():
            _write_jsonl(args.split_dir / f"{split}.jsonl", split_rows_list)

    report = {
        "input_records": total,
        "kept_records": len(rows_out),
        "removed_records": total - len(rows_out),
        "sanitized_records": sanitized,
        "reason_counts": dict(reason_counts),
        "split_kept": dict(split_kept),
        "split_removed": dict(split_removed),
        "removed_examples": removed_examples,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
