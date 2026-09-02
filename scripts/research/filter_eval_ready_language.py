#!/usr/bin/env python3
"""
Filter multilingual outliers from the eval-ready release.

The QAG stage asks the base LLM to answer in Vietnamese, but a small subset of
records still contains Chinese text or obvious English scaffolding artifacts
such as "full correct answer text". For public benchmark release we remove
those rows conservatively from the eval-ready subset.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EN_MARKER_RE = re.compile(
    r"\b(full correct answer text|correct answer is|because|therefore|In this case)\b",
    re.IGNORECASE,
)


def _is_bad_record(row: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    fields = ["question_content", "ground_truth", "legal_rationale", "candidate_answers"]
    texts: list[str] = []
    for field in fields:
        value = row.get(field, "")
        if isinstance(value, list):
            texts.extend(str(x) for x in value)
        else:
            texts.append(str(value))
    merged = " ".join(texts)
    if CJK_RE.search(merged):
        reasons.append("cjk_chars")
    if EN_MARKER_RE.search(merged):
        reasons.append("english_scaffold")
    return (len(reasons) > 0), reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter multilingual outliers from eval-ready JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset_eval_ready.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output cleaned dataset_eval_ready.jsonl")
    parser.add_argument("--split-dir", type=Path, required=True, help="Output directory for train/dev/test JSONL")
    parser.add_argument("--report-out", type=Path, required=True, help="Output JSON report")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    split_handles = {}
    split_kept = Counter()
    split_removed = Counter()
    reason_counts = Counter()
    removed_examples: list[dict] = []
    total = 0
    kept = 0

    with args.input.open("r", encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as clean_out:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            split = str(row.get("split", "unknown"))
            bad, reasons = _is_bad_record(row)
            if bad:
                split_removed[split] += 1
                reason_counts.update(reasons)
                if len(removed_examples) < 25:
                    removed_examples.append(
                        {
                            "qa_id": row.get("qa_id"),
                            "split": split,
                            "reasons": reasons,
                        }
                    )
                continue

            kept += 1
            split_kept[split] += 1
            serialized = json.dumps(row, ensure_ascii=False) + "\n"
            clean_out.write(serialized)

            if split not in split_handles:
                split_handles[split] = (args.split_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            split_handles[split].write(serialized)

    for handle in split_handles.values():
        handle.close()

    report = {
        "input_records": total,
        "kept_records": kept,
        "removed_records": total - kept,
        "reason_counts": dict(reason_counts),
        "split_kept": dict(split_kept),
        "split_removed": dict(split_removed),
        "removed_examples": removed_examples,
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
