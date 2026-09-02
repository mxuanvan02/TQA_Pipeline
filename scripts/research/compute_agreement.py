#!/usr/bin/env python3
"""
Compute pairwise agreement (percent + Cohen's kappa) between two annotators.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        return 0.0

    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return 1.0

    agree = sum(1 for x, y in zip(a, b) if x == y)
    p0 = agree / len(a)

    ca = Counter(a)
    cb = Counter(b)
    pe = 0.0
    n = len(a)
    for l in labels:
        pe += (ca[l] / n) * (cb[l] / n)

    if pe >= 1.0:
        return 1.0
    return (p0 - pe) / (1.0 - pe)


def _index_by_qa(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for r in rows:
        qa_id = str(r.get("qa_id", "")).strip()
        if qa_id:
            out[qa_id] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement.")
    parser.add_argument("--ann1", type=Path, required=True)
    parser.add_argument("--ann2", type=Path, required=True)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=[
            "ann_legal_correctness",
            "ann_groundedness",
            "ann_question_quality",
            "ann_distractor_quality",
            "ann_visual_alignment",
        ],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    r1 = _index_by_qa(_read_csv(args.ann1))
    r2 = _index_by_qa(_read_csv(args.ann2))
    common = sorted(set(r1.keys()) & set(r2.keys()))
    if not common:
        raise ValueError("No shared qa_id rows between annotator files")

    result = {"n_common_items": len(common), "fields": {}}
    for field in args.fields:
        v1, v2 = [], []
        for qa_id in common:
            a = str(r1[qa_id].get(field, "")).strip()
            b = str(r2[qa_id].get(field, "")).strip()
            if a == "" or b == "":
                continue
            v1.append(a)
            v2.append(b)
        if not v1:
            result["fields"][field] = {"n": 0, "percent_agreement": 0.0, "cohen_kappa": 0.0}
            continue
        pa = sum(1 for x, y in zip(v1, v2) if x == y) / len(v1)
        kappa = _cohen_kappa(v1, v2)
        result["fields"][field] = {"n": len(v1), "percent_agreement": pa, "cohen_kappa": kappa}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote agreement report: {args.out}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

