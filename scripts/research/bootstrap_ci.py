#!/usr/bin/env python3
"""
Bootstrap confidence interval for a numeric metric column in CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round((len(sorted_vals) - 1) * p))
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for one numeric metric column.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--column", type=str, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append(float(r[args.column]))
            except Exception:
                continue
    if not rows:
        raise ValueError(f"No numeric values found in column: {args.column}")

    rng = random.Random(args.seed)
    dist = []
    n = len(rows)
    for _ in range(args.n_bootstrap):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        dist.append(_mean(sample))
    dist.sort()

    report = {
        "column": args.column,
        "n": n,
        "mean": _mean(rows),
        "bootstrap_mean": _mean(dist),
        "ci95_low": _percentile(dist, 0.025),
        "ci95_high": _percentile(dist, 0.975),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote bootstrap report: {args.out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

