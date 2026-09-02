#!/usr/bin/env python3
"""Apply spot-check decisions and emit a reviewed shortlist for TQA source expansion."""

from __future__ import annotations

import csv
from pathlib import Path

INPUT = Path("data/processed/gov_assets_shortlist.csv")
SPOTCHECK = Path("data/processed/gov_assets_vision_spotcheck.csv")
OUT = Path("data/processed/gov_assets_reviewed_shortlist.csv")


def main() -> None:
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8")))
    manual = {}
    if SPOTCHECK.exists():
        for row in csv.DictReader(SPOTCHECK.open(encoding="utf-8")):
            manual[row["asset_path"]] = row

    out_rows = []
    for row in rows:
        path = row["filename"]
        decision = "REVIEW"
        reason = row.get("shortlist_reason", "")
        if row["shortlist_label"] == "keep_pdf_for_render":
            decision = "KEEP"
            reason = "official_pdf_for_render_and_text_extraction"
        elif row["shortlist_label"].startswith("drop"):
            decision = "DROP"
        elif path in manual:
            decision = manual[path]["decision"]
            reason = manual[path]["reason"]
        row = dict(row)
        row["review_decision"] = decision
        row["review_reason"] = reason
        out_rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else []
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    counts = {}
    for row in out_rows:
        counts[row["review_decision"]] = counts.get(row["review_decision"], 0) + 1
    print(f"wrote {OUT} rows={len(out_rows)}")
    for k in sorted(counts):
        print(f"{k}\t{counts[k]}")


if __name__ == "__main__":
    main()
