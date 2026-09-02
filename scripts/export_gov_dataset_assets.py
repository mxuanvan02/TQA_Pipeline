#!/usr/bin/env python3
"""Export reviewed government/legal assets into the dataset source vault."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

REVIEWED = Path("data/processed/gov_assets_reviewed_shortlist.csv")
RAW_ROOT = Path("data/raw_gov")
DATASET_ROOT = Path("data/gov_legal_dataset")
PDF_DIR = DATASET_ROOT / "pdf"
IMG_DIR = DATASET_ROOT / "images"
OUT_MANIFEST = DATASET_ROOT / "manifest.csv"


def main() -> None:
    rows = list(csv.DictReader(REVIEWED.open(encoding="utf-8")))
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    exported = []
    for row in rows:
        if row.get("review_decision") != "KEEP":
            continue
        src = RAW_ROOT / row["filename"]
        if not src.exists():
            continue
        kind_dir = PDF_DIR if src.suffix.lower() == ".pdf" else IMG_DIR
        dst = kind_dir / src.name
        shutil.copy2(src, dst)
        out = dict(row)
        out["dataset_path"] = str(dst)
        out["dataset_use"] = "source_for_render_and_qa"
        exported.append(out)

    fields = list(exported[0].keys()) if exported else ["dataset_path"]
    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exported)

    print(f"exported={len(exported)} root={DATASET_ROOT} manifest={OUT_MANIFEST}")


if __name__ == "__main__":
    main()
