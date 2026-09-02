#!/usr/bin/env python3
"""Create a cleaner shortlist from raw government/legal crawl manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DROP_TOKENS = {
    "logo", "favicon", "spcommon", "background", "bg-", "bg_", "banner", "bn_",
    "rss", "icon", "login", "footer", "header", "weather", "football", "mask",
    "backgroun", "attach-icon", "gg.png", "ftscore", "camera-icon",
}
KEEP_TOKENS = {
    "nghi-quyet", "nghi_dinh", "nghi-dinh", "thong-tu", "cong-van", "du-thao",
    "luat", "phap-luat", "tro-giup-phap-ly", "tuyen-truyen", "pbgdpl", "huong-dan",
    "infographic", "info-graphic", "quy-trinh", "thu-tuc", "hoi-dap", "dap-an",
}


def classify(row: dict[str, str]) -> tuple[str, str]:
    url = row.get("url", "").lower()
    name = row.get("filename", "").lower()
    ctype = row.get("content_type", "").lower()
    hay = f"{url} {name}"

    if "pdf" in ctype or url.split("?", 1)[0].endswith(".pdf"):
        if any(tok in hay for tok in KEEP_TOKENS):
            return "keep_pdf", "legal_policy_pdf_keyword"
        return "needs_review_pdf", "pdf_without_strong_keyword"

    if any(tok in hay for tok in DROP_TOKENS):
        return "drop_static", "static_logo_banner_icon_keyword"

    if any(tok in hay for tok in KEEP_TOKENS):
        return "keep_image_candidate", "legal_education_image_keyword"

    if "/anhdaidien/" in hay or "publishingimages/tintuc" in hay:
        return "needs_vision", "news_thumbnail_or_article_image"

    if "image" in ctype:
        return "needs_vision", "generic_image"

    return "drop_unknown", "unrecognized_asset"


def iter_manifests(root: Path):
    yield from root.glob("*/metadata/assets_manifest.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/raw_gov")
    ap.add_argument("--out", default="data/processed/gov_assets_clean_manifest.csv")
    args = ap.parse_args()

    rows = []
    for manifest in iter_manifests(Path(args.raw_root)):
        with manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label, reason = classify(row)
                row = dict(row)
                row["source_manifest"] = str(manifest)
                row["filter_label"] = label
                row["filter_reason"] = reason
                rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filter_label", "filter_reason", "filename", "asset_type", "url", "source_domain",
        "content_type", "bytes", "sha256", "retrieved_at", "notes", "source_manifest",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["filter_label"]] = counts.get(row["filter_label"], 0) + 1
    print(f"wrote {out} rows={len(rows)}")
    for label, count in sorted(counts.items()):
        print(f"{label}\t{count}")


if __name__ == "__main__":
    main()
