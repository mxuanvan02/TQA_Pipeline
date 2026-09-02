#!/usr/bin/env python3
"""Build a conservative shortlist of crawled government/legal assets."""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

BAD_IMAGE_TOKENS = {
    "logo", "favicon", "spcommon", "bg-", "bg_", "background", "banner", "header",
    "footer", "rss", "icon", "login", "weather", "football", "mask", "feedback",
}
GOOD_TOKENS = {
    "nghi-quyet", "nghi-dinh", "thong-tu", "cong-van", "du-thao", "luat",
    "phap-luat", "tro-giup-phap-ly", "thu-tuc", "quy-trinh", "huong-dan",
    "infographic", "pbgdpl", "tuyen-truyen",
}


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        data = path.read_bytes()[:65536]
    except OSError:
        return None, None
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in {0xD8, 0xD9}:
                continue
            if i + 2 > len(data):
                break
            seglen = struct.unpack(">H", data[i:i + 2])[0]
            if marker in range(0xC0, 0xC4) or marker in range(0xC5, 0xC8) or marker in range(0xC9, 0xCC) or marker in range(0xCD, 0xD0):
                if i + 7 <= len(data):
                    h, w = struct.unpack(">HH", data[i + 3:i + 7])
                    return w, h
            i += seglen
    return None, None


def local_path(filename: str) -> Path:
    return Path("data/raw_gov") / filename


def decide(row: dict[str, str]) -> tuple[str, str, int | None, int | None]:
    label = row.get("filter_label", "")
    filename = row.get("filename", "")
    hay = f"{filename} {row.get('url', '')}".lower()
    path = local_path(filename)

    if label == "needs_review_pdf" or filename.lower().endswith(".pdf"):
        return "keep_pdf_for_render", "official_pdf_downloaded_from_policy_article", None, None

    w, h = image_size(path)
    if any(tok in hay for tok in BAD_IMAGE_TOKENS):
        return "drop_static", "static_or_site_chrome_token", w, h
    if w is not None and h is not None:
        if w < 350 or h < 250:
            return "drop_too_small", "small_image_unlikely_to_support_vqa", w, h
        if w / max(h, 1) > 5 or h / max(w, 1) > 5:
            return "drop_banner_shape", "extreme_aspect_ratio", w, h
    if any(tok in hay for tok in GOOD_TOKENS):
        return "candidate_image_for_vision", "legal_keyword_in_url_or_name", w, h
    if label == "needs_vision":
        return "candidate_image_for_vision", "large_or_article_image_needs_visual_check", w, h
    return "drop_low_signal", "no_legal_signal", w, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/gov_assets_clean_manifest.csv")
    ap.add_argument("--out", default="data/processed/gov_assets_shortlist.csv")
    args = ap.parse_args()

    inp = Path(args.input)
    rows = list(csv.DictReader(inp.open(encoding="utf-8")))
    out_rows = []
    for row in rows:
        decision, reason, w, h = decide(row)
        row = dict(row)
        row["shortlist_label"] = decision
        row["shortlist_reason"] = reason
        row["width"] = "" if w is None else str(w)
        row["height"] = "" if h is None else str(h)
        out_rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else []
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    counts: dict[str, int] = {}
    for row in out_rows:
        counts[row["shortlist_label"]] = counts.get(row["shortlist_label"], 0) + 1
    print(f"wrote {out} rows={len(out_rows)}")
    for label, count in sorted(counts.items()):
        print(f"{label}\t{count}")


if __name__ == "__main__":
    main()
