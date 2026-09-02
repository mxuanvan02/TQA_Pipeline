#!/usr/bin/env python3
"""Copy CV-ranked diagram candidates into a VLM input directory."""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cv_json")
    ap.add_argument("rendered_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--top-per-book", type=int, default=40)
    args = ap.parse_args()

    rendered = Path(args.rendered_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.cv_json).read_text(encoding="utf-8"))

    copied = 0
    for book, info in sorted(data.items()):
        for row in info.get("top_pages", [])[: args.top_per_book]:
            fn = row.get("file")
            if not fn:
                continue
            src = rendered / fn
            if src.exists():
                shutil.copy2(src, out / fn)
                copied += 1
    print(f"COPIED {copied} candidates -> {out}")


if __name__ == "__main__":
    main()
