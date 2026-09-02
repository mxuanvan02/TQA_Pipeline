#!/usr/bin/env python3
"""Render fitz CANDIDATE pages for the un-VLM'd TEXT books (runs on conda host).

Reads the cost-estimate JSON, selects kind=="text" books with fitz_cand>0,
maps each back to its source PDF under data/raw, and renders ONLY candidate
pages via the proven mine_render_textbased_candidates logic. No VLM here.

Output dir: data/rendered_internal_textcand/  (named <base>__pNNNN.jpg)
"""
from __future__ import annotations
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EST = os.path.join(ROOT, "data", "_vlm_cost_estimate.json")
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "rendered_internal_textcand")
MINE = os.path.join(ROOT, "scripts", "mine_render_textbased_candidates.py")

def find_pdf(base):
    # base is filename without .pdf; some have trailing ".doc" etc. Try exact then fuzzy.
    cand = os.path.join(RAW, base + ".pdf")
    if os.path.exists(cand):
        return cand
    for fn in os.listdir(RAW):
        if fn.lower().endswith(".pdf") and fn[:-4] == base:
            return os.path.join(RAW, fn)
    # fuzzy: startswith first 20 chars
    key = base[:20]
    for fn in os.listdir(RAW):
        if fn.lower().endswith(".pdf") and fn.startswith(key):
            return os.path.join(RAW, fn)
    return None

def main():
    est = json.load(open(EST, encoding="utf-8"))
    books = [r for r in est["rows"]
             if r.get("kind") == "text" and r.get("fitz_cand", 0) > 0]
    books.sort(key=lambda r: -r["fitz_cand"])
    os.makedirs(OUT, exist_ok=True)
    print(f"text books w/ candidates: {len(books)}", flush=True)
    total = 0
    for r in books:
        pdf = find_pdf(r["book"])
        if not pdf:
            print(f"  MISS pdf for: {r['book']}", flush=True)
            continue
        cmd = [sys.executable, MINE, pdf, OUT,
               "--dpi", "150", "--max-cand", str(r["fitz_cand"] + 5)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        line = (res.stdout or "").strip().splitlines()
        tag = line[0] if line else (res.stderr or "ERR")[:80]
        print(f"  {tag}", flush=True)
        total += r["fitz_cand"]
    n = len([f for f in os.listdir(OUT) if f.lower().endswith(".jpg")])
    print(f"DONE render: ~{total} expected, {n} jpg on disk -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
