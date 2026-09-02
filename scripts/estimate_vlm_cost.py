#!/usr/bin/env python3
"""Estimate VLM classify cost for un-VLM'd internal books BEFORE launching jobs.

For each PDF: decide scan vs text-based (median text length per page), and for
text-based books count fitz CANDIDATE pages (the only pages worth VLM-ing).
Scan books need render-all (diagrams buried in page rasters) -> cost = page_count.
Text books need mine-candidate -> cost = candidate_count (usually tiny).

Runs on conda host env `base` (fitz). Output: JSON + human summary.
"""
from __future__ import annotations
import json, os, statistics, sys
import fitz

# candidate signal identical to mine_render_textbased_candidates.is_candidate
def page_signals(p):
    imgs = p.get_images(full=True)
    big = 0
    for im in imgs:
        try: w, h = im[2], im[3]
        except Exception: w = h = 0
        if w * h >= 350 * 350: big += 1
    try: drawings = p.get_drawings()
    except Exception: drawings = []
    nl = nr = nc = 0
    for d in drawings:
        for item in d.get("items", []):
            t = item[0]
            if t == "l": nl += 1
            elif t == "re": nr += 1
            elif t == "c": nc += 1
    txt = len(p.get_text("text").strip())
    return nr, nl, nc, big, txt

def is_cand(nr, nl, nc, big, txt):
    if nr >= 3 and nl >= 3: return True
    if nc >= 6: return True
    if nr >= 5: return True
    if big >= 1 and txt < 300: return True
    return False

def main():
    books = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    rows = []
    for path in books:
        if not os.path.exists(path):
            rows.append({"book": os.path.basename(path), "error": "missing"}); continue
        try:
            doc = fitz.open(path)
        except Exception as e:
            rows.append({"book": os.path.basename(path), "error": str(e)[:80]}); continue
        n = doc.page_count
        txt_lens = []
        cand = 0
        for i in range(n):
            nr, nl, nc, big, txt = page_signals(doc[i])
            txt_lens.append(txt)
            if is_cand(nr, nl, nc, big, txt): cand += 1
        doc.close()
        med = statistics.median(txt_lens) if txt_lens else 0
        kind = "scan" if med < 80 else "text"
        # cost: scan -> render-all pages; text -> only candidates
        cost = n if kind == "scan" else cand
        rows.append({"book": os.path.basename(path)[:-4], "pages": n,
                     "median_text": int(med), "kind": kind,
                     "fitz_cand": cand, "vlm_cost": cost})
    rows.sort(key=lambda r: -r.get("vlm_cost", 0))
    tot_scan = sum(r.get("vlm_cost",0) for r in rows if r.get("kind")=="scan")
    tot_text = sum(r.get("vlm_cost",0) for r in rows if r.get("kind")=="text")
    out = {"rows": rows,
           "n_books": len(rows),
           "scan_books": sum(1 for r in rows if r.get("kind")=="scan"),
           "text_books": sum(1 for r in rows if r.get("kind")=="text"),
           "vlm_calls_scan_renderall": tot_scan,
           "vlm_calls_text_minecand": tot_text,
           "vlm_calls_total": tot_scan + tot_text}
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"books={out['n_books']} scan={out['scan_books']} text={out['text_books']}")
    print(f"VLM calls: scan(render-all)={tot_scan}  text(mine-cand)={tot_text}  TOTAL={out['vlm_calls_total']}")
    print("--- per book (top cost) ---")
    for r in rows:
        if "error" in r: print(f"  ERR {r['book']}: {r['error']}"); continue
        print(f"  {r['kind']:4} pages={r['pages']:4} cand={r['fitz_cand']:4} cost={r['vlm_cost']:4}  {r['book'][:45]}")

if __name__ == "__main__":
    main()
