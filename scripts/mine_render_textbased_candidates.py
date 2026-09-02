#!/usr/bin/env python3
"""Mine + render CANDIDATE diagram pages from TEXT-BASED law textbooks (cheap, fitz-only).

Rationale (skill multimodal-legal-tqa-pipeline): for text-based PDFs do NOT VLM
every page. Use vector/layout signals to shortlist candidate pages, render ONLY
those, then VLM-classify the rendered candidates. This keeps VLM cost bounded.

Candidate signal (broadened vs probe_diagrams.flow_signal so we also catch
hierarchy/tree/relation diagrams, not just arrow flowcharts):
  - rects >= 3 AND lines >= 3            (boxed/flow structure)
  - OR curves >= 6                        (ar쇼/relation arcs, org trees)
  - OR (rects >= 5)                       (dense boxes = table/diagram, let VLM judge)
  - OR (big standalone image, low text)   (embedded figure on a mostly-text page)
We deliberately OVER-collect candidates; the VLM is the acceptance gate, fitz is
only the cheap filter. We cap candidates/book to keep VLM bounded.

Usage:
  mine_render_textbased_candidates.py <pdf_path> <render_out_dir> [--dpi 150]
        [--max-cand 60] [--cand-json OUT.json]
Outputs:
  - rendered candidate JPEGs into render_out_dir (named <base>__pNNNN.jpg)
  - a candidate manifest JSON (page list + per-page signals)
"""
from __future__ import annotations
import argparse, json, os
import fitz


def page_signals(p):
    rect = p.rect
    imgs = p.get_images(full=True)
    big = 0
    for im in imgs:
        try:
            w, h = im[2], im[3]
        except Exception:
            w = h = 0
        if w * h >= 350 * 350:
            big += 1
    try:
        drawings = p.get_drawings()
    except Exception:
        drawings = []
    nl = nr = nc = 0
    for d in drawings:
        for item in d.get("items", []):
            t = item[0]
            if t == "l":
                nl += 1
            elif t == "re":
                nr += 1
            elif t == "c":
                nc += 1
    txt = len(p.get_text("text").strip())
    return {"rects": nr, "lines": nl, "curves": nc, "big_imgs": big, "txt_len": txt}


def is_candidate(s):
    if s["rects"] >= 3 and s["lines"] >= 3:
        return True, "boxed_flow"
    if s["curves"] >= 6:
        return True, "curves_relation"
    if s["rects"] >= 5:
        return True, "dense_boxes"
    if s["big_imgs"] >= 1 and s["txt_len"] < 300:
        return True, "embedded_figure"
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("render_out_dir")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-cand", type=int, default=60)
    ap.add_argument("--cand-json", default="")
    a = ap.parse_args()

    doc = fitz.open(a.pdf_path)
    base = os.path.splitext(os.path.basename(a.pdf_path))[0]
    os.makedirs(a.render_out_dir, exist_ok=True)
    mat = fitz.Matrix(a.dpi / 72, a.dpi / 72)

    cands = []
    for i in range(doc.page_count):
        s = page_signals(doc[i])
        ok, why = is_candidate(s)
        if ok:
            cands.append({"page": i + 1, "why": why, **s})

    # rank: prefer boxed_flow / curves_relation, then by structural richness
    prio = {"boxed_flow": 0, "curves_relation": 1, "dense_boxes": 2, "embedded_figure": 3}
    cands.sort(key=lambda c: (prio.get(c["why"], 9),
                              -(c["rects"] + c["lines"] + c["curves"])))
    cands = cands[: a.max_cand]

    rendered = []
    for c in cands:
        p = doc[c["page"] - 1]
        pix = p.get_pixmap(matrix=mat)
        fn = f"{base}__p{c['page']:04d}.jpg"
        pix.save(os.path.join(a.render_out_dir, fn))
        c["file"] = fn
        rendered.append(c)
    doc.close()

    manifest = {"pdf": a.pdf_path, "base": base, "n_candidates": len(rendered),
                "dpi": a.dpi, "candidates": rendered}
    out_json = a.cand_json or os.path.join(a.render_out_dir, f"_cand_{base}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"CAND {base}: {len(rendered)} candidate pages rendered -> {a.render_out_dir}", flush=True)
    print(f"  manifest: {out_json}", flush=True)


if __name__ == "__main__":
    main()
