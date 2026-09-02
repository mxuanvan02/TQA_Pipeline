#!/usr/bin/env python3
"""Probe diagram density in a legal textbook PDF.
Detects BOTH:
  (1) raster diagrams: large embedded images (likely scanned figure) vs small (logo)
  (2) vector diagrams: pages with many drawing primitives (lines/rects/curves) = flowchart-like
Outputs per-page signals + a shortlist of candidate diagram pages.
No gold labels are created. This only counts to size the benchmark.
"""
import sys, json, fitz
from collections import defaultdict

def probe(pdf_path, out_json):
    doc = fitz.open(pdf_path)
    n = doc.page_count
    pages = []
    for i in range(n):
        p = doc[i]
        rect = p.rect
        page_area = max(rect.width * rect.height, 1)
        # raster images
        imgs = p.get_images(full=True)
        big_imgs = 0
        full_scan = False
        for im in imgs:
            xref = im[0]
            try:
                w = im[2]; h = im[3]
            except Exception:
                w = h = 0
            # heuristic: image covering >50% page side = likely full scanned page
            # large standalone figure if reasonably big but not full page
            if w*h >= 400*400:
                big_imgs += 1
        # vector drawings
        try:
            drawings = p.get_drawings()
        except Exception:
            drawings = []
        n_lines = n_rects = n_curves = 0
        for d in drawings:
            for item in d.get("items", []):
                t = item[0]
                if t == "l": n_lines += 1
                elif t == "re": n_rects += 1
                elif t == "c": n_curves += 1
        vec_prims = n_lines + n_rects + n_curves
        txt_len = len(p.get_text("text"))
        # flowchart signal: many rects + connecting lines on a page
        flow_signal = (n_rects >= 4 and n_lines >= 4)
        pages.append({
            "page": i+1,
            "txt_len": txt_len,
            "n_embedded_imgs": len(imgs),
            "big_imgs": big_imgs,
            "rects": n_rects, "lines": n_lines, "curves": n_curves,
            "vec_prims": vec_prims,
            "flow_signal": flow_signal,
        })
    doc.close()
    # candidate pages = flow_signal OR (has big image AND low text => figure page)
    cands_vec = [p["page"] for p in pages if p["flow_signal"]]
    cands_ras = [p["page"] for p in pages if p["big_imgs"]>=1 and p["txt_len"]<300]
    summary = {
        "pdf": pdf_path,
        "n_pages": n,
        "pages_with_big_img": sum(1 for p in pages if p["big_imgs"]>=1),
        "pages_flow_signal": len(cands_vec),
        "pages_raster_figure": len(cands_ras),
        "candidate_vector_pages": cands_vec[:80],
        "candidate_raster_pages": cands_ras[:80],
        "total_vec_prims": sum(p["vec_prims"] for p in pages),
        "median_txt_len": sorted(p["txt_len"] for p in pages)[n//2] if n else 0,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pages": pages}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    probe(sys.argv[1], sys.argv[2])
