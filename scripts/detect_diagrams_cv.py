#!/usr/bin/env python3
"""Detect diagram-like pages in rendered scanned textbook images (glass-box, no API).
Signals per page image:
  - long straight lines (Hough)  -> connectors/borders
  - rectangle-like contours      -> boxes/nodes
  - text-area ratio (rough)      -> distinguish figure page vs dense text
A page is 'diagram candidate' if it has enough boxes AND connectors AND is not
pure dense text. Outputs a ranked shortlist per book. This only COUNTS to size
the benchmark; final confirmation will be by expert/VLM on the shortlist.
"""
import sys, os, json, glob
import cv2
import numpy as np

def analyze(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    # binarize
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    ink_ratio = float(np.count_nonzero(th)) / (h*w)
    # long lines
    lines = cv2.HoughLinesP(th, 1, np.pi/180, threshold=120,
                            minLineLength=int(w*0.12), maxLineGap=12)
    n_long = 0; n_h = 0; n_v = 0
    if lines is not None:
        for l in lines[:2000]:
            x1,y1,x2,y2 = l[0]
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            length = np.hypot(x2-x1, y2-y1)
            if length >= w*0.12:
                n_long += 1
                if ang < 12 or ang > 168: n_h += 1
                elif 78 < ang < 102: n_v += 1
    # rectangle-ish contours
    cnts,_ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    n_box = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (h*w)*0.002 or area > (h*w)*0.5:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)
        x,y,bw,bh = cv2.boundingRect(c)
        if len(approx) in (4,5,6) and bw>w*0.04 and bh>h*0.02:
            rect_fill = area/(bw*bh+1e-6)
            if rect_fill > 0.55:
                n_box += 1
    # diagram score: needs boxes + connectors, moderate ink (not full text page)
    flow_like = (n_box >= 3 and n_long >= 4 and 0.01 < ink_ratio < 0.22)
    score = n_box*2 + n_long + (n_h+n_v)
    return {"box": n_box, "long": n_long, "h": n_h, "v": n_v,
            "ink": round(ink_ratio,3), "flow_like": bool(flow_like), "score": int(score)}

def main(img_dir, out_json):
    files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    by_book = {}
    for fp in files:
        base = os.path.basename(fp).split("__p")[0]
        r = analyze(fp)
        if r is None: continue
        r["file"] = os.path.basename(fp)
        by_book.setdefault(base, []).append(r)
    out = {}
    for book, rows in by_book.items():
        cands = [r for r in rows if r["flow_like"]]
        cands.sort(key=lambda x: x["score"], reverse=True)
        out[book] = {
            "n_pages": len(rows),
            "n_diagram_candidates": len(cands),
            "top_pages": [{"file": c["file"], "box": c["box"], "long": c["long"], "score": c["score"]} for c in cands[:40]],
        }
        print(f"{book[:46]:46} pages={len(rows):>4} diagram_cands={len(cands):>4}", flush=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("WROTE", out_json, flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
