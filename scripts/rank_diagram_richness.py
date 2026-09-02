#!/usr/bin/env python3
"""Rank all law textbook PDFs by diagram richness (multimodal value).
Fast pass (no page rendering):
  - vector PDFs: count flow-like pages (rects+lines clusters) + total vector prims
  - scanned PDFs (median text ~0): flag as 'scanned -> needs render+VLM'
Output: ranked table to stdout + JSON.
"""
import sys, os, json, glob, fitz

def probe(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"pdf": os.path.basename(pdf_path), "error": str(e)}
    n = doc.page_count
    flow_pages = 0
    fig_pages = 0          # vector figure: many prims but some text
    txt_lens = []
    total_prims = 0
    big_img_pages = 0
    for i in range(n):
        p = doc[i]
        txt = len(p.get_text("text"))
        txt_lens.append(txt)
        try:
            drawings = p.get_drawings()
        except Exception:
            drawings = []
        nl = nr = nc = 0
        for d in drawings:
            for item in d.get("items", []):
                t = item[0]
                if t == "l": nl += 1
                elif t == "re": nr += 1
                elif t == "c": nc += 1
        prims = nl + nr + nc
        total_prims += prims
        if nr >= 4 and nl >= 4:
            flow_pages += 1
        if prims >= 30:
            fig_pages += 1
        imgs = p.get_images(full=True)
        for im in imgs:
            try:
                if im[2]*im[3] >= 400*400:
                    big_img_pages += 1; break
            except Exception:
                pass
    doc.close()
    txt_lens.sort()
    median_txt = txt_lens[n//2] if n else 0
    scanned = median_txt < 50
    return {
        "pdf": os.path.basename(pdf_path),
        "pages": n,
        "median_txt": median_txt,
        "scanned": scanned,
        "flow_pages": flow_pages,
        "vec_fig_pages": fig_pages,
        "total_vec_prims": total_prims,
        "big_img_pages": big_img_pages,
    }

def main(raw_dir, out_json):
    pdfs = sorted(glob.glob(os.path.join(raw_dir, "*.pdf")))
    rows = []
    for pf in pdfs:
        r = probe(pf)
        rows.append(r)
        # progress line
        if "error" in r:
            print(f"ERR  {r['pdf'][:42]:42} {r['error'][:30]}", flush=True)
        else:
            tag = "SCAN" if r["scanned"] else "VEC "
            print(f"{tag} {r['pdf'][:42]:42} pg={r['pages']:>4} flow={r['flow_pages']:>3} fig={r['vec_fig_pages']:>4} prims={r['total_vec_prims']:>7}", flush=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    # ranking summary
    vec = [r for r in rows if not r.get("scanned") and "error" not in r]
    vec.sort(key=lambda x: (x["flow_pages"], x["vec_fig_pages"]), reverse=True)
    print("\n=== TOP vector/diagram-rich (text-based) ===", flush=True)
    for r in vec[:12]:
        print(f"  {r['pdf'][:46]:46} flow={r['flow_pages']:>3} fig={r['vec_fig_pages']:>4} prims={r['total_vec_prims']:>7}", flush=True)
    scan = [r for r in rows if r.get("scanned")]
    print(f"\nScanned (need render+VLM): {len(scan)} books", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
