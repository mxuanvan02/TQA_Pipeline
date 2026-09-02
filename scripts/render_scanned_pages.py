#!/usr/bin/env python3
"""Render pages of SCANNED law textbooks to images for diagram detection.
For scanned PDFs (text~0) the only way to find diagrams is to look at pixels.
This renders pages at moderate DPI and saves JPEGs + an index.
To keep it bounded we sample: render EVERY page but downscale; caller can
later run a VLM/layout detector on the saved images.
"""
import sys, os, json, fitz

def render_book(pdf_path, out_dir, dpi=110, max_pages=None):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    n = doc.page_count
    if max_pages:
        n = min(n, max_pages)
    mat = fitz.Matrix(dpi/72, dpi/72)
    idx = []
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for i in range(n):
        p = doc[i]
        pix = p.get_pixmap(matrix=mat)
        fn = f"{base}__p{i+1:04d}.jpg"
        fp = os.path.join(out_dir, fn)
        pix.save(fp)
        idx.append({"page": i+1, "file": fn, "w": pix.width, "h": pix.height})
    doc.close()
    with open(os.path.join(out_dir, f"_index_{base}.json"), "w", encoding="utf-8") as f:
        json.dump({"pdf": pdf_path, "n_rendered": len(idx), "dpi": dpi, "pages": idx}, f, ensure_ascii=False)
    print(f"RENDERED {base}: {len(idx)} pages -> {out_dir}", flush=True)
    return len(idx)

if __name__ == "__main__":
    # args: out_dir dpi pdf1 [pdf2 ...]
    # Use @filelist.txt for paths with spaces/newlines.
    out_dir = sys.argv[1]
    dpi = int(sys.argv[2])
    pdfs = []
    for arg in sys.argv[3:]:
        if arg.startswith("@"):
            with open(arg[1:], encoding="utf-8") as f:
                pdfs.extend(line.strip() for line in f if line.strip())
        else:
            pdfs.append(arg)
    total = 0
    for pdf in pdfs:
        total += render_book(pdf, out_dir, dpi=dpi)
    print(f"TOTAL pages rendered: {total}", flush=True)
