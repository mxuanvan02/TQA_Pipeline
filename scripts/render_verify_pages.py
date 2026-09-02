#!/usr/bin/env python3
"""For selected text-based books, find pages with many vector primitives
(candidate diagrams) and render ONLY those pages to images for eyeballing.
This lets us distinguish real process-diagrams from forms/tables/charts.
"""
import sys, os, json, fitz

BOOKS = [
    "data/raw/49. TBG HOAT DONG CONG CHUNG, CHUNG THUC.pdf",
    "data/raw/47. LY LUAN DINH TOI DANH.pdf",
    "data/raw/39. PHAP LUAT SO HUU TRI TUE.pdf",
    "data/raw/19.LUAT HIEN PHAP NUOC NGOAI.pdf",
]
OUT = "data/diagram_verify"

def vec_pages(doc, topk=8):
    scored = []
    for i in range(doc.page_count):
        p = doc[i]
        nl=nr=nc=0
        for d in p.get_drawings():
            for it in d.get("items", []):
                t=it[0]
                if t=="l": nl+=1
                elif t=="re": nr+=1
                elif t=="c": nc+=1
        prims = nl+nr+nc
        # flow-like prefers rects+lines together
        flow = (nr>=4 and nl>=4)
        scored.append((i, prims, nr, nl, flow))
    scored.sort(key=lambda x:(x[4], x[1]), reverse=True)
    return scored[:topk]

def main():
    os.makedirs(OUT, exist_ok=True)
    mat = fitz.Matrix(130/72, 130/72)
    summary={}
    for bp in BOOKS:
        if not os.path.exists(bp):
            print("MISSING", bp, flush=True); continue
        doc=fitz.open(bp)
        base=os.path.splitext(os.path.basename(bp))[0][:22].replace(" ","_")
        tops=vec_pages(doc)
        pages=[]
        for (i,prims,nr,nl,flow) in tops:
            pix=doc[i].get_pixmap(matrix=mat)
            fn=f"{base}__p{i+1:04d}_re{nr}_li{nl}.jpg"
            pix.save(os.path.join(OUT,fn))
            pages.append({"page":i+1,"prims":prims,"rect":nr,"line":nl,"file":fn})
        summary[base]=pages
        doc.close()
        print(f"{base}: rendered {len(pages)} candidate pages", flush=True)
    json.dump(summary, open(os.path.join(OUT,"_verify_index.json"),"w"), ensure_ascii=False, indent=2)
    print("DONE", flush=True)

if __name__=="__main__":
    main()
