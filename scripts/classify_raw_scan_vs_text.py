import fitz, os, glob

RAW = "data/raw"
SCANNED = {"28. LUAT TO TUNG HINH SU VIET NAM.pdf", "21. LUAT HANH CHINH VIET NAM.pdf",
           "17,18. LUAT HIEN PHAP VIET NAM.pdf", "22. LUAT TO TUNG HANH CHINH VIET NAM.pdf"}
TEXTBASED_PARTIAL = {"19.LUAT HIEN PHAP NUOC NGOAI.pdf", "47. LY LUAN DINH TOI DANH.pdf",
                     "39. PHAP LUAT SO HUU TRI TUE.pdf", "49. TBG HOAT DONG CONG CHUNG, CHUNG THUC.pdf"}
done = SCANNED | TEXTBASED_PARTIAL

rows = []
files = sorted(glob.glob(os.path.join(RAW, "*.pdf")) + glob.glob(os.path.join(RAW, "*.PDF")))
for p in files:
    fn = os.path.basename(p)
    try:
        d = fitz.open(p); n = d.page_count
        idxs = sorted(set([int(i * (n - 1) / 11) for i in range(12)])) if n > 1 else [0]
        tot_txt = 0
        for i in idxs:
            tot_txt += len(d[i].get_text("text").strip())
        avg_txt = tot_txt / len(idxs)
        kind = "scan" if avg_txt < 60 else "text"
        rows.append((fn, n, round(avg_txt), kind))
        d.close()
    except Exception as e:
        rows.append((fn, -1, -1, "ERR:" + str(e)[:30]))

scan = [r for r in rows if r[3] == "scan"]
text = [r for r in rows if r[3] == "text"]
print(f"TONG: {len(rows)} | scan~{len(scan)} | text~{len(text)}")
print("\n=== SCAN (can render full + VLM) ===")
for fn, n, t, k in scan:
    mark = "[FULL-DONE]" if fn in SCANNED else ("[partial]" if fn in TEXTBASED_PARTIAL else "[CHUA]")
    print(f"  {mark:11} {n:>4}tr  {fn[:55]}")
print("\n=== TEXT-BASED (mining vector + VLM trang ung vien) ===")
for fn, n, t, k in text:
    mark = "[FULL-DONE]" if fn in SCANNED else ("[partial]" if fn in TEXTBASED_PARTIAL else "[CHUA]")
    print(f"  {mark:11} {n:>4}tr  {fn[:55]}")
scan_todo = [r for r in scan if r[0] not in done]
text_todo = [r for r in text if r[0] not in done]
print(f"\n>>> SCAN chua quet: {len(scan_todo)} sach, ~{sum(n for fn,n,t,k in scan_todo if n>0)} trang (VLM full)")
print(f">>> TEXT chua quet: {len(text_todo)} sach, ~{sum(n for fn,n,t,k in text_todo if n>0)} trang (chi VLM trang ung vien)")
