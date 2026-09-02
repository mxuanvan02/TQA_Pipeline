#!/usr/bin/env bash
# Build filelist of the 28 un-VLM'd internal books (full paths under data/raw).
set -u
cd "$(dirname "$0")/.." || exit 1
RAW=data/raw
OUT=data/_todo_books.txt
: > "$OUT"
# match by listing data/raw and excluding prefixes already in any vlm_*.jsonl
python3 - "$RAW" "$OUT" <<'PY'
import os, sys, json, glob
raw, out = sys.argv[1], sys.argv[2]
# classified prefixes
done = set()
for jf in glob.glob("data/vlm_*.jsonl"):
    for line in open(jf, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try: f = json.loads(line).get("file","")
        except Exception: continue
        if "__" in f: done.add(f.split("__")[0].strip())
# also external-classified handled separately; here only data/raw books
todo = []
for fn in sorted(os.listdir(raw)):
    if not fn.lower().endswith(".pdf"): continue
    base = fn[:-4]
    # a book is "done" if its base (normalized) appears as a classified prefix
    norm = base.strip()
    hit = any(norm[:18].lower() in d.lower() or d.lower() in norm.lower() for d in done if d)
    if not hit:
        todo.append(os.path.join(raw, fn))
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(todo) + "\n")
print(f"todo PDFs written: {len(todo)} -> {out}")
for t in todo: print("  ", os.path.basename(t))
PY
