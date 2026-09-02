#!/usr/bin/env bash
# Round-3 VLM classify: 4 sách text-based đã render nhưng chưa quét.
# Append vào data/vlm_textbased_round3.jsonl, resume-safe.
set -u
cd "$(dirname "$0")/.." || exit 1

IMG_DIR="data/rendered_cand_textbased"
OUT="data/vlm_textbased_round3.jsonl"
MODEL="cx/gpt-5.4-mini"

BOOKS=(
  "36. LUAT KINH TE QUOC TE"
  "43. THUC HANH NGHE NGHIEP"
  "48. TBG. PHAP LUAT AN SINH XA HOI"
  "LUAT CHUNG KHOAN"
)

echo "=== ROUND3 VLM START $(date '+%F %T') model=$MODEL ==="
for b in "${BOOKS[@]}"; do
  echo ">>> BOOK: $b"
  python3 scripts/classify_pages_vlm.py "$IMG_DIR" "$OUT" \
    --glob "${b}__p*.jpg" --model "$MODEL" --resume
done
echo "=== ROUND3 VLM DONE $(date '+%F %T') ==="
echo "--- verdict distribution round3 ---"
python3 - "$OUT" <<'PY'
import json,sys,collections
c=collections.Counter()
books=collections.Counter()
for line in open(sys.argv[1],encoding="utf-8"):
    try:
        o=json.loads(line)
    except: continue
    c[o.get("label","?")]+=1
    f=o.get("file","")
    books[f.split("__")[0]]+=1
print("verdict:", dict(c))
print("per-book:", dict(books))
PY
