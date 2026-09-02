#!/usr/bin/env bash
# Retry 110 trang lỗi (đã strip HTTP_ERR/PARSE_ERR) từ _retry_round2.
# Chạy SAU khi external xong để tránh 429 do double-rate. --resume idempotent.
set -u
cd "$(dirname "$0")/.." || exit 1
M=cx/gpt-5.4-mini
echo "=== [$(date +%H:%M:%S)] RETRY 110 error pages (_retry_round2) ==="
python3 scripts/classify_pages_vlm.py data/_retry_round2 data/vlm_retry_round2.jsonl --model "$M" --resume 2>&1
echo "=== [$(date +%H:%M:%S)] RETRY done. rows: $(wc -l < data/vlm_retry_round2.jsonl 2>/dev/null) ==="
python3 -c "
import json,collections
c=collections.Counter()
for l in open('data/vlm_retry_round2.jsonl'):
    try:c[json.loads(l)['label']]+=1
    except:pass
print('LABELS:',dict(c))
"
