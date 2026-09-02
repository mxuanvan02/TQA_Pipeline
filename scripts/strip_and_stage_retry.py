#!/usr/bin/env python3
"""Strip HTTP_ERR/PARSE_ERR rows from VLM jsonl (so --resume re-tries them),
and stage their source JPEGs into _retry_round2/ for re-classification."""
import json, shutil
from pathlib import Path

base = Path(__file__).resolve().parents[1]
data = base / "data"
ERR = {"HTTP_ERR", "PARSE_ERR"}

render_dirs = ["local_cv_candidates", "rendered_cand_textbased", "shortlist_textbased_multimodal",
               "diagram_verify", "scan_tthc", "scan_hienphapvn", "rendered_external",
               "scan_tths", "scan_hanhchinh"]
idx = {}
for d in render_dirs:
    p = data / d
    if p.exists():
        for f in p.glob("*.jpg"):
            idx.setdefault(f.name, f)

# scan ALL vlm_*.jsonl for error rows
retry_dir = data / "_retry_round2"
retry_dir.mkdir(exist_ok=True)

staged, missing = 0, []
for fp in sorted(data.glob("vlm_*.jsonl")):
    rows = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
    errs = [r for r in rows if r.get("label") in ERR]
    if not errs:
        continue
    keep = [r for r in rows if r.get("label") not in ERR]
    for r in errs:
        bn = Path(r["file"]).name
        src = idx.get(bn)
        if src and src.exists():
            dst = retry_dir / bn
            if not dst.exists():
                shutil.copy2(src, dst)
            staged += 1
        else:
            missing.append(bn)
    fp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + ("\n" if keep else ""))
    print(f"{fp.name}: total={len(rows)} keep={len(keep)} stripped_err={len(errs)}")

print(f"\nStaged retry imgs: {staged} | missing src: {len(missing)}")
if missing[:8]:
    print("missing sample:", missing[:8])
print("retry_dir count:", len(list(retry_dir.glob('*.jpg'))))
