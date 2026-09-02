#!/usr/bin/env python3
"""Blinded LLM-as-judge for ECM-TQAG work24 generation cells.

Design guards:
  * The judge never sees method, condition, generator model, or run id.
  * Choice order is deterministically permuted per item (seeded), so the judge
    cannot exploit position priors; the gold index is remapped accordingly.
  * Rubric scores are integers 1-5 on four independent dimensions.
  * Dry-run by default; --execute is required before any HTTP request.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 20260806

RUBRIC = """Bạn là giám khảo độc lập chấm chất lượng MỘT câu hỏi trắc nghiệm pháp luật tiếng Việt.
Bạn chỉ được dùng ĐOẠN BẰNG CHỨNG được cung cấp làm chuẩn. Không dùng kiến thức ngoài.

Chấm 4 tiêu chí, mỗi tiêu chí số nguyên 1..5:
1. faithfulness: nội dung câu hỏi/đáp án có đúng và suy ra được từ bằng chứng không.
   1 = mâu thuẫn/bịa; 3 = phần lớn đúng nhưng có chi tiết không kiểm được; 5 = hoàn toàn suy ra được.
2. answerability: chỉ đọc bằng chứng có xác định được DUY NHẤT một đáp án đúng không.
   1 = không trả lời được hoặc nhiều đáp án đúng; 5 = xác định được duy nhất, rõ ràng.
3. distractor_quality: các phương án sai có hợp lý, cùng phạm trù, không loại trừ hiển nhiên không.
   1 = phương án sai vô lý/lạc đề; 5 = đều hợp lý, đòi hỏi phân biệt thật.
4. depth: câu hỏi đòi hỏi mức nhận thức nào.
   1 = tra cứu nguyên văn một cụm từ; 3 = hiểu/diễn giải; 5 = phân tích, tổng hợp, so sánh nhiều ý.

Ngoài ra:
  - gold_supported: true nếu phương án được đánh dấu đúng thực sự là đáp án đúng theo bằng chứng.
  - Nếu bằng chứng có hình ảnh, cân nhắc nội dung hình khi chấm.

CHỈ trả về JSON thuần, không rào ```:
{"faithfulness":int,"answerability":int,"distractor_quality":int,"depth":int,"gold_supported":bool,"note":"<=200 ký tự"}"""


def stable_perm(seed_text: str, n: int) -> list[int]:
    """Deterministic permutation of range(n) from a stable string seed."""
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    order = list(range(n))
    rng.shuffle(order)
    return order


def build_judge_payload(row: dict[str, Any], pkg: dict[str, Any], model: str,
                        max_image_bytes: int = 900_000) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (request_payload, blinding_audit) for one generation cell."""
    parsed = row["parsed_response"]
    choices = list(parsed["choices"])
    gold_old = int(parsed["answer_index"])
    seed_text = f"{row['chunk_id']}|{row['method']}|{row['condition']}|{DEFAULT_SEED}"
    order = stable_perm(seed_text, len(choices))
    shuffled = [choices[i] for i in order]
    gold_new = order.index(gold_old)

    letters = ["A", "B", "C", "D", "E", "F"][: len(shuffled)]
    choice_block = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(shuffled))

    evidence = pkg["evidence"]
    text_block = evidence["text"]
    struct = evidence.get("document_structure")
    struct_block = ""
    if struct:
        struct_block = "\n\n[CẤU TRÚC TÀI LIỆU]\n" + json.dumps(struct, ensure_ascii=False)[:1200]

    user_text = (
        f"[BẰNG CHỨNG]\n{text_block}{struct_block}\n\n"
        f"[CÂU HỎI]\n{parsed['question']}\n\n"
        f"[CÁC PHƯƠNG ÁN]\n{choice_block}\n\n"
        f"[PHƯƠNG ÁN ĐƯỢC ĐÁNH DẤU LÀ ĐÚNG]\n{letters[gold_new]}\n\n"
        "Hãy chấm theo rubric và trả JSON."
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    image_used = 0
    for img in (evidence.get("images") or []):
        img_path = img.get("path")
        if not img_path:
            continue
        p = Path(img_path)
        if not p.is_file():
            continue
        raw_bytes = p.read_bytes()
        if len(raw_bytes) > max_image_bytes:
            continue
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        suffix = p.suffix.lower().lstrip(".") or "jpeg"
        mime = "png" if suffix == "png" else "jpeg"
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}"}})
        image_used += 1

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": content},
        ],
    }
    audit = {
        "choice_permutation": order,
        "gold_index_original": gold_old,
        "gold_index_shown": gold_new,
        "images_attached": image_used,
        "blinded_fields": ["method", "condition", "generator_model", "run_id"],
    }
    return payload, audit


_VALID_ESCAPES = set('"\\/bfnrtu')


def repair_invalid_escapes(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in _VALID_ESCAPES:
                out.append(ch)
                out.append(nxt)
            else:
                out.append("\\\\")
                out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def strip_fence(raw: str) -> str:
    s = raw.strip()
    if not s.startswith("```"):
        return s
    body = s.split("\n", 1)[1] if "\n" in s else ""
    end = body.rfind("```")
    return (body[:end] if end != -1 else body).strip()


DIMS = ("faithfulness", "answerability", "distractor_quality", "depth")


def parse_scores(raw: str) -> dict[str, Any]:
    candidate = strip_fence(raw)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            obj = json.loads(repair_invalid_escapes(candidate))
        except json.JSONDecodeError as exc:
            return {"status": "REJECTED", "reason": f"invalid_json:{type(exc).__name__}"}
    if not isinstance(obj, dict):
        return {"status": "REJECTED", "reason": "json_not_object"}
    scores: dict[str, int] = {}
    for dim in DIMS:
        val = obj.get(dim)
        if not isinstance(val, int) or not 1 <= val <= 5:
            return {"status": "REJECTED", "reason": f"bad_score:{dim}"}
        scores[dim] = val
    return {"status": "SCORED", "scores": scores,
            "gold_supported": bool(obj.get("gold_supported", False)),
            "note": str(obj.get("note", ""))[:200]}


def post(url: str, payload: dict[str, Any], api_key: str | None, timeout: int) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Blinded judge over generation cells (dry-run by default).")
    ap.add_argument("--gen-run", required=True, help="Directory under runs/ holding results.jsonl")
    ap.add_argument("--manifest", type=Path, default=ROOT / "ecm_inputs_final_v2.json")
    ap.add_argument("--out", required=True, help="Output jsonl path (must not exist)")
    ap.add_argument("--judge-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env")
    ap.add_argument("--limit", type=int, default=0, help="0 = all cells")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"refuse to overwrite existing {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"env {args.api_key_env} not set")

    gen_rows = [json.loads(l) for l in (ROOT / "runs" / args.gen_run / "results.jsonl")
                .read_text().splitlines() if l.strip()]
    gen_rows = [r for r in gen_rows if r.get("status") == "PARSED"]
    manifest = json.loads(args.manifest.read_text())
    pkg_map = {(p["chunk_id"], p["condition"]): p for p in manifest["packages"]}

    if args.limit:
        gen_rows = gen_rows[: args.limit]

    counts: dict[str, int] = {}
    cost_total = 0.0
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(gen_rows):
            key = (row["chunk_id"], row["condition"])
            pkg = pkg_map.get(key)
            rec: dict[str, Any]
            if pkg is None:
                rec = {"status": "REJECTED", "reason": "package_missing"}
            else:
                payload, audit = build_judge_payload(row, pkg, args.model)
                if not args.execute:
                    rec = {"status": "PLANNED_DRY_RUN", "blinding_audit": audit}
                else:
                    rec = {"status": "ERROR", "reason": "unattempted"}
                    for attempt in range(args.retries + 1):
                        try:
                            resp = post(args.base_url, payload, api_key, args.timeout)
                            raw = resp["choices"][0]["message"]["content"]
                            rec = parse_scores(raw)
                            rec["raw_response"] = raw
                            usage = resp.get("usage") or {}
                            rec["usage"] = usage
                            cost_total += float(usage.get("cost") or 0.0)
                            rec["retry_count"] = attempt
                            break
                        except urllib.error.HTTPError as exc:
                            rec = {"status": "ERROR", "reason": f"http_{exc.code}",
                                   "retry_count": attempt}
                            if exc.code not in (429, 500, 502, 503, 504):
                                break
                            time.sleep(2 * (attempt + 1))
                        except Exception as exc:  # noqa: BLE001
                            rec = {"status": "ERROR", "reason": f"{type(exc).__name__}",
                                   "retry_count": attempt}
                            time.sleep(2 * (attempt + 1))
                    rec["blinding_audit"] = audit
            rec.update({
                "judge_id": args.judge_id,
                "judge_model": args.model,
                "chunk_id": row["chunk_id"],
                "method": row["method"],
                "condition": row["condition"],
                "generator_model": row.get("model"),
                "cell_index": idx,
            })
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    print(json.dumps({"judge_id": args.judge_id, "model": args.model,
                      "cells": len(gen_rows), "status_counts": counts,
                      "cost_usd": round(cost_total, 4), "out": str(out_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
