#!/usr/bin/env python3
"""ECM-v2 runner: closed-catalog motif + compiled program + executed anchors.

Reuses the frozen v1 manifest loader / request builder / transport from
run_ecm24_matrix.py so the ONLY thing that differs from the v1 `ecm` arm is:
  * the ECM prompt (motif must come from the closed catalog; steps are typed)
  * the acceptance gate (ecm_v2_core.guard_ecm_v2 instead of a str isinstance)

Dry-run by default. --execute is required to make any HTTP call.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_ecm24_matrix as v1  # noqa: E402  (frozen v1 machinery)
import ecm_v2_core as core  # noqa: E402

METHOD = "ecm_v2"
PROMPT_VERSION = "ecm_v2.p1"
DECODING = v1.DECODING


def catalog_spec_text() -> str:
    """Human-readable rendering of the closed catalog for the prompt."""
    lines = []
    for name, spec in core.MOTIF_CATALOG.items():
        ops = " -> ".join(
            f'{s["op"]}({"anchor" if s["needs"] else "derived"})' for s in spec["steps"]
        )
        desc = "; ".join(s["desc"] for s in spec["steps"])
        lines.append(f'- "{name}": {ops}\n    {desc}')
    return "\n".join(lines)


def prompt_for_v2(evidence: dict[str, Any]) -> tuple[str, list[str]]:
    fields = ["evidence.text"]
    text = evidence["text"]
    structure = evidence.get("document_structure")
    structure_text = ""
    if isinstance(structure, dict):
        fields += [
            "evidence.document_structure.section_title",
            "evidence.document_structure.header_level",
            "evidence.document_structure.declared_image_order",
        ]
        structure_text = (
            "\nCấu trúc tài liệu:\n- Tiêu đề mục: %s\n- Cấp mục: %s\n"
            "- Thứ tự ảnh đã khai báo: %s\n"
            % (
                structure["section_title"],
                structure["header_level"],
                structure["declared_image_order"],
            )
        )
    image_text = ""
    if "images" in evidence:
        fields.append("evidence.images[pixels_attached]")
        image_text = "\nẢnh gốc liên quan đã được đính kèm. Chỉ dùng quan sát trực tiếp khi cần.\n"

    prompt = (
        "Tạo một câu hỏi trắc nghiệm tiếng Việt CHỈ từ bằng chứng dưới đây. "
        "Không suy đoán ngoài bằng chứng. Trả đúng một đối tượng JSON, không markdown.\n\n"
        "Bằng chứng văn bản:\n" + text + structure_text + image_text +
        "\n=== QUY TRÌNH BẮT BUỘC ===\n"
        "Bước 1. Chọn MỘT motif từ danh mục ĐÓNG sau (dùng đúng tên, không tự đặt tên khác):\n"
        + catalog_spec_text() +
        "\n\nBước 2. Viết `steps` đúng số bước và đúng thứ tự `op` của motif đã chọn.\n"
        "  - Bước có (anchor): PHẢI có field `anchor` = một đoạn TRÍCH NGUYÊN VĂN từ bằng chứng "
        "(>= 10 ký tự) chứa dữ kiện của bước đó, và `result` = dữ kiện rút ra.\n"
        "  - Bước (derived): KHÔNG cần `anchor`, chỉ cần `result` = kết quả của phép suy luận.\n"
        "  - HAI anchor PHẢI trích từ HAI CÂU KHÁC NHAU của bằng chứng. Nếu chỉ có thể lấy từ một câu, "
        "hãy chọn motif khác.\n"
        "\nBước 3. Viết câu hỏi sao cho người trả lời PHẢI thực hiện đủ các bước trên mới trả lời được. "
        "Không được hỏi lại một dữ kiện đơn lẻ.\n"
        "  - CẤM sao chép nguyên văn: câu hỏi KHÔNG được chứa chuỗi 8 từ liên tiếp trùng bằng chứng. "
        "Hãy diễn đạt lại.\n"
        "\nBước 4. Bốn lựa chọn: đúng một đáp án đúng. Ba phương án nhiễu phải là kết quả của việc "
        "làm SAI MỘT bước ở trên (ví dụ dùng sai anchor, bỏ bước suy luận, đảo chiều so sánh), "
        "không phải phương án sai hiển nhiên.\n"
        "\nJSON contract:\n"
        '{"motif":"<tên trong danh mục>",'
        '"steps":[{"op":"locate","anchor":"<trích nguyên văn>","result":"..."},'
        '{"op":"locate","anchor":"<trích nguyên văn câu KHÁC>","result":"..."},'
        '{"op":"<op cuối của motif>","result":"..."}],'
        '"question":"...","choices":["...","...","...","..."],'
        '"answer_index":0,"rationale":"...","distractor_faults":["...","...","..."]}\n'
    )
    return prompt, fields


def validate_v2(raw: str, evidence_text: str, idf_table, idf_cut) -> dict[str, Any]:
    """Transport-level parse (reused from v1) then the ECM-v2 semantic gate."""
    candidate, fence_stripped = v1.strip_code_fence(raw)
    escape_repaired = 0
    try:
        parsed = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        repaired_text, escape_repaired = v1.repair_invalid_escapes(candidate)
        if escape_repaired == 0:
            return {"status": "REJECTED", "reason": "invalid_json:JSONDecodeError",
                    "parsed_response": None, "fence_stripped": fence_stripped,
                    "escape_repaired": 0, "gate": None}
        try:
            parsed = json.loads(repaired_text)
        except (TypeError, json.JSONDecodeError) as exc:
            return {"status": "REJECTED", "reason": f"invalid_json:{type(exc).__name__}",
                    "parsed_response": None, "fence_stripped": fence_stripped,
                    "escape_repaired": escape_repaired, "gate": None}
    if not isinstance(parsed, dict):
        return {"status": "REJECTED", "reason": "json_not_object",
                "parsed_response": None, "fence_stripped": fence_stripped,
                "escape_repaired": escape_repaired, "gate": None}

    # core MCQ schema (identical to v1 so the arms stay comparable)
    for field in ("question", "choices", "answer_index", "rationale"):
        if field not in parsed or parsed[field] in (None, "", [], {}):
            return {"status": "REJECTED", "reason": f"missing:{field}",
                    "parsed_response": parsed, "fence_stripped": fence_stripped,
                    "escape_repaired": escape_repaired, "gate": None}
    if (not isinstance(parsed["question"], str) or not isinstance(parsed["rationale"], str)
            or not isinstance(parsed["choices"], list) or len(parsed["choices"]) != 4
            or any(not isinstance(c, str) or not c.strip() for c in parsed["choices"])
            or isinstance(parsed["answer_index"], bool)
            or not isinstance(parsed["answer_index"], int)
            or not 0 <= parsed["answer_index"] < 4):
        return {"status": "REJECTED", "reason": "invalid_core_schema",
                "parsed_response": parsed, "fence_stripped": fence_stripped,
                "escape_repaired": escape_repaired, "gate": None}

    gate = core.guard_ecm_v2(parsed, evidence_text, idf_cut, idf_table)
    return {"status": gate["status"],
            "reason": gate["reason"] if gate["status"] != "PARSED" else None,
            "parsed_response": parsed, "fence_stripped": fence_stripped,
            "escape_repaired": escape_repaired, "gate": gate}


def build_plan(manifest_path: Path, experiment_id: str, seed: int, model: str,
               conditions, chunk_ids) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, manifest_hash = v1.load_manifest(manifest_path)
    selected = v1.CONDITIONS if conditions is None else conditions
    rows: list[dict[str, Any]] = []
    for package in manifest["packages"]:
        if package["condition"] not in selected:
            continue
        if chunk_ids is not None and package["chunk_id"] not in chunk_ids:
            continue
        evidence = package["evidence"]
        prompt, fields = prompt_for_v2(evidence)
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_audit: list[dict[str, Any]] = []
        for image in evidence.get("images", []):
            part, audit = v1.image_part(image)
            parts.append(part)
            image_audit.append(audit)
        request = {"model": model, **DECODING,
                   "messages": [{"role": "system", "content": "Return valid JSON only."},
                                {"role": "user", "content": parts}]}
        input_hash = v1.sha256_bytes(
            v1.canonical({"package": package, "method": METHOD}).encode())
        rows.append({
            "experiment_id": experiment_id,
            "run_id": v1.sha256_bytes(f"{experiment_id}|{input_hash}".encode())[:24],
            "chunk_id": package["chunk_id"], "doc_id": package.get("doc_id"),
            "condition": package["condition"], "method": METHOD,
            "source_hash": manifest_hash, "input_hash": input_hash, "model": model,
            "prompt_version": PROMPT_VERSION, "decoding": DECODING,
            "prompt_sha256": v1.sha256_bytes(prompt.encode()),
            "included_fields": fields, "image_audit": image_audit,
            "request": request, "evidence_text": evidence["text"],
        })
    random.Random(seed).shuffle(rows)
    for order, row in enumerate(rows, 1):
        row["run_order"] = order
    return rows, manifest


def run(manifest_path: Path, output_dir: Path, experiment_id: str, seed: int, *,
        execute: bool, base_url: str, model: str, timeout_sec: int, retries: int,
        conditions, chunk_ids, api_key: str | None, api_key_env: str | None,
        provider: str) -> dict[str, Any]:
    plan, manifest = build_plan(manifest_path, experiment_id, seed, model,
                               conditions, chunk_ids)
    if not plan:
        raise ValueError("filters selected no design cells")
    # IDF built from the manifest corpus (deterministic, no API)
    texts = sorted({p["evidence"]["text"] for p in manifest["packages"]})
    idf_table, idf_cut = core.build_idf(texts)

    v1.prepare_output_dir(output_dir)
    results_path = output_dir / "results.jsonl"
    audit_path = output_dir / "prompt_audit.jsonl"
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    with results_path.open("x", encoding="utf-8") as results, \
            audit_path.open("x", encoding="utf-8") as audit:
        for row in plan:
            audit.write(json.dumps({k: row[k] for k in (
                "run_id", "prompt_version", "prompt_sha256", "included_fields")},
                ensure_ascii=False) + "\n")
            if not execute:
                result = {"status": "PLANNED_DRY_RUN", "reason": None,
                          "raw_response": None, "parsed_response": None,
                          "retry_count": 0, "elapsed_sec": 0.0, "gate": None}
            else:
                result = execute_row_v2(row, base_url, timeout_sec, retries,
                                        api_key, idf_table, idf_cut)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if result.get("reason"):
                key = str(result["reason"]).split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
            record = {k: row[k] for k in (
                "experiment_id", "run_id", "chunk_id", "doc_id", "source_hash",
                "condition", "method", "model", "prompt_version", "decoding",
                "input_hash", "prompt_sha256", "run_order", "image_audit")}
            record.update({"provider": provider, "base_url": base_url, "seed": seed,
                           "created_utc": v1.utc_now(), **result})
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()

    summary = {
        "status": "EXECUTED" if execute else "PLANNED_DRY_RUN_NO_API_CALLS",
        "api_call_count": len(plan) if execute else 0,
        "cell_count": len(plan), "status_counts": counts,
        "reject_reason_families": reasons,
        "seed": seed, "model": model, "method": METHOD,
        "prompt_version": PROMPT_VERSION,
        "motif_catalog": sorted(core.MOTIF_CATALOG),
        "guard": {"min_steps": 2, "min_distinct_anchor_sentences": 2,
                  "max_verbatim_run": core.MAX_VERBATIM_RUN,
                  "idf_cut": round(idf_cut, 4)},
        "design_scope": {
            "conditions": list(v1.CONDITIONS if conditions is None else conditions),
            "chunk_ids": sorted(chunk_ids) if chunk_ids is not None else "ALL_WORK24_CHUNKS",
            "methods": [METHOD]},
        "authentication": f"file:{api_key_env}" if api_key_env else "none",
        "manifest_sha256": plan[0]["source_hash"],
        "results_path": str(results_path),
        "scope_note": ("PARSED means the record satisfied the compiled-program and "
                       "executed-anchor contract; it does NOT establish question "
                       "quality, which requires blinded judging."),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def execute_row_v2(row, base_url, timeout_sec, retries, api_key,
                   idf_table, idf_cut) -> dict[str, Any]:
    reason = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            response = v1.post_json(base_url, row["request"], timeout_sec, api_key)
            content = response.get("choices", [{}])[0].get("message", {}).get("content")
            usage = response.get("usage")
            if not isinstance(content, str):
                return {"status": "REJECTED", "reason": "invalid_provider_response",
                        "raw_response": "", "parsed_response": None, "gate": None,
                        "retry_count": attempt,
                        "elapsed_sec": round(time.perf_counter() - start, 3),
                        "usage": usage}
            result = validate_v2(content, row["evidence_text"], idf_table, idf_cut)
            # TRANSPORT-ONLY retry: a stream that was cut mid-flight yields an
            # opening ``` fence with no closing fence and unparseable JSON. That
            # is a delivery failure, not a contract violation, so it is retried.
            # Guard rejections (anti_verbatim / exec:* / compile:*) are NEVER
            # retried - re-rolling them would silently inflate the accept rate.
            truncated = (
                result["status"] == "REJECTED"
                and str(result.get("reason", "")).startswith("invalid_json")
                and content.lstrip().startswith("```")
                and content.rstrip().count("```") < 2
            )
            if truncated and attempt < retries:
                reason = "transport:truncated_stream"
                time.sleep(min(2 ** attempt, 4))
                continue
            return {**result, "raw_response": content, "retry_count": attempt,
                    "elapsed_sec": round(time.perf_counter() - start, 3),
                    "usage": usage,
                    "transport_truncation_retried": bool(truncated)}
        except Exception as exc:  # noqa: BLE001 - classified below
            code = getattr(exc, "code", None)
            reason = f"http_{code}" if code else f"transport:{type(exc).__name__}"
            transient = code == 429 or (code is not None and 500 <= code < 600) or code is None
        if not transient or attempt == retries:
            break
        time.sleep(min(2 ** attempt, 4))
    return {"status": "ERROR", "reason": reason, "raw_response": "",
            "parsed_response": None, "gate": None, "retry_count": retries,
            "elapsed_sec": None, "usage": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="ECM-v2 runner; dry-run by default.")
    # Default is the SCREENED manifest (16 chunks x 3 conditions, 0 unclean-text
    # packages). ecm_inputs_24chunks_v1.json is the pre-screening raw pool and
    # still carries OCR markers, so it fails load_manifest by design.
    parser.add_argument("--manifest", type=Path, default=ROOT / "ecm_inputs_final_v2.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=v1.DEFAULT_SEED)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default=v1.BASE_URL)
    parser.add_argument("--model", default=v1.MODEL)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--condition", action="append", choices=v1.CONDITIONS)
    parser.add_argument("--chunk-id", action="append")
    parser.add_argument("--api-key-file")
    parser.add_argument("--provider", default="openrouter")
    args = parser.parse_args()
    api_key = None
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
        if not api_key:
            parser.error("api key file is empty")
    try:
        print(json.dumps(run(
            args.manifest, args.out_dir, args.experiment_id, args.seed,
            execute=args.execute, base_url=args.base_url, model=args.model,
            timeout_sec=args.timeout_sec, retries=args.retries,
            conditions=tuple(args.condition) if args.condition else None,
            chunk_ids=set(args.chunk_id) if args.chunk_id else None,
            api_key=api_key, api_key_env=args.api_key_file,
            provider=args.provider), ensure_ascii=False, indent=2))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
