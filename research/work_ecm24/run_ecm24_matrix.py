#!/usr/bin/env python3
"""Run the 17-chunk x 3-condition x 3-method matrix for the ECM-24 work dataset.

This mirrors scripts/research/run_omniproxy_matrix_72cell.py's contract
(validate_response, prompt_for, execute_row, retry policy) but reads the
work-24 manifest (schema ecm-tqag.multimodal-inputs.v4-work24, 17 chunks)
instead of the frozen legacy 8-chunk manifest. Output is written under
research/work_ecm24/ only -- it never touches the original 8-chunk results.

Dry-run is the default; --execute is an explicit opt-in to OmniProxy HTTP
POST requests. Each invocation owns a new output directory.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONDITIONS = ("T", "TL_struct", "TLV")
METHODS = ("direct", "answer_first", "ecm")
SCHEMA = "ecm-tqag.multimodal-inputs.v4-work24"
MODEL = "claude-sonnet-4.6"
BASE_URL = "http://127.0.0.1:20131/v1/chat/completions"
PROMPT_VERSION = "ecm-omniproxy-work24-v1"
DEFAULT_SEED = 20260806
DECODING = {"temperature": 0, "max_tokens": 1200}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def strip_code_fence(raw: str) -> tuple[str, bool]:
    """Remove a surrounding Markdown code fence, if present.

    Providers frequently wrap valid JSON in ```json ... ```; the payload is
    unchanged, so unwrapping is transport normalisation, not content repair.
    """
    if not isinstance(raw, str):
        return raw, False
    text = raw.strip()
    if not text.startswith("```"):
        return raw, False
    body = text[3:]
    newline = body.find("\n")
    if newline == -1:
        return raw, False
    first_line = body[:newline].strip().lower()
    if first_line not in ("", "json"):
        return raw, False
    body = body[newline + 1:]
    end = body.rfind("```")
    if end == -1:
        return raw, False
    return body[:end].strip(), True


_JSON_VALID_ESCAPES = set('"\\/bfnrtu')


def repair_invalid_escapes(text: str) -> tuple[str, int]:
    """Escape stray backslashes that JSON forbids but Markdown allows.

    Source chunks contain Markdown escapes such as ``\\*``; when a provider
    copies them verbatim into a JSON string the payload becomes unparsable.
    Doubling only the offending backslash preserves the literal character the
    provider emitted, so this is transport repair rather than content editing.
    """
    if not isinstance(text, str):
        return text, 0
    out: list[str] = []
    repaired = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\" and index + 1 < length:
            nxt = text[index + 1]
            if nxt in _JSON_VALID_ESCAPES:
                out.append(char)
                out.append(nxt)
                index += 2
                continue
            out.append("\\\\")
            out.append(nxt)
            repaired += 1
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out), repaired


def validate_response(method: str, raw: str) -> dict[str, Any]:
    """Validate the method contract; PARSED does not imply factual correctness."""
    candidate, fence_stripped = strip_code_fence(raw)
    escape_repaired = 0
    try:
        parsed = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        repaired_text, escape_repaired = repair_invalid_escapes(candidate)
        if escape_repaired == 0:
            return {"status": "REJECTED", "reason": "invalid_json:JSONDecodeError",
                    "parsed_response": None, "fence_stripped": fence_stripped,
                    "escape_repaired": 0}
        try:
            parsed = json.loads(repaired_text)
        except (TypeError, json.JSONDecodeError) as exc:
            return {"status": "REJECTED", "reason": f"invalid_json:{type(exc).__name__}",
                    "parsed_response": None, "fence_stripped": fence_stripped,
                    "escape_repaired": escape_repaired}
    if not isinstance(parsed, dict):
        return {"status": "REJECTED", "reason": "json_not_object", "parsed_response": None}
    required = ["question", "choices", "answer_index", "rationale"]
    if method == "answer_first":
        required += ["evidence_excerpt", "answer"]
    elif method == "ecm":
        required += ["motif", "derivation", "answer_parts", "trace"]
    elif method not in METHODS:
        return {"status": "REJECTED", "reason": "unknown_method", "parsed_response": parsed}
    missing = [field for field in required if field not in parsed or parsed[field] in (None, "", [], {})]
    if missing:
        return {"status": "REJECTED", "reason": "missing:" + ",".join(missing), "parsed_response": parsed}
    if (not isinstance(parsed["question"], str) or not isinstance(parsed["rationale"], str)
            or not isinstance(parsed["choices"], list) or len(parsed["choices"]) != 4
            or any(not isinstance(choice, str) or not choice.strip() for choice in parsed["choices"])
            or isinstance(parsed["answer_index"], bool) or not isinstance(parsed["answer_index"], int)
            or not 0 <= parsed["answer_index"] < 4):
        return {"status": "REJECTED", "reason": "invalid_core_schema", "parsed_response": parsed}
    if method == "answer_first" and (not isinstance(parsed["evidence_excerpt"], str) or not isinstance(parsed["answer"], str)):
        return {"status": "REJECTED", "reason": "invalid_answer_first_schema", "parsed_response": parsed}
    if method == "ecm":
        trace = parsed["trace"]
        if (not isinstance(parsed["motif"], str) or not isinstance(parsed["derivation"], str)
                or not isinstance(parsed["answer_parts"], list) or not parsed["answer_parts"]
                or not isinstance(trace, list) or not trace
                or any(not isinstance(x, dict) or x.get("source") not in {"text", "structure", "image"}
                       or not isinstance(x.get("support"), str) or not x["support"].strip() for x in trace)):
            return {"status": "REJECTED", "reason": "invalid_ecm_schema", "parsed_response": parsed}
    return {"status": "PARSED", "reason": None, "parsed_response": parsed}


def prompt_for(evidence: dict[str, Any], method: str) -> tuple[str, list[str]]:
    fields = ["evidence.text"]
    text = evidence["text"]
    structure = evidence.get("document_structure")
    structure_text = ""
    if isinstance(structure, dict):
        fields += ["evidence.document_structure.section_title", "evidence.document_structure.header_level",
                   "evidence.document_structure.declared_image_order"]
        structure_text = "\nCấu trúc tài liệu:\n- Tiêu đề mục: %s\n- Cấp mục: %s\n- Thứ tự ảnh đã khai báo: %s\n" % (
            structure["section_title"], structure["header_level"], structure["declared_image_order"])
    image_text = ""
    if "images" in evidence:
        fields.append("evidence.images[pixels_attached]")
        image_text = "\nẢnh gốc liên quan đã được đính kèm. Chỉ dùng quan sát trực tiếp khi cần.\n"
    common = ("Tạo một câu hỏi trắc nghiệm tiếng Việt chỉ từ bằng chứng dưới đây. Không suy đoán ngoài bằng chứng. "
              "Trả đúng một đối tượng JSON, không markdown.\n\nBằng chứng văn bản:\n" + text + structure_text + image_text +
              "\nchoices phải là đúng bốn chuỗi không rỗng; answer_index là số nguyên 0–3.\n")
    contracts = {
        "direct": '{"question":"...","choices":["...","...","...","..."],"answer_index":0,"rationale":"..."}',
        "answer_first": '{"evidence_excerpt":"...","answer":"...","question":"...","choices":["...","...","...","..."],"answer_index":0,"rationale":"..."}',
        "ecm": ('Nêu motif, derivation, answer_parts và trace trước khi viết câu hỏi. trace gồm source=text|structure|image và support.\n'
                '{"motif":"...","derivation":"...","answer_parts":["..."],"trace":[{"source":"text","support":"..."}],"question":"...","choices":["...","...","...","..."],"answer_index":0,"rationale":"..."}')}
    if method not in contracts:
        raise ValueError("unknown method")
    return common + contracts[method], fields


def image_part(image: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if (not isinstance(image, dict) or not isinstance(image.get("path"), str)
            or isinstance(image.get("bytes"), bool) or not isinstance(image.get("bytes"), int)
            or image["bytes"] < 1 or not isinstance(image.get("sha256"), str)
            or len(image["sha256"]) != 64 or not isinstance(image.get("declared_order"), int)
            or image["declared_order"] < 1):
        raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_image_record")
    path = Path(image["path"])
    if not path.is_file() or path.stat().st_size != image["bytes"] or sha256_file(path) != image["sha256"]:
        raise ValueError(f"BLOCKED_INPUT_INTEGRITY:image_hash_or_size_mismatch:{path.name}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}, {
        "bytes": image["bytes"], "sha256": image["sha256"], "declared_order": image["declared_order"]}


def load_manifest(manifest_path: Path) -> tuple[dict[str, Any], str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"BLOCKED_INPUT_INTEGRITY:manifest_unreadable:{type(exc).__name__}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("BLOCKED_INPUT_INTEGRITY:unexpected_manifest_schema")
    packages = manifest.get("packages")
    if not isinstance(packages, list) or len(packages) == 0:
        raise ValueError("BLOCKED_INPUT_INTEGRITY:no_packages")
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("chunk_id"), str) or package.get("condition") not in CONDITIONS:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_package")
        key = (package["chunk_id"], package["condition"])
        if key in keyed:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:duplicate_chunk_condition")
        evidence = package.get("evidence")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("text"), str) or not evidence["text"].strip():
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_text")
        forbidden = ("![](", "_page_")
        if any(marker in evidence["text"] for marker in forbidden):
            raise ValueError("BLOCKED_INPUT_INTEGRITY:unclean_text")
        condition = package["condition"]
        if condition == "T" and set(evidence) != {"text"}:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_T_evidence")
        if condition == "TL_struct" and set(evidence) != {"text", "document_structure"}:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_TL_struct_evidence")
        if condition == "TLV" and set(evidence) != {"text", "document_structure", "images"}:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_TLV_evidence")
        if condition != "T":
            structure = evidence.get("document_structure")
            if not isinstance(structure, dict) or set(structure) != {"section_title", "header_level", "declared_image_order"}:
                raise ValueError("BLOCKED_INPUT_INTEGRITY:unsafe_structure")
            if (not isinstance(structure["section_title"], str)
                    or isinstance(structure["header_level"], bool)
                    or not isinstance(structure["header_level"], int)
                    or not isinstance(structure["declared_image_order"], list)
                    or structure["declared_image_order"] != list(range(1, len(structure["declared_image_order"]) + 1))):
                raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_structure")
        if condition == "TLV":
            images = evidence.get("images")
            if not isinstance(images, list) or not images:
                raise ValueError("BLOCKED_INPUT_INTEGRITY:missing_tlv_images")
            if evidence["document_structure"]["declared_image_order"] != list(range(1, len(images) + 1)):
                raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_declared_image_order")
            for image in images:
                image_part(image)  # validates bytes; data is discarded
        keyed[key] = package
    chunks = {key[0] for key in keyed}
    if any((chunk, condition) not in keyed for chunk in chunks for condition in CONDITIONS):
        raise ValueError("BLOCKED_INPUT_INTEGRITY:incomplete_chunk_x_condition")
    for chunk in chunks:
        text = keyed[(chunk, "T")]["evidence"]["text"]
        structure = keyed[(chunk, "TL_struct")]["evidence"]["document_structure"]
        tlv = keyed[(chunk, "TLV")]["evidence"]
        if text != keyed[(chunk, "TL_struct")]["evidence"]["text"] or text != tlv["text"]:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:unpaired_text")
        if structure != tlv["document_structure"]:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:unpaired_structure")
    return manifest, sha256_file(manifest_path)


def build_run_plan(manifest_path: Path, experiment_id: str, seed: int = DEFAULT_SEED, model: str = MODEL,
                   conditions: tuple[str, ...] | None = None, chunk_ids: set[str] | None = None) -> list[dict[str, Any]]:
    manifest, manifest_hash = load_manifest(manifest_path)
    selected_conditions = CONDITIONS if conditions is None else conditions
    if not selected_conditions or any(condition not in CONDITIONS for condition in selected_conditions):
        raise ValueError("invalid selected conditions")
    rows: list[dict[str, Any]] = []
    for package in manifest["packages"]:
        if package["condition"] not in selected_conditions:
            continue
        if chunk_ids is not None and package["chunk_id"] not in chunk_ids:
            continue
        for method in METHODS:
            evidence = package["evidence"]
            prompt, fields = prompt_for(evidence, method)
            parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            image_audit: list[dict[str, Any]] = []
            for image in evidence.get("images", []):
                part, audit = image_part(image)
                parts.append(part)
                image_audit.append(audit)
            request = {"model": model, **DECODING, "messages": [{"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": parts}]}
            input_hash = sha256_bytes(canonical({"package": package, "method": method}).encode())
            rows.append({"experiment_id": experiment_id, "run_id": sha256_bytes(f"{experiment_id}|{input_hash}".encode())[:24],
                         "chunk_id": package["chunk_id"], "doc_id": package.get("doc_id"), "condition": package["condition"], "method": method,
                         "source_hash": manifest_hash, "input_hash": input_hash, "model": model,
                         "prompt_version": PROMPT_VERSION, "decoding": DECODING, "prompt_sha256": sha256_bytes(prompt.encode()),
                         "included_fields": fields, "image_audit": image_audit, "request": request})
    random.Random(seed).shuffle(rows)
    for order, row in enumerate(rows, 1):
        row["run_order"] = order
    return rows


def post_json(url: str, payload: dict[str, Any], timeout_sec: int, api_key: str | None = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # nosec B310: explicit --execute only
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("response_not_object")
    return value


def execute_row(row: dict[str, Any], base_url: str, timeout_sec: int, retries: int, api_key: str | None = None) -> dict[str, Any]:
    raw, reason = "", None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            response = post_json(base_url, row["request"], timeout_sec, api_key)
            content = response.get("choices", [{}])[0].get("message", {}).get("content")
            if not isinstance(content, str):
                return {"status": "REJECTED", "reason": "invalid_provider_response", "raw_response": "", "parsed_response": None,
                        "retry_count": attempt, "elapsed_sec": round(time.perf_counter() - start, 3)}
            result = validate_response(row["method"], content)
            return {**result, "raw_response": content, "retry_count": attempt, "elapsed_sec": round(time.perf_counter() - start, 3)}
        except urllib.error.HTTPError as exc:
            reason = f"http_{exc.code}"
            transient = exc.code == 429 or 500 <= exc.code < 600
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            reason, transient = f"transport:{type(exc).__name__}", True
        if not transient or attempt == retries:
            break
        time.sleep(min(2 ** attempt, 4))
    return {"status": "ERROR", "reason": reason, "raw_response": raw, "parsed_response": None,
            "retry_count": retries, "elapsed_sec": None}


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise ValueError(f"Refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)


def run(manifest_path: Path, output_dir: Path, experiment_id: str, seed: int = DEFAULT_SEED, *, execute: bool = False,
        base_url: str = BASE_URL, model: str = MODEL, timeout_sec: int = 90, retries: int = 2,
        conditions: tuple[str, ...] | None = None, chunk_ids: set[str] | None = None,
        api_key: str | None = None, api_key_env: str | None = None, provider: str = "omniproxy") -> dict[str, Any]:
    if retries < 0 or timeout_sec <= 0:
        raise ValueError("timeout and retries must be positive")
    plan = build_run_plan(manifest_path, experiment_id, seed, model, conditions, chunk_ids)
    if not plan:
        raise ValueError("filters selected no design cells")
    prepare_output_dir(output_dir)
    results_path, audit_path = output_dir / "results.jsonl", output_dir / "prompt_audit.jsonl"
    counts: dict[str, int] = {}
    with results_path.open("x", encoding="utf-8") as results, audit_path.open("x", encoding="utf-8") as audit:
        for row in plan:
            audit.write(json.dumps({key: row[key] for key in ("run_id", "prompt_version", "prompt_sha256", "included_fields")}, ensure_ascii=False) + "\n")
            result = execute_row(row, base_url, timeout_sec, retries, api_key) if execute else {"status": "PLANNED_DRY_RUN", "reason": None, "raw_response": None, "parsed_response": None, "retry_count": 0, "elapsed_sec": 0.0}
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            record = {key: row[key] for key in ("experiment_id", "run_id", "chunk_id", "doc_id", "source_hash", "condition", "method", "model", "prompt_version", "decoding", "input_hash", "prompt_sha256", "run_order", "image_audit")}
            record.update({"provider": provider, "base_url": base_url, "seed": seed, "created_utc": utc_now(), **result})
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()
    summary = {"status": "EXECUTED" if execute else "PLANNED_DRY_RUN_NO_API_CALLS", "api_call_count": len(plan) if execute else 0,
               "cell_count": len(plan), "status_counts": counts, "seed": seed, "model": model,
               "design_scope": {"conditions": list(CONDITIONS if conditions is None else conditions),
                                "chunk_ids": sorted(chunk_ids) if chunk_ids is not None else "ALL_WORK24_CHUNKS",
                                "methods": list(METHODS)},
               "authentication": f"environment:{api_key_env}" if api_key_env else "none",
               "manifest_sha256": plan[0]["source_hash"], "results_path": str(results_path),
               "prompt_audit_path": str(audit_path), "retry_policy": {"max_retries": retries, "transient": ["429", "5xx", "transport"]}}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="17-chunk x 3-condition x 3-method work-24 runner; dry-run by default.")
    parser.add_argument("--manifest", type=Path, default=root / "ecm_inputs_24chunks_v1.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="New, nonexistent output directory.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--execute", action="store_true", help="Explicitly permit OmniProxy HTTP POST requests.")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--chunk-id", action="append")
    parser.add_argument("--api-key-env")
    parser.add_argument("--provider", default="omniproxy")
    args = parser.parse_args()
    conditions = tuple(args.condition) if args.condition else None
    chunk_ids = set(args.chunk_id) if args.chunk_id else None
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        parser.error(f"environment variable {args.api_key_env} is not set")
    try:
        print(json.dumps(run(args.manifest, args.out_dir, args.experiment_id, args.seed, execute=args.execute, base_url=args.base_url,
                             model=args.model, timeout_sec=args.timeout_sec, retries=args.retries,
                             conditions=conditions, chunk_ids=chunk_ids, api_key=api_key,
                             api_key_env=args.api_key_env, provider=args.provider), ensure_ascii=False, indent=2))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
