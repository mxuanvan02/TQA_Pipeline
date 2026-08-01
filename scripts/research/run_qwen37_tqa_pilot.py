#!/usr/bin/env python3
"""Run a non-overwriting graph--motif--program TQA experiment.

The runner accepts an immutable 8-chunk x 3-condition evidence manifest and
writes a JSONL ledger. Direct and Answer-first use one generation request. ECM
uses two requests per cell: a planner proposes a source-bound document graph
and motif request; local code matches, compiles, executes, and seals the
construction; then an isolated realizer emits one MCQ from that construction.

A PARSED record is schema- and provenance-valid only. It is not a claim of
semantic, legal, pedagogical, or visual-grounding correctness.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ecm_graph_program import (  # noqa: E402 -- local repository module
    ContractError,
    GRAPH_SCHEMA,
    MOTIF_CATALOG,
    anchor_errors,
    atom_choice_binding,
    build_construction,
    validate_construction,
)

CONDITIONS = ("T", "TL_struct", "TLV")
METHODS = ("direct", "answer_first", "ecm")
SCHEMA = "ecm-tqag.multimodal-inputs.v3"
DEFAULT_MODEL = "qwen/qwen3.7-plus"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DECODING = {"temperature": 0, "max_tokens": 4000}
PROMPT_VERSION = "ecm-qwen37-graph-motif-rpl-rho-trace-v8"
REJECT_REASONS = {"insufficient_evidence", "no_graph_motif_match", "image_not_necessary"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def image_part(image: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(str(image.get("path", "")))
    expected_size, expected_sha = image.get("bytes"), image.get("sha256")
    if (not path.is_file() or isinstance(expected_size, bool) or not isinstance(expected_size, int)
            or path.stat().st_size != expected_size or not isinstance(expected_sha, str)
            or sha256_file(path) != expected_sha):
        raise ValueError(f"BLOCKED_INPUT_INTEGRITY:image:{path.name}")
    media = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return ({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}},
            {"bytes": expected_size, "sha256": expected_sha, "declared_order": image.get("declared_order")})


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or not isinstance(value.get("packages"), list) or len(value["packages"]) != 24:
        raise ValueError("BLOCKED_INPUT_INTEGRITY:unexpected_manifest")
    package_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for package in value["packages"]:
        chunk, condition, evidence = package.get("chunk_id"), package.get("condition"), package.get("evidence")
        if not isinstance(chunk, str) or condition not in CONDITIONS or not isinstance(evidence, dict) or not isinstance(evidence.get("text"), str):
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_package")
        key = (chunk, condition)
        if key in package_by_key:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:duplicate_package")
        if condition == "T" and set(evidence) != {"text"}:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_T")
        if condition == "TL_struct" and set(evidence) != {"text", "document_structure"}:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_TL_struct")
        if condition == "TLV":
            if set(evidence) != {"text", "document_structure", "images"} or not evidence["images"]:
                raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_TLV")
            for image in evidence["images"]:
                image_part(image)
        package_by_key[key] = package
    chunks = {chunk for chunk, _ in package_by_key}
    if len(chunks) != 8 or any((chunk, condition) not in package_by_key for chunk in chunks for condition in CONDITIONS):
        raise ValueError("BLOCKED_INPUT_INTEGRITY:matrix_not_8x3")
    return value, sha256_file(path)


def evidence_text(evidence: dict[str, Any]) -> tuple[str, list[str]]:
    text = evidence["text"]
    structure = evidence.get("document_structure")
    fields = ["evidence.text"]
    structure_text = ""
    if isinstance(structure, dict):
        fields += ["evidence.document_structure.section_title", "evidence.document_structure.header_level", "evidence.document_structure.declared_image_order"]
        structure_text = "\nCẤU TRÚC TÀI LIỆU:\n" + canonical(structure)
    image_text = ""
    if evidence.get("images"):
        fields.append("evidence.images[pixels_attached]")
        image_text = "\nẢNH GỐC đã được đính kèm theo đúng thứ tự khai báo."
    return "EVIDENCE TEXT:\n" + text + structure_text + image_text + "\n", fields


def common_quality_gate() -> str:
    return """QUY TẮC CHUNG:
- Chỉ dùng evidence trong input; không dùng kiến thức ngoài evidence.
- Không tự tạo nhãn Hình A/B/C và không diễn giải ý nghĩa pháp lý của mũi tên nếu ảnh không ghi rõ.
- MCQ phải có đúng một đáp án tốt nhất; ba phương án còn lại phải bị evidence loại trừ.
- Bốn lựa chọn không được trùng nghĩa.
- Nếu evidence không đủ để thỏa các điều kiện, chỉ trả {\"reject_reason\":\"insufficient_evidence\"}.
- Trả JSON hợp lệ duy nhất, không markdown và không giải thích ngoài JSON.
"""


def direct_or_answer_prompt(evidence: dict[str, Any], method: str, condition: str) -> tuple[str, list[str]]:
    rendered, fields = evidence_text(evidence)
    allowed_anchor_sources = {
        "T": "text",
        "TL_struct": "text hoặc structure",
        "TLV": "text, structure hoặc image",
    }[condition]
    contracts = {
        "direct": """DIRECT:
Hỏi một mệnh đề trực tiếp được nêu hoặc quan sát trực tiếp trong evidence. Chỉ cần một evidence chính; không dùng câu hỏi đòi hỏi ghép nhiều quan hệ.
SCHEMA:
{\"question\":\"...\",\"choices\":[\"...\",\"...\",\"...\",\"...\"],\"answer_index\":0,\"rationale\":\"...\",\"evidence_anchor\":[{\"source\":\"text|structure\",\"support\":\"...\"}]}
""",
        "answer_first": """ANSWER-FIRST:
Chọn trước một kết luận có căn cứ, rồi dựng tình huống hoặc yêu cầu phân loại. Ba distractor phải cùng trường khái niệm và sai có chủ đích do nhầm chủ thể, điều kiện, phạm vi hoặc hệ quả. Không chỉ diễn đạt lại một câu Direct.
SCHEMA:
{\"evidence_excerpt\":\"...\",\"answer\":\"phải bằng choices[answer_index]\",\"question\":\"...\",\"choices\":[\"...\",\"...\",\"...\",\"...\"],\"answer_index\":0,\"rationale\":\"...\",\"evidence_anchor\":[{\"source\":\"text|structure\",\"support\":\"...\"}]}
""",
    }
    if method not in contracts:
        raise ValueError("unknown_non_ecm_method")
    # These anchor rules are byte-for-byte the same obligations the ECM planner
    # receives. Stating them for only one protocol would confound the
    # comparison: a baseline would be rejected for a rule it was never told.
    anchor_rule = (
        f"QUY TẮC EVIDENCE_ANCHOR (bắt buộc):\n"
        f"- evidence_anchor.source chỉ được là: {allowed_anchor_sources}.\n"
        "- Với source=text: support phải TRÍCH NGUYÊN VĂN một đoạn liên tục có trong EVIDENCE TEXT, không rút gọn, không viết lại, không ghép hai đoạn rời.\n"
        "- Với source=structure: support phải đúng bằng section_title đã khai báo.\n"
        "- Với source=image: support phải mô tả quan sát trực tiếp và nêu rõ “Ảnh n” đúng thứ tự ảnh đã khai báo.\n"
    )
    return "Bạn tạo ĐÚNG MỘT câu hỏi trắc nghiệm tiếng Việt.\n" + common_quality_gate() + "\n" + anchor_rule + contracts[method] + "\n" + rendered, fields


def ecm_planner_prompt(evidence: dict[str, Any], condition: str) -> tuple[str, list[str]]:
    rendered, fields = evidence_text(evidence)
    image_rule = ""
    if condition == "TLV":
        image_rule = "Vì đây là TLV, motif_request phải chọn ít nhất một node source=image và image_necessary=true.\n"
    motif_ids = ", ".join(sorted(MOTIF_CATALOG))
    return """Bạn là bước GRAPH PROPOSAL của ECM. Bạn CHƯA được tạo câu hỏi, lựa chọn, chương trình hay answer atom.
Hãy đề xuất một document graph source-bound và một motif request. Matcher, compiler, executor và answer_atoms sẽ do code cục bộ xác định; tuyệt đối không tự xuất các artifact đó.

""" + common_quality_gate() + f"""
QUY TẮC DOCUMENT GRAPH:
- document_graph.schema phải là `{GRAPH_SCHEMA}`.
- Graph có ít nhất hai evidence node và đúng một claim node.
- Evidence node: kind=evidence; ID T1... cho text, S1... cho structure, I1... cho image; role thuộc premise|mapping|constraint.
- Text value phải trích nguyên văn. Structure value phải đúng section_title. Image value phải mô tả quan sát trực tiếp và nêu rõ “Ảnh n”.
- Claim node: ID C1..., kind=claim, source=derived, role=conclusion; value là kết luận dự kiến được nhiều evidence node cùng hỗ trợ.
- Mỗi edge có ID E1..., directed=true, đi từ evidence node đến claim node; relation thuộc supports|maps_to|constrains.
- motif_id chỉ được thuộc catalog đóng: {motif_ids}. Vai trò các evidence_node_ids phải khớp chính xác motif đã chọn.
- Không xuất restricted_program hoặc answer_atoms; code sẽ match motif, compile program và execute để sinh atom.
- Nếu không tạo được graph nhiều evidence thật sự, chỉ trả {{\"reject_reason\":\"no_graph_motif_match\"}}.
""" + image_rule + """
SCHEMA PLANNER:
{\"document_graph\":{\"schema\":\"ecm-tqag.document-graph.v1\",\"nodes\":[{\"id\":\"T1\",\"kind\":\"evidence\",\"source\":\"text\",\"role\":\"premise\",\"value\":\"trích nguyên văn\"},{\"id\":\"T2\",\"kind\":\"evidence\",\"source\":\"text\",\"role\":\"mapping\",\"value\":\"trích nguyên văn\"},{\"id\":\"C1\",\"kind\":\"claim\",\"source\":\"derived\",\"role\":\"conclusion\",\"value\":\"kết luận\"}],\"edges\":[{\"id\":\"E1\",\"source\":\"T1\",\"target\":\"C1\",\"relation\":\"supports\",\"directed\":true},{\"id\":\"E2\",\"source\":\"T2\",\"target\":\"C1\",\"relation\":\"maps_to\",\"directed\":true}]},\"motif_request\":{\"motif_id\":\"premise_mapping_to_claim\",\"evidence_node_ids\":[\"T1\",\"T2\"],\"target_node_id\":\"C1\",\"image_necessary\":false}}

""" + rendered, fields


def ecm_realizer_prompt(locked_construction: dict[str, Any]) -> tuple[str, list[str]]:
    construction_json = canonical(locked_construction)
    return """Bạn là bước MCQ REALIZER của ECM. Bạn CHỈ nhận construction artifact đã được code KHÓA; bạn không truy cập evidence hay ảnh gốc.
Artifact chứa document_graph, matched_motif, restricted_program và answer_atoms do executor sinh. Không được thêm, bỏ hoặc sửa bất kỳ artifact nào.

""" + common_quality_gate() + """
YÊU CẦU REALIZATION:
- Câu hỏi phải dùng motif đã match và buộc kết hợp các evidence nodes trong matched_motif.bindings.evidence_node_ids.
- answer_atom_ids và answer_parts phải giữ nguyên ID và value của toàn bộ answer_atoms theo đúng thứ tự.
- program_trace phải giữ nguyên step_ids trong support của answer atom.
- evidence_anchor phải có đúng một mục cho mỗi evidence node đã match, dùng đúng evidence_id, source và value trong document_graph.
- Các distractor phải cùng trường khái niệm và sai do hoán đổi có chủ đích.
- Nếu artifact không đủ để tạo MCQ, chỉ trả {\"reject_reason\":\"no_graph_motif_match\"}.

LOCKED CONSTRUCTION:
""" + construction_json + """

SCHEMA REALIZER:
{\"answer_atom_ids\":[\"A1\"],\"answer_parts\":[\"giá trị atom\"],\"program_trace\":[\"S1\",\"S2\",\"S3\",\"S4\"],\"question\":\"...\",\"choices\":[\"...\",\"...\",\"...\",\"...\"],\"answer_index\":0,\"rationale\":\"...\",\"evidence_anchor\":[{\"evidence_id\":\"T1\",\"source\":\"text\",\"support\":\"đúng value của node đã khóa\"}]}

""", ["locked_construction"]


def strip_code_fence(raw: str) -> str:
    """Remove a single surrounding markdown code fence, if present.

    Providers sometimes wrap otherwise valid JSON in ``` fences despite an
    explicit JSON-only instruction. Removing the fence is a transport-level
    normalization: it deletes no JSON token and changes no semantic content.
    """
    text = raw.strip()
    if not text.startswith("```"):
        return raw
    body = text[3:]
    newline = body.find("\n")
    if newline < 0:
        return raw
    if body[:newline].strip().lower() not in {"", "json"}:
        return raw
    body = body[newline + 1:]
    end = body.rfind("```")
    return body[:end] if end >= 0 else raw


def parse_response_json(raw: str) -> tuple[dict[str, Any] | Any | None, list[str], str | None]:
    """Parse JSON, repairing only observed transport-level wrappers/escapes.

    The provider raw response remains immutable in the ledger. Each applied
    normalization is recorded so a reviewer can replay exactly what changed.
    """
    normalizations: list[str] = []
    candidate = raw
    try:
        return json.loads(candidate), normalizations, None
    except json.JSONDecodeError as first_error:
        error_name = type(first_error).__name__

    unfenced = strip_code_fence(candidate)
    if unfenced != candidate:
        candidate = unfenced
        normalizations.append("stripped_markdown_code_fence")
        try:
            return json.loads(candidate), normalizations, None
        except json.JSONDecodeError:
            pass

    unescaped = re.sub(r"(?<!\\)\\\*", "*", candidate)
    if unescaped != candidate:
        candidate = unescaped
        normalizations.append("invalid_json_escape:\\*->*")
        try:
            return json.loads(candidate), normalizations, None
        except json.JSONDecodeError:
            pass
    return None, normalizations, f"invalid_json:{error_name}"


def reject_or_object(raw: str) -> tuple[dict[str, Any] | None, list[str], str | None, str | None]:
    value, normalizations, parse_error = parse_response_json(raw)
    if parse_error:
        return None, normalizations, parse_error, None
    if not isinstance(value, dict):
        return None, normalizations, "json_not_object", None
    if value.get("reject_reason") in REJECT_REASONS and set(value) == {"reject_reason"}:
        return value, normalizations, None, "model_" + value["reject_reason"]
    return value, normalizations, None, None


def valid_anchor_list(anchors: Any, evidence: dict[str, Any], condition: str) -> list[str]:
    """Delegate to the shared contract so runner and audit cannot diverge."""
    return anchor_errors(anchors, evidence, condition)


def validate_ecm_plan(raw: str, evidence: dict[str, Any], condition: str,
                      source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse a model graph proposal and deterministically construct v5 artifacts."""
    value, normalizations, parse_error, model_reject = reject_or_object(raw)
    if parse_error:
        return {"status": "REJECTED", "reason": parse_error, "parsed_response": None,
                "raw_json_normalizations": normalizations}
    if model_reject:
        return {"status": "REJECTED", "reason": model_reject, "parsed_response": value,
                "raw_json_normalizations": normalizations}
    assert isinstance(value, dict)
    try:
        construction = build_construction(value, evidence, condition, source)
    except (ContractError, KeyError, TypeError) as exc:
        return {"status": "REJECTED", "reason": "construction_" + (str(exc) or type(exc).__name__),
                "parsed_response": value, "raw_json_normalizations": normalizations}
    replay_errors = validate_construction(construction, evidence, condition, source)
    if replay_errors:
        return {"status": "REJECTED", "reason": "construction_replay_" + replay_errors[0],
                "parsed_response": construction, "raw_json_normalizations": normalizations}
    return {"status": "PARSED", "reason": None, "parsed_response": construction,
            "raw_json_normalizations": normalizations}


def validate_response(method: str, raw: str, allowed_sources: set[str], evidence: dict[str, Any] | None = None,
                      condition: str | None = None, locked_construction: dict[str, Any] | None = None,
                      source: dict[str, Any] | None = None) -> dict[str, Any]:
    value, normalizations, parse_error, model_reject = reject_or_object(raw)
    if parse_error:
        return {"status": "REJECTED", "reason": parse_error, "parsed_response": None,
                "raw_json_normalizations": normalizations}
    if model_reject:
        return {"status": "REJECTED", "reason": model_reject, "parsed_response": value,
                "raw_json_normalizations": normalizations}
    assert isinstance(value, dict)
    required = {"question", "choices", "answer_index", "rationale", "evidence_anchor"}
    if method == "answer_first":
        required |= {"evidence_excerpt", "answer"}
    if method == "ecm":
        required |= {"answer_atom_ids", "answer_parts", "program_trace"}
    if not required.issubset(value):
        return {"status": "REJECTED", "reason": "missing:" + ",".join(sorted(required - set(value))),
                "parsed_response": value, "raw_json_normalizations": normalizations}
    choices, index = value["choices"], value["answer_index"]
    if (not isinstance(value["question"], str) or not value["question"].strip()
            or not isinstance(value["rationale"], str) or not value["rationale"].strip()
            or not isinstance(choices, list) or len(choices) != 4
            or not all(isinstance(item, str) and item.strip() for item in choices)
            or isinstance(index, bool) or not isinstance(index, int) or index not in range(4)):
        return {"status": "REJECTED", "reason": "invalid_core_schema", "parsed_response": value,
                "raw_json_normalizations": normalizations}
    if len({normalize(choice) for choice in choices}) != 4:
        return {"status": "REJECTED", "reason": "duplicate_choices", "parsed_response": value,
                "raw_json_normalizations": normalizations}
    if method != "ecm":
        if evidence is None or condition is None:
            return {"status": "REJECTED", "reason": "missing_evidence_for_anchor_check",
                    "parsed_response": value, "raw_json_normalizations": normalizations}
        found = valid_anchor_list(value["evidence_anchor"], evidence, condition)
        if found:
            return {"status": "REJECTED", "reason": "invalid_evidence_anchor:" + found[0],
                    "parsed_response": value, "raw_json_normalizations": normalizations}
    if method == "answer_first" and (not isinstance(value.get("answer"), str)
            or value["answer"] != choices[index] or not isinstance(value.get("evidence_excerpt"), str)):
        return {"status": "REJECTED", "reason": "answer_first_inconsistent", "parsed_response": value,
                "raw_json_normalizations": normalizations}
    if method == "ecm":
        if locked_construction is None or evidence is None or condition is None:
            return {"status": "REJECTED", "reason": "missing_locked_construction", "parsed_response": value,
                    "raw_json_normalizations": normalizations}
        replay_errors = validate_construction(locked_construction, evidence, condition, source)
        if replay_errors:
            return {"status": "REJECTED", "reason": "locked_construction_" + replay_errors[0],
                    "parsed_response": value, "raw_json_normalizations": normalizations}
        atoms = locked_construction["answer_atoms"]
        expected_atom_ids = [atom["id"] for atom in atoms]
        expected_parts = [atom["value"] for atom in atoms]
        expected_trace = list(dict.fromkeys(step_id for atom in atoms for step_id in atom["support"]["step_ids"]))
        if value.get("answer_atom_ids") != expected_atom_ids:
            return {"status": "REJECTED", "reason": "ecm_answer_atom_ids_changed", "parsed_response": value,
                    "raw_json_normalizations": normalizations}
        if value.get("answer_parts") != expected_parts:
            return {"status": "REJECTED", "reason": "ecm_answer_atoms_changed", "parsed_response": value,
                    "raw_json_normalizations": normalizations}
        if value.get("program_trace") != expected_trace:
            return {"status": "REJECTED", "reason": "ecm_program_trace_changed", "parsed_response": value,
                    "raw_json_normalizations": normalizations}
        if len(expected_parts) != 1:
            return {"status": "REJECTED", "reason": "ecm_expected_single_answer_atom", "parsed_response": value,
                    "raw_json_normalizations": normalizations}
        binding = atom_choice_binding(expected_parts[0], choices[index])
        if not binding["ok"]:
            return {"status": "REJECTED",
                    "reason": "ecm_selected_choice_not_bound_to_atom:" + str(binding["relation"]),
                    "parsed_response": value, "raw_json_normalizations": normalizations}
        value["atom_choice_binding"] = binding

        graph_nodes = {node["id"]: node for node in locked_construction["document_graph"]["nodes"]}
        required_ids = locked_construction["matched_motif"]["bindings"]["evidence_node_ids"]
        anchors = value.get("evidence_anchor")
        if not isinstance(anchors, list) or len(anchors) != len(required_ids):
            return {"status": "REJECTED", "reason": "invalid_ecm_evidence_anchor", "parsed_response": value,
                    "raw_json_normalizations": normalizations}
        seen: set[str] = set()
        for anchor in anchors:
            if not isinstance(anchor, dict) or set(anchor) != {"evidence_id", "source", "support"}:
                return {"status": "REJECTED", "reason": "invalid_ecm_evidence_anchor", "parsed_response": value,
                        "raw_json_normalizations": normalizations}
            evidence_id = anchor.get("evidence_id")
            if (not isinstance(evidence_id, str) or evidence_id not in required_ids or evidence_id in seen
                    or anchor.get("source") != graph_nodes[evidence_id]["source"]
                    or anchor.get("support") != graph_nodes[evidence_id]["value"]):
                return {"status": "REJECTED", "reason": "ecm_evidence_anchor_not_locked", "parsed_response": value,
                        "raw_json_normalizations": normalizations}
            seen.add(evidence_id)
        if seen != set(required_ids):
            return {"status": "REJECTED", "reason": "ecm_evidence_anchor_incomplete", "parsed_response": value,
                    "raw_json_normalizations": normalizations}
    return {"status": "PARSED", "reason": None, "parsed_response": value,
            "raw_json_normalizations": normalizations}


def post(base_url: str, payload: dict[str, Any], timeout: int, api_key: str) -> tuple[str, str | None]:
    """Return the message content and the provider finish_reason.

    ``finish_reason == "length"`` means the provider truncated the response, so
    a downstream JSON error is a budget failure rather than a malformed reply.
    """
    request = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 -- explicit user-authorized generation
        body = json.loads(response.read().decode("utf-8"))
    choice = body.get("choices", [{}])[0] if isinstance(body, dict) else {}
    content = choice.get("message", {}).get("content") if isinstance(choice, dict) else None
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if not isinstance(content, str):
        raise ValueError("invalid_provider_response")
    return content, finish_reason if isinstance(finish_reason, str) else None


def request_for(model: str, prompt: str, evidence: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    image_audit: list[dict[str, Any]] = []
    for image in evidence.get("images", []):
        part, audit = image_part(image)
        parts.append(part)
        image_audit.append(audit)
    return ({"model": model, **DECODING, "messages": [{"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": parts}]}, image_audit)


def build_plan(manifest: dict[str, Any], manifest_hash: str, experiment_id: str, model: str, chunk_ids: set[str],
               conditions: tuple[str, ...], seed: int) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for package in manifest["packages"]:
        if package["chunk_id"] not in chunk_ids or package["condition"] not in conditions:
            continue
        for method in METHODS:
            evidence, condition = package["evidence"], package["condition"]
            if method == "ecm":
                prompt, fields = ecm_planner_prompt(evidence, condition)
            else:
                prompt, fields = direct_or_answer_prompt(evidence, method, condition)
            request, image_audit = request_for(model, prompt, evidence)
            input_hash = sha256_bytes(canonical({"package": package, "method": method, "prompt_version": PROMPT_VERSION}).encode("utf-8"))
            plan.append({
                "experiment_id": experiment_id,
                "run_id": sha256_bytes(f"{experiment_id}|{input_hash}".encode("utf-8"))[:24],
                "chunk_id": package["chunk_id"], "condition": condition, "method": method, "model": model,
                "source_hash": manifest_hash, "input_hash": input_hash, "planner_or_generation_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "included_fields": fields, "image_audit": image_audit, "evidence": evidence, "request": request,
                "source_descriptor": {
                    "doc_id": package.get("doc_id"),
                    "chunk_id": package["chunk_id"],
                    "condition": condition,
                    "manifest_sha256": manifest_hash,
                },
            })
    random.Random(seed).shuffle(plan)
    for index, row in enumerate(plan, 1):
        row["run_order"] = index
    return plan


def attempt_post(base_url: str, request: dict[str, Any], timeout: int, api_key: str,
                 retries: int) -> tuple[str, str | None, int, float, str | None]:
    """Post with bounded retries; also surface the provider finish_reason."""
    raw, reason, elapsed, retry_count = "", "unknown", 0.0, 0
    for retry_count in range(retries + 1):
        started = time.perf_counter()
        try:
            raw, finish_reason = post(base_url, request, timeout, api_key)
            return raw, None, retry_count, round(time.perf_counter() - started, 3), finish_reason
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            # Deliberately retain only the exception class: HTTP exception bodies can include request details.
            reason, elapsed = type(exc).__name__, round(time.perf_counter() - started, 3)
            if retry_count < retries:
                time.sleep(min(2 ** retry_count, 4))
    return raw, reason, retry_count, elapsed, None


def execute_row(row: dict[str, Any], base_url: str, api_key: str, timeout: int, retries: int) -> dict[str, Any]:
    allowed = {"T": {"text"}, "TL_struct": {"text", "structure"}, "TLV": {"text", "structure", "image"}}[row["condition"]]
    raw, transport_error, retry_count, elapsed, finish_reason = attempt_post(
        base_url, row["request"], timeout, api_key, retries,
    )
    details: dict[str, Any] = {"raw_response": raw, "retry_count": retry_count, "elapsed_sec": elapsed,
                               "api_call_count": 1, "finish_reason": finish_reason,
                               "truncated_by_provider": finish_reason == "length"}
    if transport_error:
        details.update({"status": "ERROR", "reason": transport_error, "parsed_response": None, "raw_json_normalizations": []})
        return details
    if finish_reason == "length":
        details.update({"status": "REJECTED", "reason": "provider_truncated_response",
                        "parsed_response": None, "raw_json_normalizations": []})
        return details
    if row["method"] != "ecm":
        checked = validate_response(
            row["method"], raw, allowed, row["evidence"], row["condition"],
        )
        details.update(checked)
        return details

    source = row["source_descriptor"]
    planner_checked = validate_ecm_plan(raw, row["evidence"], row["condition"], source)
    details["planner_raw_response"] = raw
    details["planner_parsed_response"] = planner_checked["parsed_response"]
    details["planner_json_normalizations"] = planner_checked["raw_json_normalizations"]
    if planner_checked["status"] != "PARSED":
        details.update({"status": "REJECTED", "reason": "planner_" + str(planner_checked["reason"]), "parsed_response": None,
                        "raw_json_normalizations": planner_checked["raw_json_normalizations"]})
        return details

    locked_construction = planner_checked["parsed_response"]
    assert isinstance(locked_construction, dict)
    details["construction_sha256"] = locked_construction["construction_sha256"]
    realization_prompt, realization_fields = ecm_realizer_prompt(locked_construction)
    realization_request, realization_image_audit = request_for(row["model"], realization_prompt, {})
    (realization_raw, realization_error, realization_retry,
     realization_elapsed, realization_finish) = attempt_post(
        base_url, realization_request, timeout, api_key, retries,
    )
    details.update({"realizer_raw_response": realization_raw, "realizer_prompt_sha256": sha256_bytes(realization_prompt.encode("utf-8")),
                    "realizer_included_fields": realization_fields, "realizer_image_audit": realization_image_audit,
                    "realizer_retry_count": realization_retry, "realizer_elapsed_sec": realization_elapsed, "api_call_count": 2,
                    "realizer_finish_reason": realization_finish,
                    "truncated_by_provider": finish_reason == "length" or realization_finish == "length",
                    "elapsed_sec": round(elapsed + realization_elapsed, 3)})
    if realization_error:
        details.update({"status": "ERROR", "reason": "realizer_" + realization_error, "parsed_response": None, "raw_json_normalizations": []})
        return details
    if realization_finish == "length":
        details.update({"status": "REJECTED", "reason": "realizer_provider_truncated_response",
                        "parsed_response": None, "raw_json_normalizations": []})
        return details
    checked = validate_response(
        "ecm", realization_raw, allowed, row["evidence"], row["condition"], locked_construction, source,
    )
    if checked["status"] == "PARSED":
        combined = {**locked_construction, **checked["parsed_response"]}
        checked["parsed_response"] = combined
    details.update(checked)
    return details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a non-overwriting Qwen ECM-TQAG graph-program matrix or scoped pilot."
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "research/artifacts/ecm_inputs_8chunks_v3.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--condition", action="append", choices=CONDITIONS,
                        help="Repeat to select conditions; required unless --all is used.")
    parser.add_argument("--chunk-id", action="append",
                        help="Repeat to select chunks; required unless --all is used.")
    parser.add_argument("--all", action="store_true",
                        help="Run all 8 chunks and all 3 evidence conditions in the immutable manifest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the immutable manifest and print the exact plan without reading an API key or writing output.")
    args = parser.parse_args()
    if args.all and (args.condition or args.chunk_id):
        parser.error("--all cannot be combined with --condition or --chunk-id")
    if not args.all and (not args.condition or not args.chunk_id):
        parser.error("provide both --condition and --chunk-id, or use --all")
    manifest, manifest_hash = load_manifest(args.manifest)
    all_chunks = {str(package["chunk_id"]) for package in manifest["packages"]}
    selected_chunks = all_chunks if args.all else set(args.chunk_id or [])
    selected_conditions = CONDITIONS if args.all else tuple(args.condition or [])
    plan = build_plan(manifest, manifest_hash, args.experiment_id, args.model, selected_chunks, selected_conditions, args.seed)
    expected = len(selected_chunks) * len(set(selected_conditions)) * len(METHODS)
    if len(plan) != expected:
        parser.error(f"invalid plan: expected {expected}, got {len(plan)}")
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_OK", "cell_count": len(plan),
            "ecm_api_calls": sum(row["method"] == "ecm" for row in plan) * 2,
            "non_ecm_api_calls": sum(row["method"] != "ecm" for row in plan),
            "total_api_calls": sum(2 if row["method"] == "ecm" else 1 for row in plan),
            "chunk_ids": sorted(selected_chunks), "conditions": list(selected_conditions),
            "prompt_version": PROMPT_VERSION, "manifest_sha256": manifest_hash,
        }, ensure_ascii=False, indent=2))
        return
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"environment variable {args.api_key_env} is not set")
    if args.out_dir.exists():
        parser.error(f"refusing existing output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    counts: Counter[str] = Counter()
    api_call_count = 0
    with (args.out_dir / "results.jsonl").open("x", encoding="utf-8") as output, (args.out_dir / "prompt_audit.jsonl").open("x", encoding="utf-8") as audit:
        for sequence, row in enumerate(plan, 1):
            audit.write(json.dumps({key: row[key] for key in ("run_id", "planner_or_generation_prompt_sha256", "included_fields")}, ensure_ascii=False) + "\n")
            result = execute_row(row, args.base_url, api_key, args.timeout_sec, args.retries)
            api_call_count += result["api_call_count"]
            record = {key: row[key] for key in ("experiment_id", "run_id", "chunk_id", "condition", "method", "model", "source_hash", "input_hash", "planner_or_generation_prompt_sha256", "included_fields", "image_audit", "run_order")}
            record.update({"schema": "ecm-tqag.qwen37-graph-program.v5", "provider": "openrouter", "authentication": f"environment:{args.api_key_env}",
                           "created_utc": utc_now(), "seed": args.seed, "decoding": DECODING})
            record.update(result)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            counts[result["status"]] += 1
            print(f"[{sequence}/{len(plan)}] {row['method']} {row['chunk_id']} {row['condition']}: {result['status']}")
    summary = {"schema": "ecm-tqag.qwen37-graph-program-summary.v5", "status": "EXECUTED", "model": args.model, "provider": "openrouter",
               "authentication": f"environment:{args.api_key_env}", "cell_count": len(plan), "api_call_count": api_call_count,
               "status_counts": dict(counts), "seed": args.seed, "prompt_version": PROMPT_VERSION,
               "design_scope": {"chunk_ids": sorted(selected_chunks), "conditions": list(dict.fromkeys(selected_conditions)), "methods": list(METHODS)},
               "manifest_sha256": manifest_hash, "results_path": str(args.out_dir / "results.jsonl")}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if counts["PARSED"] != len(plan):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
