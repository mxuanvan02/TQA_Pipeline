#!/usr/bin/env python3
"""Run a non-overwriting evidence-chain TQA experiment with Qwen/OpenRouter.

The runner accepts an immutable 8-chunk x 3-condition evidence manifest and
writes a JSONL ledger. Direct and Answer-first use one generation request. ECM
uses two requests per cell: an evidence planner first emits a locked,
source-bound plan, then a realizer emits one MCQ constrained to that plan.

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
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("T", "TL_struct", "TLV")
METHODS = ("direct", "answer_first", "ecm")
SCHEMA = "ecm-tqag.multimodal-inputs.v3"
DEFAULT_MODEL = "qwen/qwen3.7-plus"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DECODING = {"temperature": 0, "max_tokens": 1600}
PROMPT_VERSION = "ecm-qwen37-evidencechain-v3-locked-realizer"
REJECT_REASONS = {"insufficient_evidence", "no_multi_evidence_chain", "image_not_necessary"}


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
    anchor_rule = f"evidence_anchor.source chỉ được là: {allowed_anchor_sources}.\n"
    return "Bạn tạo ĐÚNG MỘT câu hỏi trắc nghiệm tiếng Việt.\n" + common_quality_gate() + "\n" + anchor_rule + contracts[method] + "\n" + rendered, fields


def ecm_planner_prompt(evidence: dict[str, Any], condition: str) -> tuple[str, list[str]]:
    rendered, fields = evidence_text(evidence)
    image_rule = ""
    if condition == "TLV":
        image_rule = "Vì đây là TLV, plan chỉ hợp lệ nếu required_units có ít nhất một unit source=image và image_necessary=true.\n"
    return """Bạn là bước 1 của ECM (Evidence-Chain Method). Bạn CHƯA được tạo câu hỏi hay lựa chọn.
Lập một evidence plan khóa trước khi MCQ được tạo.

""" + common_quality_gate() + """
QUY TẮC ECM PLANNING:
- Trích ít nhất hai evidence unit cần thiết, có vai trò khác nhau trong chuỗi suy luận.
- Mỗi unit có `role` là một trong `premise`, `mapping`, hoặc `constraint`; required_units phải chứa ít nhất hai role khác nhau.
- ID unit phải là T1, T2... (text), S1, S2... (structure), hoặc I1, I2... (image).
- Unit text phải là trích nguyên văn từ EVIDENCE TEXT. Unit structure phải là tên mục chính xác. Unit image phải mô tả quan sát trực tiếp và nêu rõ “Ảnh n”.
- required_units phải là các ID thực sự cần cho đáp án. chain phải nêu từng bước suy luận từ các ID đó.
- Sau chain, khóa answer_atoms: các kết luận/đơn vị đáp án ngắn rút ra trực tiếp từ chain. Mỗi answer_atom phải xuất hiện nguyên văn là đầu ra `to` của ít nhất một bước chain. Chưa được tạo câu hỏi, lựa chọn, rationale hoặc distractor.
- Cấm chain chỉ đọc một nhãn, một số, hoặc một mũi tên.
- Nếu không tạo được chuỗi nhiều evidence thật sự hoặc atom có căn cứ, chỉ trả {\"reject_reason\":\"no_multi_evidence_chain\"}.
""" + image_rule + """
SCHEMA PLANNER:
{\"evidence_units\":[{\"id\":\"T1\",\"source\":\"text\",\"role\":\"premise\",\"content\":\"trích nguyên văn\"}],\"evidence_plan\":{\"target\":\"...\",\"required_units\":[\"T1\",\"T2\"],\"chain\":[{\"from\":[\"T1\"],\"to\":\"...\"},{\"from\":[\"T2\"],\"to\":\"...\"}],\"image_necessary\":false},\"answer_atoms\":[\"kết luận đã được chain chứng minh\"]}

""" + rendered, fields


def ecm_realizer_prompt(locked_plan: dict[str, Any]) -> tuple[str, list[str]]:
    plan_json = canonical(locked_plan)
    return """Bạn là bước 2 của ECM (Evidence-Chain Method). Bạn CHỈ nhận evidence plan đã bị KHÓA bởi bước 1; bạn không có quyền truy cập evidence gốc hoặc ảnh gốc.
Bạn chỉ được tạo đúng một MCQ dựa trên required_units và chain trong plan. Không thêm, bỏ, đổi ID, role, nội dung unit, chain, hay answer atom.

""" + common_quality_gate() + """
YÊU CẦU REALIZATION:
- Câu hỏi phải buộc người làm kết hợp các required_units; cấm visual lookup đơn thuần một nhãn, số hoặc mũi tên.
- Các distractor phải cùng trường khái niệm và sai do hoán đổi có chủ đích loại, chủ thể, quan hệ hoặc thứ tự.
- answer_parts phải giữ NGUYÊN danh sách answer_atoms đã khóa, không thêm, bớt, đổi thứ tự hoặc diễn đạt lại atom.
- trace phải có một mục cho mọi required_unit, dùng đúng evidence_id từ plan.
- evidence_anchor phải có đúng một mục cho mỗi required_unit, dùng đúng evidence_id, source và content đã khóa trong plan.
- Nếu locked plan không đủ để ra MCQ như trên, chỉ trả {\"reject_reason\":\"no_multi_evidence_chain\"}.

LOCKED PLAN:
""" + plan_json + """

SCHEMA REALIZER:
{\"motif\":\"...\",\"derivation\":\"...\",\"answer_parts\":[\"...\"],\"trace\":[{\"evidence_id\":\"T1\",\"supports\":\"...\"}],\"question\":\"...\",\"choices\":[\"...\",\"...\",\"...\",\"...\"],\"answer_index\":0,\"rationale\":\"...\",\"evidence_anchor\":[{\"evidence_id\":\"T1\",\"source\":\"text|structure|image\",\"support\":\"đúng content của unit đã khóa\"}]}

""", ["locked_plan"]


def parse_response_json(raw: str) -> tuple[dict[str, Any] | Any | None, list[str], str | None]:
    """Parse JSON, repairing only the observed invalid ``\\*`` escape.

    The provider raw response remains immutable in the ledger. This narrow
    transport normalization never changes semantic content.
    """
    try:
        return json.loads(raw), [], None
    except json.JSONDecodeError as original_error:
        normalized = re.sub(r"(?<!\\)\\\*", "*", raw)
        if normalized == raw:
            return None, [], f"invalid_json:{type(original_error).__name__}"
        try:
            return json.loads(normalized), ["invalid_json_escape:\\*->*"], None
        except json.JSONDecodeError:
            return None, [], f"invalid_json:{type(original_error).__name__}"


def reject_or_object(raw: str) -> tuple[dict[str, Any] | None, list[str], str | None, str | None]:
    value, normalizations, parse_error = parse_response_json(raw)
    if parse_error:
        return None, normalizations, parse_error, None
    if not isinstance(value, dict):
        return None, normalizations, "json_not_object", None
    if value.get("reject_reason") in REJECT_REASONS and set(value) == {"reject_reason"}:
        return value, normalizations, None, "model_" + value["reject_reason"]
    return value, normalizations, None, None


def valid_anchor_list(anchors: Any, allowed_sources: set[str]) -> bool:
    return (isinstance(anchors, list) and bool(anchors)
            and all(isinstance(x, dict) and set(x) == {"source", "support"}
                    and x.get("source") in allowed_sources and isinstance(x.get("support"), str) and x["support"].strip()
                    for x in anchors))


def validate_ecm_plan(raw: str, evidence: dict[str, Any], condition: str) -> dict[str, Any]:
    value, normalizations, parse_error, model_reject = reject_or_object(raw)
    if parse_error:
        return {"status": "REJECTED", "reason": parse_error, "parsed_response": None, "raw_json_normalizations": normalizations}
    if model_reject:
        return {"status": "REJECTED", "reason": model_reject, "parsed_response": value, "raw_json_normalizations": normalizations}
    assert isinstance(value, dict)
    if set(value) != {"evidence_units", "evidence_plan", "answer_atoms"}:
        return {"status": "REJECTED", "reason": "invalid_ecm_planner_top_level", "parsed_response": value, "raw_json_normalizations": normalizations}
    units, plan, answer_atoms = value["evidence_units"], value["evidence_plan"], value["answer_atoms"]
    if (not isinstance(units, list) or len(units) < 2 or not isinstance(plan, dict)
            or not isinstance(answer_atoms, list) or not answer_atoms
            or any(not isinstance(atom, str) or not atom.strip() for atom in answer_atoms)
            or len({normalize(atom) for atom in answer_atoms}) != len(answer_atoms)):
        return {"status": "REJECTED", "reason": "invalid_ecm_plan_schema", "parsed_response": value, "raw_json_normalizations": normalizations}
    allowed_sources = {"T": {"text"}, "TL_struct": {"text", "structure"}, "TLV": {"text", "structure", "image"}}[condition]
    unit_ids: set[str] = set()
    unit_sources: dict[str, str] = {}
    structure_title = str(evidence.get("document_structure", {}).get("section_title", ""))
    image_count = len(evidence.get("images", []))
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != {"id", "source", "role", "content"}:
            return {"status": "REJECTED", "reason": "invalid_ecm_evidence_unit", "parsed_response": value, "raw_json_normalizations": normalizations}
        unit_id, source, role, content = unit.get("id"), unit.get("source"), unit.get("role"), unit.get("content")
        if not isinstance(source, str):
            return {"status": "REJECTED", "reason": "invalid_ecm_evidence_unit", "parsed_response": value, "raw_json_normalizations": normalizations}
        expected_prefix = {"text": "T", "structure": "S", "image": "I"}.get(source)
        if (not isinstance(unit_id, str) or not expected_prefix or not re.fullmatch(expected_prefix + r"[1-9][0-9]*", unit_id)
                or unit_id in unit_ids or role not in {"premise", "mapping", "constraint"}
                or not isinstance(content, str) or not content.strip() or source not in allowed_sources):
            return {"status": "REJECTED", "reason": "invalid_ecm_evidence_unit", "parsed_response": value, "raw_json_normalizations": normalizations}
        if source == "text" and normalize(content) not in normalize(evidence["text"]):
            return {"status": "REJECTED", "reason": "ecm_text_unit_not_literal", "parsed_response": value, "raw_json_normalizations": normalizations}
        if source == "structure" and normalize(content) != normalize(structure_title):
            return {"status": "REJECTED", "reason": "ecm_structure_unit_not_title", "parsed_response": value, "raw_json_normalizations": normalizations}
        if source == "image":
            match = re.search(r"(?:ảnh|image)\s*(\d+)", content, flags=re.I)
            if not match or int(match.group(1)) not in range(1, image_count + 1):
                return {"status": "REJECTED", "reason": "ecm_image_unit_not_bound", "parsed_response": value, "raw_json_normalizations": normalizations}
        unit_ids.add(unit_id)
        unit_sources[unit_id] = source
    if set(plan) != {"target", "required_units", "chain", "image_necessary"}:
        return {"status": "REJECTED", "reason": "invalid_ecm_plan_schema", "parsed_response": value, "raw_json_normalizations": normalizations}
    required, chain = plan.get("required_units"), plan.get("chain")
    if (not isinstance(plan.get("target"), str) or not plan["target"].strip() or not isinstance(required, list) or len(required) < 2
            or len(set(required)) != len(required) or not all(isinstance(item, str) and item in unit_ids for item in required)
            or not isinstance(chain, list) or len(chain) < 2 or not isinstance(plan.get("image_necessary"), bool)):
        return {"status": "REJECTED", "reason": "invalid_ecm_plan_schema", "parsed_response": value, "raw_json_normalizations": normalizations}
    required_roles = {unit["role"] for unit in units if unit["id"] in required}
    if len(required_roles) < 2:
        return {"status": "REJECTED", "reason": "ecm_required_units_same_role", "parsed_response": value, "raw_json_normalizations": normalizations}
    referenced: set[str] = set()
    chain_outputs: set[str] = set()
    for step in chain:
        if not isinstance(step, dict) or set(step) != {"from", "to"} or not isinstance(step.get("from"), list) or not step["from"]:
            return {"status": "REJECTED", "reason": "invalid_ecm_chain", "parsed_response": value, "raw_json_normalizations": normalizations}
        if not isinstance(step.get("to"), str) or not step["to"].strip() or any(not isinstance(item, str) or item not in required for item in step["from"]):
            return {"status": "REJECTED", "reason": "invalid_ecm_chain", "parsed_response": value, "raw_json_normalizations": normalizations}
        referenced.update(step["from"])
        chain_outputs.add(normalize(step["to"]))
    if referenced != set(required):
        return {"status": "REJECTED", "reason": "ecm_required_units_untraced", "parsed_response": value, "raw_json_normalizations": normalizations}
    if not all(normalize(atom) in chain_outputs for atom in answer_atoms):
        return {"status": "REJECTED", "reason": "ecm_answer_atoms_not_derived", "parsed_response": value, "raw_json_normalizations": normalizations}
    if condition == "TLV" and (not plan["image_necessary"] or not any(unit_sources[item] == "image" for item in required)):
        return {"status": "REJECTED", "reason": "ecm_tlv_image_not_required", "parsed_response": value, "raw_json_normalizations": normalizations}
    if condition != "TLV" and plan["image_necessary"]:
        return {"status": "REJECTED", "reason": "ecm_non_tlv_claims_image", "parsed_response": value, "raw_json_normalizations": normalizations}
    return {"status": "PARSED", "reason": None, "parsed_response": value, "raw_json_normalizations": normalizations}


def validate_response(method: str, raw: str, allowed_sources: set[str], evidence: dict[str, Any] | None = None,
                      condition: str | None = None, locked_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    value, normalizations, parse_error, model_reject = reject_or_object(raw)
    if parse_error:
        return {"status": "REJECTED", "reason": parse_error, "parsed_response": None, "raw_json_normalizations": normalizations}
    if model_reject:
        return {"status": "REJECTED", "reason": model_reject, "parsed_response": value, "raw_json_normalizations": normalizations}
    assert isinstance(value, dict)
    required = {"question", "choices", "answer_index", "rationale", "evidence_anchor"}
    if method == "answer_first":
        required |= {"evidence_excerpt", "answer"}
    if method == "ecm":
        required |= {"motif", "derivation", "answer_parts", "trace"}
    if not required.issubset(value):
        return {"status": "REJECTED", "reason": "missing:" + ",".join(sorted(required - set(value))), "parsed_response": value, "raw_json_normalizations": normalizations}
    choices, index = value["choices"], value["answer_index"]
    if (not isinstance(value["question"], str) or not value["question"].strip() or not isinstance(value["rationale"], str)
            or not isinstance(choices, list) or len(choices) != 4 or not all(isinstance(x, str) and x.strip() for x in choices)
            or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 4):
        return {"status": "REJECTED", "reason": "invalid_core_schema", "parsed_response": value, "raw_json_normalizations": normalizations}
    if len({normalize(choice) for choice in choices}) != 4:
        return {"status": "REJECTED", "reason": "duplicate_choices", "parsed_response": value, "raw_json_normalizations": normalizations}
    if method != "ecm" and not valid_anchor_list(value["evidence_anchor"], allowed_sources):
        return {"status": "REJECTED", "reason": "invalid_evidence_anchor", "parsed_response": value, "raw_json_normalizations": normalizations}
    if method == "answer_first" and (not isinstance(value.get("answer"), str) or value["answer"] != choices[index] or not isinstance(value.get("evidence_excerpt"), str)):
        return {"status": "REJECTED", "reason": "answer_first_inconsistent", "parsed_response": value, "raw_json_normalizations": normalizations}
    if method == "ecm":
        trace = value.get("trace")
        if (not isinstance(value.get("motif"), str) or not isinstance(value.get("derivation"), str)
                or not isinstance(value.get("answer_parts"), list) or not value["answer_parts"]
                or not isinstance(trace, list) or not trace or locked_plan is None or evidence is None or condition is None):
            return {"status": "REJECTED", "reason": "invalid_ecm_schema", "parsed_response": value, "raw_json_normalizations": normalizations}
        if value["answer_parts"] != locked_plan.get("answer_atoms"):
            return {"status": "REJECTED", "reason": "ecm_answer_atoms_changed", "parsed_response": value, "raw_json_normalizations": normalizations}
        units = locked_plan["evidence_units"]
        unit_by_id = {item["id"]: item for item in units}
        required_units = set(locked_plan["evidence_plan"]["required_units"])
        anchors = value.get("evidence_anchor")
        if not isinstance(anchors, list) or len(anchors) != len(required_units):
            return {"status": "REJECTED", "reason": "invalid_ecm_evidence_anchor", "parsed_response": value, "raw_json_normalizations": normalizations}
        anchored_ids: set[str] = set()
        for anchor in anchors:
            if not isinstance(anchor, dict) or set(anchor) != {"evidence_id", "source", "support"}:
                return {"status": "REJECTED", "reason": "invalid_ecm_evidence_anchor", "parsed_response": value, "raw_json_normalizations": normalizations}
            evidence_id = anchor.get("evidence_id")
            if (not isinstance(evidence_id, str) or evidence_id not in required_units or evidence_id in anchored_ids
                    or anchor.get("source") != unit_by_id[evidence_id]["source"]
                    or anchor.get("support") != unit_by_id[evidence_id]["content"]):
                return {"status": "REJECTED", "reason": "ecm_evidence_anchor_not_locked", "parsed_response": value, "raw_json_normalizations": normalizations}
            anchored_ids.add(evidence_id)
        if anchored_ids != required_units:
            return {"status": "REJECTED", "reason": "ecm_evidence_anchor_incomplete", "parsed_response": value, "raw_json_normalizations": normalizations}
        seen_trace: set[str] = set()
        for item in trace:
            if not isinstance(item, dict) or set(item) != {"evidence_id", "supports"}:
                return {"status": "REJECTED", "reason": "invalid_ecm_trace", "parsed_response": value, "raw_json_normalizations": normalizations}
            evidence_id, support = item.get("evidence_id"), item.get("supports")
            if evidence_id not in required_units or evidence_id in seen_trace or not isinstance(support, str) or not support.strip():
                return {"status": "REJECTED", "reason": "invalid_ecm_trace", "parsed_response": value, "raw_json_normalizations": normalizations}
            seen_trace.add(evidence_id)
        if seen_trace != required_units:
            return {"status": "REJECTED", "reason": "ecm_trace_incomplete", "parsed_response": value, "raw_json_normalizations": normalizations}
        if condition == "TLV" and not any(unit_by_id[item]["source"] == "image" for item in required_units):
            return {"status": "REJECTED", "reason": "ecm_tlv_image_not_traced", "parsed_response": value, "raw_json_normalizations": normalizations}
    return {"status": "PARSED", "reason": None, "parsed_response": value, "raw_json_normalizations": normalizations}


def post(base_url: str, payload: dict[str, Any], timeout: int, api_key: str) -> str:
    request = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 -- explicit user-authorized generation
        body = json.loads(response.read().decode("utf-8"))
    content = body.get("choices", [{}])[0].get("message", {}).get("content") if isinstance(body, dict) else None
    if not isinstance(content, str):
        raise ValueError("invalid_provider_response")
    return content


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
            })
    random.Random(seed).shuffle(plan)
    for index, row in enumerate(plan, 1):
        row["run_order"] = index
    return plan


def attempt_post(base_url: str, request: dict[str, Any], timeout: int, api_key: str, retries: int) -> tuple[str, str | None, int, float]:
    raw, reason, elapsed, retry_count = "", "unknown", 0.0, 0
    for retry_count in range(retries + 1):
        started = time.perf_counter()
        try:
            raw = post(base_url, request, timeout, api_key)
            return raw, None, retry_count, round(time.perf_counter() - started, 3)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            # Deliberately retain only the exception class: HTTP exception bodies can include request details.
            reason, elapsed = type(exc).__name__, round(time.perf_counter() - started, 3)
            if retry_count < retries:
                time.sleep(min(2 ** retry_count, 4))
    return raw, reason, retry_count, elapsed


def execute_row(row: dict[str, Any], base_url: str, api_key: str, timeout: int, retries: int) -> dict[str, Any]:
    allowed = {"T": {"text"}, "TL_struct": {"text", "structure"}, "TLV": {"text", "structure", "image"}}[row["condition"]]
    raw, transport_error, retry_count, elapsed = attempt_post(base_url, row["request"], timeout, api_key, retries)
    details: dict[str, Any] = {"raw_response": raw, "retry_count": retry_count, "elapsed_sec": elapsed, "api_call_count": 1}
    if transport_error:
        details.update({"status": "ERROR", "reason": transport_error, "parsed_response": None, "raw_json_normalizations": []})
        return details
    if row["method"] != "ecm":
        checked = validate_response(row["method"], raw, allowed)
        details.update(checked)
        return details

    planner_checked = validate_ecm_plan(raw, row["evidence"], row["condition"])
    details["planner_raw_response"] = raw
    details["planner_parsed_response"] = planner_checked["parsed_response"]
    details["planner_json_normalizations"] = planner_checked["raw_json_normalizations"]
    if planner_checked["status"] != "PARSED":
        details.update({"status": "REJECTED", "reason": "planner_" + str(planner_checked["reason"]), "parsed_response": None,
                        "raw_json_normalizations": planner_checked["raw_json_normalizations"]})
        return details

    locked_plan = planner_checked["parsed_response"]
    assert isinstance(locked_plan, dict)
    realization_prompt, realization_fields = ecm_realizer_prompt(locked_plan)
    realization_request, realization_image_audit = request_for(row["model"], realization_prompt, {})
    realization_raw, realization_error, realization_retry, realization_elapsed = attempt_post(base_url, realization_request, timeout, api_key, retries)
    details.update({"realizer_raw_response": realization_raw, "realizer_prompt_sha256": sha256_bytes(realization_prompt.encode("utf-8")),
                    "realizer_included_fields": realization_fields, "realizer_image_audit": realization_image_audit,
                    "realizer_retry_count": realization_retry, "realizer_elapsed_sec": realization_elapsed, "api_call_count": 2,
                    "elapsed_sec": round(elapsed + realization_elapsed, 3)})
    if realization_error:
        details.update({"status": "ERROR", "reason": "realizer_" + realization_error, "parsed_response": None, "raw_json_normalizations": []})
        return details
    checked = validate_response("ecm", realization_raw, allowed, row["evidence"], row["condition"], locked_plan)
    if checked["status"] == "PARSED":
        combined = {**locked_plan, **checked["parsed_response"]}
        checked["parsed_response"] = combined
    details.update(checked)
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a non-overwriting Qwen ECM-TQAG matrix or scoped pilot.")
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
            record.update({"schema": "ecm-tqag.qwen37-evidencechain.v3", "provider": "openrouter", "authentication": f"environment:{args.api_key_env}",
                           "created_utc": utc_now(), "seed": args.seed, "decoding": DECODING})
            record.update(result)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            counts[result["status"]] += 1
            print(f"[{sequence}/{len(plan)}] {row['method']} {row['chunk_id']} {row['condition']}: {result['status']}")
    summary = {"schema": "ecm-tqag.qwen37-evidencechain-summary.v3", "status": "EXECUTED", "model": args.model, "provider": "openrouter",
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
