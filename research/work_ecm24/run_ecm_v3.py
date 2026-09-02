#!/usr/bin/env python3
"""ECM-v3 runner: the manuscript spine, executed in two sealed stages.

    OCR chunk -> G -> G_m -> z -> (A, Gamma) -> q -> C

Stage 1 (PLANNER)  sees the evidence package and proposes motif + typed steps.
                   Deterministic code matches the motif against the closed
                   catalog, compiles a restricted program z, executes it against
                   the evidence, and derives answer atoms A with trace Gamma.
Stage 2 (REALIZER) sees ONLY the sealed construction (atoms, short anchor
                   excerpts, visual node ids). It never receives the chunk, so
                   it cannot copy the source or re-choose the answer.

Role/endpoint binding is enforced, not assumed. Generation MUST run on
OpenRouter; the runner refuses to start otherwise. This is a hard guard against
the failure that produced 48/48 HTTP 500 on a local proxy that had no route for
the generator model.

Dry-run by default; --execute is required for any HTTP call.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ecm_v3_core as core  # noqa: E402
import run_ecm24_matrix as v1  # noqa: E402  (frozen manifest loader / image packer)

METHOD = "ecm_v3"
PROMPT_VERSION = "ecm_v3.p1"

# ---- role -> endpoint binding (hard guard) --------------------------------
GEN_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
GEN_MODEL = "qwen/qwen3-vl-32b-instruct"
DECODING = {"temperature": 0, "max_tokens": 1200}


def assert_generation_endpoint(base_url: str) -> None:
    """Refuse to generate anywhere except OpenRouter.

    A silent endpoint swap is the single most expensive failure mode observed
    in this project: the model string is accepted, the transport returns 500 or
    routes to a different family, and the whole matrix is invalidated.
    """
    if not base_url.startswith("https://openrouter.ai/"):
        raise ValueError(
            f"BLOCKED_ROLE_ENDPOINT:generation_must_use_openrouter:got={base_url}")


# ---- prompts ---------------------------------------------------------------
def catalog_spec_text(condition: str | None = None) -> str:
    """Render the closed motif catalog, filtered by what the condition allows.

    Cross-modal motifs structurally require an image (compile_program refuses
    them without a visual anchor, and require_visual_binding refuses them
    outside TLV). Offering them under T / TL_struct would therefore guarantee a
    compile rejection, and offering text-only motifs under TLV is exactly what
    let TLV degenerate into "TL_struct + a label". So:

        TLV            -> cross-modal motifs ONLY
        T, TL_struct   -> text-only motifs ONLY
        None (legacy)  -> everything, for dry-run/inspection
    """
    if condition == "TLV":
        allowed = core.CROSS_MODAL_MOTIFS
    elif condition in ("T", "TL_struct"):
        allowed = set(core.MOTIF_CATALOG) - core.CROSS_MODAL_MOTIFS
    else:
        allowed = set(core.MOTIF_CATALOG)

    lines = []
    for name, spec in core.MOTIF_CATALOG.items():
        if name not in allowed:
            continue
        parts = []
        for s in spec["steps"]:
            if s["needs"]:
                kind = f'anchor:{s.get("modality", "text")}'
            else:
                kind = "derived"
            parts.append(f'{s["op"]}({kind})')
        desc = "; ".join(s["desc"] for s in spec["steps"])
        lines.append(f'- "{name}": {" -> ".join(parts)}\n    {desc}')
    return "\n".join(lines)


def planner_prompt(evidence: dict[str, Any], condition: str) -> tuple[str, list[str]]:
    """Stage-1 prompt. The planner sees the evidence and proposes a construction."""
    fields = ["evidence.text"]
    text = evidence["text"]

    structure_text = ""
    structure = evidence.get("document_structure")
    if isinstance(structure, dict):
        fields += [
            "evidence.document_structure.section_title",
            "evidence.document_structure.header_level",
            "evidence.document_structure.declared_image_order",
        ]
        structure_text = (
            "\nCấu trúc tài liệu:\n"
            f"- Tiêu đề mục: {structure['section_title']}\n"
            f"- Cấp mục: {structure['header_level']}\n"
            f"- Thứ tự ảnh đã khai báo: {structure['declared_image_order']}\n")

    visual_text = ""
    n_images = len(evidence.get("images") or [])
    if n_images:
        fields.append("evidence.images[pixels_attached]")
        # Under TLV the motif itself requires a visual anchor, so the prompt no
        # longer *invites* the planner to tag a step -- it states the contract
        # the compiler will enforce, including the bbox that
        # verify_visual_anchors() checks. Demanding a localising bbox is what
        # stops "visual_node: 1" from being an unfalsifiable claim.
        visual_text = (
            f"\nCó {n_images} ảnh trang gốc được đính kèm, đánh số 1..{n_images}.\n"
            "Bước có `anchor:visual` BẮT BUỘC phải:\n"
            '  - khai `"visual_node": <số thứ tự ảnh, 1..%d>`;\n' % n_images +
            '  - khai `"bbox": [x0,y0,x1,y1]` là vùng CỤ THỂ trên ảnh đó chứa dữ '
            "kiện, toạ độ chuẩn hoá trong khoảng 0..1 (x0<x1, y0<y1). KHÔNG được "
            "lấy cả trang (diện tích phải nhỏ hơn 80% ảnh);\n"
            '  - `anchor` = mô tả NGẮN vùng đó (nhãn/ô/mũi tên/số liệu nhìn thấy), '
            '`result` = dữ kiện ĐỌC ĐƯỢC từ vùng đó, tối thiểu 4 từ.\n'
            '  - TUYỆT ĐỐI KHÔNG viết chung chung như "theo hình", "xem sơ đồ".\n'
            "Nếu ảnh thực sự không chứa dữ kiện nào dùng được cho suy luận, hãy trả "
            '{"abstain":"no_visual_evidence"} thay vì bịa.\n')

    prompt = (
        "Bạn là bộ LẬP KẾ HOẠCH (planner). Nhiệm vụ: đề xuất một CẤU TRÚC SUY LUẬN "
        "từ bằng chứng dưới đây. TUYỆT ĐỐI KHÔNG viết câu hỏi ở bước này.\n\n"
        "Bằng chứng văn bản:\n" + text + structure_text + visual_text +
        "\n=== QUY TRÌNH BẮT BUỘC ===\n"
        "Bước 1. Chọn MỘT motif từ danh mục ĐÓNG sau (dùng đúng tên):\n"
        + catalog_spec_text(condition) +
        "\n\nBước 2. Viết `steps` đúng số bước và đúng thứ tự `op` của motif đã chọn.\n"
        "  - Bước có (anchor): PHẢI có `anchor` = đoạn TRÍCH NGUYÊN VĂN từ bằng chứng, "
        "NGẮN (10-25 từ, chỉ phần cốt lõi mang dữ kiện) và `result` = dữ kiện rút ra "
        "từ chính đoạn đó, viết NGẮN GỌN (1 câu).\n"
        "  - Bước (derived): KHÔNG cần `anchor`, chỉ cần `result` = kết quả suy luận, "
        "viết NGẮN GỌN (1 câu).\n"
        + ("  - Bước `anchor:text` PHẢI trích từ văn bản; bước `anchor:visual` PHẢI "
           "đọc từ ẢNH (kèm visual_node + bbox) và KHÔNG được trích văn bản.\n"
           if condition == "TLV" else
           "  - HAI anchor PHẢI trích từ HAI CÂU KHÁC NHAU. Nếu chỉ lấy được từ một câu, "
           "hãy chọn motif khác.\n") +
        "  - TUYỆT ĐỐI KHÔNG thêm chữ giải thích, ghi chú, hay bình luận nào ngoài "
        "đối tượng JSON. Dừng ngay sau dấu `}` cuối cùng.\n"
        "\nTrả đúng một đối tượng JSON, không markdown, không giải thích thêm:\n"
        + ('{"motif":"<tên trong danh mục>",'
           '"steps":[{"op":"locate","visual_node":1,"bbox":[0.12,0.34,0.48,0.52],'
           '"anchor":"<mô tả vùng nhìn thấy trên ảnh>","result":"<dữ kiện đọc từ vùng đó>"},'
           '{"op":"locate","anchor":"<trích nguyên văn từ VĂN BẢN>","result":"..."},'
           '{"op":"<op cuối của motif>","result":"..."}]}\n'
           if condition == "TLV" else
           '{"motif":"<tên trong danh mục>",'
           '"steps":[{"op":"locate","anchor":"<trích nguyên văn>","result":"..."},'
           '{"op":"locate","anchor":"<trích nguyên văn câu KHÁC>","result":"..."},'
           '{"op":"<op cuối của motif>","result":"..."}]}\n'))
    return prompt, fields


def realizer_prompt(sealed: dict[str, Any]) -> str:
    """Stage-2 prompt. The realizer sees ONLY the sealed construction."""
    lines = ["Bạn là bộ DIỄN ĐẠT (realizer). Bạn KHÔNG được xem tài liệu gốc.",
             "Bạn chỉ nhận cấu trúc suy luận đã được KHOÁ dưới đây và phải giữ nguyên nó.",
             "", f"Motif: {sealed['motif']}", "", "Các dữ kiện đã chốt (answer atoms):"]
    # seal_construction() emits atoms as {atom_id, op, value} and anchors as
    # {atom_id, excerpt, sentence_index}. An atom is "derived" exactly when no
    # anchor references it, so the tag is computed here rather than read from a
    # key the sealed payload never carried (the earlier atom["kind"] lookup
    # crashed the pilot with KeyError).
    anchored_ids = {a.get("atom_id") for a in sealed.get("anchors", [])}
    for atom in sealed["atoms"]:
        tag = "trích" if atom["atom_id"] in anchored_ids else "suy ra"
        lines.append(f"  [{atom['atom_id']}] ({tag}, op={atom['op']}): {atom['value']}")
    lines.append("")
    lines.append("Trích dẫn ngắn làm neo (chỉ để diễn đạt, KHÔNG được chép nguyên văn):")
    for anchor in sealed["anchors"]:
        lines.append(f"  - ({anchor['atom_id']}) \"{anchor['excerpt']}\"")
    if sealed.get("visual_nodes"):
        lines.append("")
        lines.append(f"Bước dựa vào ảnh: {sealed['visual_nodes']}")
    # Visual observations are part of the locked construction under TLV: the
    # realizer must be able to phrase a question that actually refers to what
    # was read off the image. Carrying the observation + its region keeps the
    # question genuinely multimodal WITHOUT unsealing the source prose.
    for vis in sealed.get("visual_observations", []) or []:
        lines.append(
            f"  - ({vis['atom_id']}) ảnh {vis['visual_node']}, vùng {vis['bbox']}: "
            f"{vis['observation']}")
    lines += [
        "",
        "=== YÊU CẦU ===",
        f"1. Viết MỘT câu hỏi trắc nghiệm tiếng Việt buộc người trả lời phải đi qua đủ "
        f"{len(sealed['atoms'])} bước trên mới trả lời được. Không hỏi lại một dữ kiện đơn lẻ.",
        "2. CẤM sao chép nguyên văn: câu hỏi KHÔNG được chứa 8 từ liên tiếp trùng với "
        "các trích dẫn neo ở trên. Hãy diễn đạt lại bằng lời khác.",
        "3. Đáp án đúng PHẢI là dữ kiện của atom cuối cùng. Không được đổi đáp án.",
        "4. Ba phương án nhiễu = kết quả của việc làm SAI MỘT bước (dùng sai neo, bỏ bước "
        "suy luận, đảo chiều so sánh). Không dùng phương án sai hiển nhiên.",
        "",
        "Trả đúng một đối tượng JSON, không markdown:",
        '{"question":"...","choices":["...","...","...","..."],"answer_index":0,'
        '"rationale":"...","distractor_faults":["...","...","..."]}',
    ]
    return "\n".join(lines)


# ---- transport -------------------------------------------------------------
def post_json(url: str, payload: dict[str, Any], timeout_sec: int,
              api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # nosec B310
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("response_not_object")
    return value


def extract_first_json_object(text: str) -> str | None:
    """Return the first complete top-level {...} object in `text`, else None.

    Needed because a pilot row emitted a fully valid JSON object followed by a
    prose epilogue ("...}]}\\n\\nNote: Bước cuối không yêu cầu anchor..."), which
    made json.loads raise "Extra data" and got misfiled as invalid_json even
    though the payload was complete and usable.

    Bracket depth is tracked outside string literals only, so Vietnamese prose
    containing braces cannot terminate the object early. This is a TRANSPORT
    repair: it recovers the bytes the model sent and changes no contract check.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def repair_missing_closers(text: str) -> str | None:
    """Insert a closing bracket the model forgot, at the position it forgot it.

    full_v3d found 11/48 rows where the model wrote a 3-step `steps` array
    whose LAST step has no `anchor` key (derived steps carry none), then
    closed with `}}` instead of `]}}`: the array-closer `]` for `steps` was
    dropped mid-string, not at the end. An earlier version of this function
    only ever *appended* brackets, which is wrong for a mid-string drop: on
    `...}}`  with stack ["}","]"] at the final `}`, appending produces an
    extra unmatched `}` and reparsing fails 0/11 times.

    The correct repair: walk the bracket stack: whenever a closer doesn't
    match the top of the stack, that closer actually belongs to the bracket
    ONE LEVEL UP, so the missing bracket (the stack top) must be inserted
    right before it, then the same closer is reprocessed against the new
    top. This is a pure transport patch -- it only inserts a byte-for-byte
    copy of a bracket the model's own steps array required, and changes no
    field value, so a contract violation the model actually wrote is still
    rejected downstream (guard_plan_v3 / guard_realized_v3 run unchanged on
    the repaired dict). Truncated streams (nothing left to close because the
    model just stopped) still have a non-empty stack with NO mismatch ever
    observed, so they fall through unrepaired here and stay flagged by
    is_truncated() -- confirmed against 11/11 real fixtures + 6 negative
    controls (prose_only, empty, unterminated_string, valid_plain,
    valid_nested, truncated_array all correctly return None).
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    out: list[str] = []
    mutated = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            stack_len_before = None
            out.append(ch)
            continue
        if ch in pairs:
            stack.append(pairs[ch])
            out.append(ch)
        elif ch in ('}', ']'):
            if stack and stack[-1] == ch:
                stack.pop()
                out.append(ch)
            elif stack:
                # `ch` belongs one level up: the closer for the CURRENT top
                # was skipped by the model. Insert it here, then let `ch`
                # close the level it actually matches.
                out.append(stack.pop())
                mutated = True
                if stack and stack[-1] == ch:
                    stack.pop()
                out.append(ch)
            else:
                out.append(ch)
        else:
            out.append(ch)
    if not mutated:
        return None
    candidate = "".join(out) + "".join(reversed(stack))
    try:
        json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        return None
    return candidate


def parse_model_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Transport-level parse only. Returns (parsed, reason_if_failed)."""
    candidate, _ = v1.strip_code_fence(raw)
    parsed: Any = None
    try:
        parsed = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        repaired, n = v1.repair_invalid_escapes(candidate)
        if n:
            try:
                parsed = json.loads(repaired)
            except (TypeError, json.JSONDecodeError):
                parsed = None
        if parsed is None:
            # Last resort 1: the object may be complete but wrapped in prose.
            sliced = extract_first_json_object(candidate)
            if sliced is not None:
                try:
                    parsed = json.loads(sliced)
                except (TypeError, json.JSONDecodeError):
                    parsed = None
        if parsed is None:
            # Last resort 2: the model forgot one or more closing brackets.
            fixed = repair_missing_closers(candidate)
            if fixed is None:
                return None, "invalid_json:JSONDecodeError"
            try:
                parsed = json.loads(fixed)
            except (TypeError, json.JSONDecodeError) as exc:
                return None, f"invalid_json:{type(exc).__name__}"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    return parsed, None


def is_truncated(raw: str) -> bool:
    """Detect a stream that was cut mid-flight (transport failure).

    Two signatures, both observed in pilots:
      (a) an opening ``` fence with no closing fence;
      (b) unbalanced JSON brackets -- e.g. the pilot emitted `[` once and `]`
          zero times because the `steps` array never closed. Case (b) carries
          no fence at all, so the fence-only check silently missed it and the
          row was misfiled as a contract rejection.

    Brackets inside string literals are ignored so ordinary Vietnamese prose
    containing "[" or "{" cannot fake a truncation.
    """
    text = raw.strip()
    if text.startswith("```") and text.count("```") < 2:
        return True

    depth_curly = depth_square = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
    return in_string or depth_curly != 0 or depth_square != 0


def call_with_retry(url: str, payload: dict[str, Any], timeout_sec: int,
                    retries: int, api_key: str) -> dict[str, Any]:
    """Retry ONLY transport failures. Contract rejections are never re-rolled."""
    reason = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            response = post_json(url, payload, timeout_sec, api_key)
            content = response.get("choices", [{}])[0].get("message", {}).get("content")
            usage = response.get("usage")
            if not isinstance(content, str):
                return {"ok": False, "reason": "invalid_provider_response", "raw": "",
                        "usage": usage, "retry_count": attempt,
                        "elapsed_sec": round(time.perf_counter() - start, 3)}
            if is_truncated(content) and attempt < retries:
                reason = "transport:truncated_stream"
                time.sleep(min(2 ** attempt, 4))
                continue
            return {"ok": True, "reason": None, "raw": content, "usage": usage,
                    "retry_count": attempt,
                    "elapsed_sec": round(time.perf_counter() - start, 3)}
        except urllib.error.HTTPError as exc:
            reason = f"http_{exc.code}"
            transient = exc.code == 429 or 500 <= exc.code < 600
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, ValueError) as exc:
            reason, transient = f"transport:{type(exc).__name__}", True
        if not transient or attempt == retries:
            break
        time.sleep(min(2 ** attempt, 4))
    return {"ok": False, "reason": reason, "raw": "", "usage": None,
            "retry_count": retries, "elapsed_sec": None}


# ---- plan ------------------------------------------------------------------
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
        prompt, fields = planner_prompt(evidence, package["condition"])
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_audit: list[dict[str, Any]] = []
        for image in evidence.get("images", []):
            part, audit = v1.image_part(image)
            parts.append(part)
            image_audit.append(audit)
        input_hash = v1.sha256_bytes(
            v1.canonical({"package": package, "method": METHOD}).encode())
        rows.append({
            "experiment_id": experiment_id,
            "run_id": v1.sha256_bytes(f"{experiment_id}|{input_hash}".encode())[:24],
            "chunk_id": package["chunk_id"], "doc_id": package.get("doc_id"),
            "condition": package["condition"], "method": METHOD,
            "source_hash": manifest_hash, "input_hash": input_hash, "model": model,
            "prompt_version": PROMPT_VERSION, "decoding": DECODING,
            "planner_prompt_sha256": v1.sha256_bytes(prompt.encode()),
            "included_fields": fields, "image_audit": image_audit,
            "planner_parts": parts, "package": package,
        })
    random.Random(seed).shuffle(rows)
    for order, row in enumerate(rows, 1):
        row["run_order"] = order
    return rows, manifest


# ---- one cell, two stages --------------------------------------------------
def execute_cell(row: dict[str, Any], base_url: str, model: str, timeout_sec: int,
                 retries: int, api_key: str, idf_table, idf_cut) -> dict[str, Any]:
    package = row["package"]
    evidence_text = package["evidence"]["text"]
    condition = row["condition"]

    # ---------- stage 1: planner
    req1 = {"model": model, **DECODING,
            "messages": [{"role": "system", "content": "Return valid JSON only."},
                         {"role": "user", "content": row["planner_parts"]}]}
    call1 = call_with_retry(base_url, req1, timeout_sec, retries, api_key)
    stage = {"stage1_usage": call1["usage"], "stage1_retry": call1["retry_count"],
             "stage1_elapsed": call1["elapsed_sec"]}
    if not call1["ok"]:
        return {"status": "ERROR", "reason": f"stage1:{call1['reason']}",
                "stage1_raw": call1["raw"], "stage2_raw": None,
                "plan": None, "sealed": None, "parsed_response": None, **stage}

    parsed1, why = parse_model_json(call1["raw"])
    if parsed1 is None:
        return {"status": "REJECTED", "reason": f"stage1:{why}",
                "stage1_raw": call1["raw"], "stage2_raw": None,
                "plan": None, "sealed": None, "parsed_response": None, **stage}
    if isinstance(parsed1.get("abstain"), str):
        return {"status": "ABSTAINED", "reason": f"stage1:abstain:{parsed1['abstain']}",
                "stage1_raw": call1["raw"], "stage2_raw": None,
                "plan": None, "sealed": None, "parsed_response": None, **stage}

    declared_images = (package.get("evidence", {}) or {}).get("images")
    # require_cross_modal: under TLV the motif must STRUCTURALLY need the image
    # (a cross-modal template), not merely carry a self-declared visual_node on
    # a text-only construction. This is the switch that makes "multimodal" a
    # mechanism instead of a label.
    plan = core.guard_plan_v3(parsed1, evidence_text, condition,
                              declared_images=declared_images,
                              idf_cut=idf_cut, idf_table=idf_table,
                              require_cross_modal=(condition == "TLV"))
    if plan["status"] != "PLAN_OK":
        return {"status": "REJECTED", "reason": f"stage1:{plan['reason']}",
                "stage1_raw": call1["raw"], "stage2_raw": None,
                "plan": plan, "sealed": None, "parsed_response": None, **stage}

    # ---------- seal: the realizer must not see the chunk
    image_nodes = [f"img{i+1}" for i in range(len(declared_images or []))] \
        if condition == "TLV" else []
    sealed = core.seal_construction(plan["program"], plan["trace"],
                                    condition, image_nodes,
                                    evidence_text=evidence_text)
    cap = core.excerpt_word_budget(
        evidence_text, sum(1 for a in sealed.get("anchors", [])))
    blind = core.assert_realizer_blind(sealed, evidence_text, max_excerpt_words=cap)
    if not blind["ok"]:
        return {"status": "REJECTED", "reason": f"seal:{';'.join(blind['problems'])}",
                "stage1_raw": call1["raw"], "stage2_raw": None,
                "plan": plan, "sealed": sealed, "parsed_response": None,
                "blindness": blind, **stage}

    # ---------- stage 2: realizer (text only; no chunk, no page image)
    req2 = {"model": model, **DECODING,
            "messages": [{"role": "system", "content": "Return valid JSON only."},
                         {"role": "user", "content": realizer_prompt(sealed)}]}
    call2 = call_with_retry(base_url, req2, timeout_sec, retries, api_key)
    stage.update({"stage2_usage": call2["usage"], "stage2_retry": call2["retry_count"],
                  "stage2_elapsed": call2["elapsed_sec"]})
    if not call2["ok"]:
        return {"status": "ERROR", "reason": f"stage2:{call2['reason']}",
                "stage1_raw": call1["raw"], "stage2_raw": call2["raw"],
                "plan": plan, "sealed": sealed, "parsed_response": None,
                "blindness": blind, **stage}

    parsed2, why2 = parse_model_json(call2["raw"])
    if parsed2 is None:
        return {"status": "REJECTED", "reason": f"stage2:{why2}",
                "stage1_raw": call1["raw"], "stage2_raw": call2["raw"],
                "plan": plan, "sealed": sealed, "parsed_response": None,
                "blindness": blind, **stage}

    check = core.guard_realized_v3(parsed2, sealed, evidence_text)
    status = "PARSED" if check["status"] in ("ACCEPTED", "PARSED") else "REJECTED"
    return {"status": status,
            "reason": None if status == "PARSED" else f"stage2:{check['reason']}",
            "stage1_raw": call1["raw"], "stage2_raw": call2["raw"],
            "plan": plan, "sealed": sealed,
            "parsed_response": parsed2 if status == "PARSED" else parsed2,
            "check": check, "blindness": blind, **stage}


# ---- driver ----------------------------------------------------------------
def run(manifest_path: Path, output_dir: Path, experiment_id: str, seed: int, *,
        execute: bool, base_url: str, model: str, timeout_sec: int, retries: int,
        conditions, chunk_ids, api_key: str | None, api_key_file: str | None,
        ) -> dict[str, Any]:
    assert_generation_endpoint(base_url)
    plan_rows, manifest = build_plan(manifest_path, experiment_id, seed, model,
                                     conditions, chunk_ids)
    if not plan_rows:
        raise ValueError("filters selected no design cells")

    texts = sorted({p["evidence"]["text"] for p in manifest["packages"]})
    idf_table, idf_cut = core.build_idf(texts)

    v1.prepare_output_dir(output_dir)
    results_path = output_dir / "results.jsonl"
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    tok_in = tok_out = 0

    with results_path.open("x", encoding="utf-8") as results:
        for row in plan_rows:
            if not execute:
                outcome = {"status": "PLANNED_DRY_RUN", "reason": None,
                           "stage1_raw": None, "stage2_raw": None, "plan": None,
                           "sealed": None, "parsed_response": None}
            else:
                outcome = execute_cell(row, base_url, model, timeout_sec, retries,
                                       api_key or "", idf_table, idf_cut)
            counts[outcome["status"]] = counts.get(outcome["status"], 0) + 1
            if outcome.get("reason"):
                fam = ":".join(str(outcome["reason"]).split(":")[:2])
                reasons[fam] = reasons.get(fam, 0) + 1
            for key in ("stage1_usage", "stage2_usage"):
                usage = outcome.get(key)
                if isinstance(usage, dict):
                    tok_in += usage.get("prompt_tokens") or 0
                    tok_out += usage.get("completion_tokens") or 0

            record = {k: row[k] for k in (
                "experiment_id", "run_id", "chunk_id", "doc_id", "source_hash",
                "condition", "method", "model", "prompt_version", "decoding",
                "input_hash", "planner_prompt_sha256", "run_order", "image_audit")}
            record.update({"base_url": base_url, "seed": seed,
                           "created_utc": v1.utc_now(), **outcome})
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()

    summary = {
        "status": "EXECUTED" if execute else "PLANNED_DRY_RUN_NO_API_CALLS",
        "api_call_count": (2 * len(plan_rows)) if execute else 0,
        "cell_count": len(plan_rows), "status_counts": counts,
        "reject_reason_families": reasons,
        "tokens": {"prompt": tok_in, "completion": tok_out},
        "seed": seed, "model": model, "method": METHOD,
        "prompt_version": PROMPT_VERSION, "base_url": base_url,
        "motif_catalog": sorted(core.MOTIF_CATALOG),
        "guard": {"min_distinct_anchor_sentences": 2,
                  "max_verbatim_run": core.MAX_VERBATIM_RUN,
                  "max_excerpt_leak_ratio": 0.35,
                  "idf_cut": round(idf_cut, 4)},
        "two_stage": {"stage1": "planner_sees_evidence",
                      "stage2": "realizer_sees_sealed_construction_only"},
        "design_scope": {
            "conditions": list(v1.CONDITIONS if conditions is None else conditions),
            "chunk_ids": sorted(chunk_ids) if chunk_ids is not None else "ALL_SCREENED_CHUNKS",
            "methods": [METHOD]},
        "authentication": f"file:{api_key_file}" if api_key_file else "none",
        "manifest_sha256": plan_rows[0]["source_hash"],
        "results_path": str(results_path),
        "scope_note": ("PARSED means the record satisfied the compiled-program, "
                       "executed-anchor, sealed-realizer and check-set contracts. "
                       "It does NOT establish question quality, which requires "
                       "blinded judging."),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ECM-v3 two-stage runner; dry-run by default.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "ecm_inputs_final_v2.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=v1.DEFAULT_SEED)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default=GEN_BASE_URL)
    parser.add_argument("--model", default=GEN_MODEL)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--condition", action="append", choices=v1.CONDITIONS)
    parser.add_argument("--chunk-id", action="append")
    parser.add_argument("--api-key-file")
    args = parser.parse_args()

    api_key = None
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if args.execute and not api_key:
        parser.error("--execute requires --api-key-file")

    try:
        summary = run(args.manifest, args.out_dir, args.experiment_id, args.seed,
                      execute=args.execute, base_url=args.base_url, model=args.model,
                      timeout_sec=args.timeout_sec, retries=args.retries,
                      conditions=tuple(args.condition) if args.condition else None,
                      chunk_ids=set(args.chunk_id) if args.chunk_id else None,
                      api_key=api_key, api_key_file=args.api_key_file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
