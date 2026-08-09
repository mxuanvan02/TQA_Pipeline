"""Closed answerer, image-auditor, and blinded-judge instruments.

Pure request builders and strict parsers. Model identity is injected only by the
frozen transport roster.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from .direct import parse_item
from .prompts import DECODING, canonical
from .structure_reader import strip_fence

ANSWER_SCHEMA = frozenset({"answer_index", "abstain", "confidence"})
AUDIT_SCHEMA = frozenset({"supported", "relation_supported", "notes"})
JUDGE_SCHEMA = frozenset({"answerability", "single_best_answer", "clarity", "notes"})


def _fail(kind: str, reason: str) -> ValueError:
    return ValueError(f"{kind}_schema:{reason}")


def _load(raw: str, kind: str) -> dict[str, Any]:
    body, _ = strip_fence(raw if isinstance(raw, str) else "")
    try:
        obj = json.loads(body)
    except Exception as exc:
        raise _fail(kind, f"invalid_json:{type(exc).__name__}") from exc
    if not isinstance(obj, dict):
        raise _fail(kind, "top_level_not_object")
    return obj


def answerer_request(item: dict[str, Any], *, text: str | None,
                     image_data_urls: Sequence[str] = ()) -> dict[str, Any]:
    parsed = parse_item(json.dumps(item, ensure_ascii=False))
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("answerer_request:invalid_text")
    content: list[dict[str, Any]] = [{"type": "text", "text": (
        "Làm câu hỏi trắc nghiệm sau. Chỉ dùng bằng chứng được cung cấp. "
        "Nếu không đủ bằng chứng, abstain=true và answer_index=null. "
        "Chỉ trả JSON {\"answer_index\":0,\"abstain\":false,\"confidence\":0.0}.\n"
        f"ITEM={canonical({'question': parsed['question'], 'choices': parsed['choices']})}\n"
        f"TEXT={text or '<REMOVED>'}"
    )}]
    for url in image_data_urls:
        if not isinstance(url, str) or not url.startswith("data:image/"):
            raise ValueError("answerer_request:invalid_image_data_url")
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": 200}


def parse_answer(raw: str) -> dict[str, Any]:
    obj = _load(raw, "answer")
    if set(obj) != ANSWER_SCHEMA:
        raise _fail("answer", "keys")
    abstain = obj["abstain"]
    index = obj["answer_index"]
    confidence = obj["confidence"]
    if not isinstance(abstain, bool):
        raise _fail("answer", "abstain_not_bool")
    if abstain:
        if index is not None:
            raise _fail("answer", "abstention_must_have_null_index")
    elif isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 4:
        raise _fail("answer", "index_out_of_range")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise _fail("answer", "confidence_not_unit_interval")
    return obj


def image_audit_request(interface: dict[str, Any], image_data_url: str) -> dict[str, Any]:
    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:image/"):
        raise ValueError("image_audit_request:invalid_image_data_url")
    prompt = (
        "Kiểm tra độc lập interface có được ảnh hỗ trợ hay không. Không dùng văn bản nguồn. "
        "Chỉ trả JSON {\"supported\":true,\"relation_supported\":true,\"notes\":\"...\"}.\n"
        f"INTERFACE={canonical(interface)}"
    )
    return {"messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]}], "temperature": 0, "max_tokens": 300}


def parse_image_audit(raw: str) -> dict[str, Any]:
    obj = _load(raw, "audit")
    if set(obj) != AUDIT_SCHEMA:
        raise _fail("audit", "keys")
    if not isinstance(obj["supported"], bool) or not isinstance(obj["relation_supported"], bool):
        raise _fail("audit", "verdict_not_bool")
    if not isinstance(obj["notes"], str):
        raise _fail("audit", "notes_not_string")
    return obj


def judge_request(item: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_item(json.dumps(item, ensure_ascii=False))
    public = {"question": parsed["question"], "choices": parsed["choices"]}
    prompt = (
        "Đánh giá câu hỏi đã làm mù nguồn tạo. Bạn chỉ được chấm các thuộc tính có thể "
        "quan sát từ câu hỏi và bốn lựa chọn; không suy đoán mức hỗ trợ của nguồn vì nguồn "
        "không được cung cấp. Chỉ trả JSON với điểm nguyên 1..5: "
        "{\"answerability\":1,\"single_best_answer\":1,\"clarity\":1,\"notes\":\"...\"}.\n"
        f"ITEM={canonical(public)}"
    )
    return {"messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 250}


def parse_judgement(raw: str) -> dict[str, Any]:
    obj = _load(raw, "judge")
    if set(obj) != JUDGE_SCHEMA:
        raise _fail("judge", "keys")
    for key in ("answerability", "single_best_answer", "clarity"):
        value = obj[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise _fail("judge", f"invalid_score:{key}")
    if not isinstance(obj["notes"], str):
        raise _fail("judge", "notes_not_string")
    return obj


__all__ = ["answerer_request", "parse_answer", "image_audit_request",
           "parse_image_audit", "judge_request", "parse_judgement"]
