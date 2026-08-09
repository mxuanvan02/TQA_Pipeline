"""Direct generation: the two baseline arms that skip answer-first construction.

``text_only`` (OCR transcript only) and ``direct`` (OCR transcript + page pixels)
both ask a single model call to write the finished item. They exist to bound how
much of ECM--TQAG's item quality is attributable to the planner/realizer split
rather than to the underlying model.

Two properties are enforced here because the comparison is meaningless without
them:

1. ONE TEMPLATE. ``direct_prompt`` renders a single byte-identical instruction
   block for both arms; the only difference is the declared ``INPUT_MODE`` value
   and, for the multimodal arm, the presence of image parts in the request. A
   difference between the arms is therefore a difference in INPUT, not in
   instructions.

2. ONE ITEM SCHEMA. The emitted object is ``prompts.ITEM_SCHEMA_BLOCK``, the same
   closed schema the sealed realizer emits, so every arm's item is the same
   object and item-level gates apply unchanged.

``parse_item`` fails closed and never repairs: a provider that returns three
choices has broken the contract, and silently padding it would fabricate data.

No network I/O; request builders return payload dicts and the transport stamps
model identity from the approved execution roster.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from . import structure_reader
from .arms import ARM_BY_NAME
from .prompts import (
    DECODING,
    ITEM_KEYS,
    ITEM_SCHEMA_BLOCK,
    N_CHOICES,
    N_DISTRACTOR_FAULTS,
)

DIRECT_TEMPLATE_ID = "ecm-tqag.direct-generation"

# Declared by the frozen design rather than restated, so a change to arms.py can
# never leave this module describing a different experiment.
DIRECT_INPUT_MODES = (
    ARM_BY_NAME["text_only"].input_mode,
    ARM_BY_NAME["direct"].input_mode,
)
MULTIMODAL_INPUT_MODE = ARM_BY_NAME["direct"].input_mode

_INSTRUCTIONS = (
    "Bạn là bộ SINH CÂU HỎI TRỰC TIẾP. Đọc tài liệu được cung cấp và viết một câu "
    "hỏi trắc nghiệm tiếng Việt có đúng bốn lựa chọn, trong đó đúng một lựa chọn "
    "đúng. Câu hỏi phải phụ thuộc vào một quy tắc hoặc điều kiện cụ thể trong tài "
    "liệu, không hỏi kiến thức chung. Không dùng cách dẫn chung như \"theo hình\" "
    "hay \"dựa vào sơ đồ\"; hãy nêu trực tiếp nội dung cần suy luận.\n"
    "distractor_faults nêu ba lỗi tương ứng với ba lựa chọn sai, theo đúng thứ tự "
    "xuất hiện của chúng trong choices.\n"
    "Chỉ trả về một đối tượng JSON theo schema sau, không rào mã, không lời dẫn.\n"
    f"{ITEM_SCHEMA_BLOCK}\n"
)


def _fail(schema: str, reason: str) -> ValueError:
    return ValueError(f"{schema}:{reason}")


def direct_prompt(text: str, *, input_mode: str) -> str:
    """Render the single direct-generation template for either baseline arm."""
    if input_mode not in DIRECT_INPUT_MODES:
        raise _fail("direct_prompt", f"unknown_input_mode:{input_mode}")
    if not isinstance(text, str) or not text.strip():
        raise _fail("direct_prompt", "empty_text")
    return f"{_INSTRUCTIONS}INPUT_MODE={input_mode}\nTEXT={text}"


def direct_static_fingerprint() -> str:
    """Digest of the template with both variable slots neutralised.

    Normalising the input-mode value lets the text-only and multimodal arms prove
    the surrounding instructions are byte-identical, exactly as
    ``prompts.planner_static_fingerprint`` does for the two interface arms.
    """
    sentinel = direct_prompt("<TEXT>", input_mode=DIRECT_INPUT_MODES[0])
    sentinel = sentinel.replace(
        f"INPUT_MODE={DIRECT_INPUT_MODES[0]}", "INPUT_MODE=<MODE>"
    )
    return hashlib.sha256(sentinel.encode("utf-8")).hexdigest()


def direct_request(
    text: str,
    *,
    input_mode: str,
    image_data_urls: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the provider-neutral direct-generation request.

    The text-only arm must be text-only by construction: supplying images to it
    is a hard error rather than a silently ignored argument, because a leaked
    image would collapse the baseline into the multimodal arm.
    """
    prompt = direct_prompt(text, input_mode=input_mode)
    images = list(image_data_urls or [])
    if input_mode != MULTIMODAL_INPUT_MODE:
        if images:
            raise _fail("direct_request", "text_only_must_not_receive_images")
    elif not images:
        raise _fail("direct_request", "missing_image_payload")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for url in images:
        if not isinstance(url, str) or not url.startswith("data:image/"):
            raise _fail("direct_request", "invalid_image_data_url")
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {
        "messages": [{"role": "user", "content": content}],
        "temperature": DECODING["temperature"],
        "max_tokens": DECODING["max_tokens"],
    }


def parse_item(raw: str) -> dict[str, Any]:
    """Parse the closed item schema shared by every construction path."""
    payload, _fenced = structure_reader.strip_fence(raw if isinstance(raw, str) else "")
    try:
        obj = json.loads(payload)
    except Exception as exc:
        raise _fail("item_schema", f"invalid_json:{type(exc).__name__}") from exc
    if not isinstance(obj, dict):
        raise _fail("item_schema", "top_level_not_object")

    keys = set(obj)
    extra = sorted(keys - set(ITEM_KEYS))
    if extra:
        raise _fail("item_schema", f"unexpected_keys:{extra}")
    missing = sorted(set(ITEM_KEYS) - keys)
    if missing:
        raise _fail("item_schema", f"missing_keys:{missing}")

    question = obj["question"]
    if not isinstance(question, str) or not question.strip():
        raise _fail("item_schema", "empty_question")

    choices = obj["choices"]
    if not isinstance(choices, list) or len(choices) != N_CHOICES:
        raise _fail("item_schema", f"choices_must_have_four_entries:{_length(choices)}")
    for i, choice in enumerate(choices):
        if not isinstance(choice, str) or not choice.strip():
            raise _fail("item_schema", f"invalid_choice:{i}")
    normalised = [" ".join(c.split()).casefold() for c in choices]
    if len(set(normalised)) != N_CHOICES:
        raise _fail("item_schema", "duplicate_choices")

    answer_index = obj["answer_index"]
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        raise _fail("item_schema", "answer_index_not_int")
    if not 0 <= answer_index < N_CHOICES:
        raise _fail("item_schema", f"answer_index_out_of_range:{answer_index}")

    rationale = obj["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise _fail("item_schema", "empty_rationale")

    faults = obj["distractor_faults"]
    if not isinstance(faults, list) or len(faults) != N_DISTRACTOR_FAULTS:
        raise _fail(
            "item_schema",
            f"distractor_faults_must_have_three_entries:{_length(faults)}",
        )
    for i, fault in enumerate(faults):
        if not isinstance(fault, str) or not fault.strip():
            raise _fail("item_schema", f"invalid_distractor_fault:{i}")
    return obj


def _length(value: Any) -> str:
    return str(len(value)) if isinstance(value, (list, tuple)) else type(value).__name__


__all__ = [
    "DIRECT_INPUT_MODES",
    "DIRECT_TEMPLATE_ID",
    "ITEM_KEYS",
    "ITEM_SCHEMA_BLOCK",
    "MULTIMODAL_INPUT_MODE",
    "direct_prompt",
    "direct_request",
    "direct_static_fingerprint",
    "parse_item",
]
