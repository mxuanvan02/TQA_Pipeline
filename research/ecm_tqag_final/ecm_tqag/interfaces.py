"""Visual interface extraction: the two pixel channels the planner may consume.

Every arm that sees a page image must reduce it to ONE of two closed interfaces
before the answer-first planner is allowed to look at anything:

  * ``closed_graph``  -- the structure graph of :mod:`ecm_tqag.structure_reader`
  * ``caption``       -- a short relational caption produced from pixels only

Both are then rendered through the single downstream template in
:func:`ecm_tqag.prompts.planner_prompt`, so the arms differ in the interface
slot and nowhere else. That is the whole point of the design: if the graph and
caption arms shared no template, a difference between them would be
confounded with a difference in instructions.

Two invariants are enforced structurally rather than by convention:

1. The caption channel is BLIND. ``caption_prompt`` and ``caption_request``
   take no parameter through which OCR text, page prose, or any transcript can
   enter, mirroring ``structure_reader.read_structure``. A caption computed from
   the text would make "visual dependence" a theorem about the protocol instead
   of a measurement (see the module docstring of ``structure_reader``).

2. Parsers FAIL CLOSED and never repair. Every rejection carries a stable
   ``<schema>:<reason>`` string so the ledger records exactly which contract a
   provider broke.

Nothing here performs network I/O; request builders return payload dicts and the
model field is supplied later by the transport's execution roster.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import structure_reader
from .prompts import DECODING

SCHEMA_CAPTION = "ecm-tqag.visual-caption.release"
SCHEMA_GRAPH_INTERFACE = "ecm-tqag.graph-interface.release"

CAPTION_TEMPLATE_ID = "ecm-tqag.pixels-only-caption"
OCR_ASSISTED_GRAPH_TEMPLATE_ID = "ecm-tqag.ocr-assisted-structure-read"

# A caption is an INTERFACE, not a transcription. The word cap keeps the channel
# narrow enough that it cannot smuggle the page prose to the planner, and the
# relation cap keeps it comparable in size to a structure graph.
CAPTION_MAX_CAPTION_WORDS = 60
CAPTION_MAX_RELATIONS = 12
CAPTION_MIN_RELATION_WORDS = 2
CAPTION_KEYS = frozenset({"caption", "relations", "confidence"})

# Provider-enforced form of the same frozen contract. The local parser remains
# authoritative and still rejects rather than repairs; supported transports can
# prevent an otherwise valid answer from exceeding the relation ceiling.
CAPTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ecm_tqag_visual_caption",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["caption", "relations", "confidence"],
            "properties": {
                "caption": {"type": "string", "minLength": 1},
                "relations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": CAPTION_MAX_RELATIONS,
                    "items": {"type": "string", "minLength": 1},
                },
                "confidence": {
                    "type": "number", "minimum": 0.0, "maximum": 1.0,
                },
            },
        },
    },
}

CAPTION_PROMPT = """\
Bạn chỉ được xem một ảnh trang sách. Bạn KHÔNG được cung cấp bản chữ của trang
và KHÔNG được suy đoán nội dung văn xuôi của trang.

Nhiệm vụ duy nhất: mô tả CẤU TRÚC QUAN HỆ của sơ đồ, biểu đồ hoặc hình vẽ trong
ảnh, bằng tiếng Việt, chỉ dựa trên những gì nhìn thấy.

Chỉ trả về một đối tượng JSON, không rào mã, không lời dẫn, đúng ba khóa sau:

{"caption": "<mô tả cấu trúc, tối đa 60 từ>",
 "relations": ["<một quan hệ giữa hai thành phần nhìn thấy được>"],
 "confidence": 0.0}

Quy tắc:
- "caption" mô tả bố cục và các thành phần nhìn thấy được, tối đa 60 từ.
- "relations" liệt kê từ 1 đến 12 quan hệ, mỗi quan hệ là một câu ngắn nêu rõ
  hai thành phần và mối liên hệ giữa chúng (ví dụ: mũi tên, chứa trong, cùng hàng).
- Không mô tả văn xuôi của trang, không trích dẫn đoạn văn, không tóm tắt bài học.
- "confidence" là mức tin cậy của bạn về cấu trúc đã đọc, từ 0.0 đến 1.0.

Chỉ phát ra đối tượng JSON.
"""

CAPTION_PROMPT_SHA256 = hashlib.sha256(CAPTION_PROMPT.encode("utf-8")).hexdigest()

# The OCR-assisted reader arm deliberately reuses the frozen blind graph
# instrument verbatim and appends the transcript slot. Keeping the schema block
# byte-identical to ``structure_reader.PROMPT`` is what makes
# ``text_assisted_reader`` a manipulation of the INPUT only.
OCR_ASSISTED_SUFFIX = """
Ngoài ảnh, bạn còn được cung cấp bản chữ OCR của trang. Bản chữ chỉ dùng để đọc
đúng nhãn trong hình; cấu trúc (nút, cạnh, bbox) vẫn phải đọc từ ảnh. Không thêm
nút nào chỉ xuất hiện trong bản chữ mà không nhìn thấy trong hình.

"""


def _fail(schema: str, reason: str) -> ValueError:
    return ValueError(f"{schema}:{reason}")


# --------------------------------------------------------------------------
# caption channel (pixels only, by construction)
# --------------------------------------------------------------------------
def caption_prompt() -> str:
    """The frozen pixels-only caption prompt.

    Takes no arguments at all: there is no slot a transcript could occupy.
    """
    return CAPTION_PROMPT


def caption_request(image_data_url: str, *, max_tokens: int = 400) -> dict[str, Any]:
    """Build the provider-neutral caption request for exactly one image.

    ``model`` is intentionally absent; the transport stamps model identity from
    the approved execution roster so a prompt builder can never pick a provider.
    """
    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:image/"):
        raise _fail("caption_request", "invalid_image_data_url")
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "temperature": DECODING["temperature"],
        "max_tokens": min(int(DECODING["max_tokens"]), int(max_tokens)),
        "response_format": CAPTION_RESPONSE_FORMAT,
    }


def parse_caption(raw: str) -> dict[str, Any]:
    """Parse the closed caption schema. Never repairs, never defaults."""
    payload, _fenced = structure_reader.strip_fence(raw if isinstance(raw, str) else "")
    try:
        obj = json.loads(payload)
    except Exception as exc:
        raise _fail("caption_schema", f"invalid_json:{type(exc).__name__}") from exc
    if not isinstance(obj, dict):
        raise _fail("caption_schema", "top_level_not_object")

    keys = set(obj)
    extra = sorted(keys - CAPTION_KEYS)
    if extra:
        raise _fail("caption_schema", f"unexpected_keys:{extra}")
    missing = sorted(CAPTION_KEYS - keys)
    if missing:
        raise _fail("caption_schema", f"missing_keys:{missing}")

    caption = obj["caption"]
    if not isinstance(caption, str) or not caption.strip():
        raise _fail("caption_schema", "empty_caption")
    n_words = len(caption.split())
    if n_words > CAPTION_MAX_CAPTION_WORDS:
        raise _fail(
            "caption_schema", f"caption_too_long:{n_words}>{CAPTION_MAX_CAPTION_WORDS}"
        )

    relations = obj["relations"]
    if not isinstance(relations, list) or not relations:
        raise _fail("caption_schema", "no_relations")
    if len(relations) > CAPTION_MAX_RELATIONS:
        raise _fail(
            "caption_schema", f"too_many_relations:{len(relations)}>{CAPTION_MAX_RELATIONS}"
        )
    for i, relation in enumerate(relations):
        if (
            not isinstance(relation, str)
            or len(relation.split()) < CAPTION_MIN_RELATION_WORDS
        ):
            raise _fail("caption_schema", f"invalid_relation:{i}")

    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise _fail("caption_schema", "confidence_not_unit_interval")
    if not 0.0 <= float(confidence) <= 1.0:
        raise _fail("caption_schema", "confidence_not_unit_interval")
    return obj


def interface_from_caption(parsed: dict[str, Any]) -> dict[str, Any]:
    """Project a validated caption onto the planner-visible interface.

    ``confidence`` is extraction metadata for the ledger, not evidence, so it is
    dropped here: the planner must not be able to condition on how sure the
    captioner was.
    """
    caption = parsed.get("caption")
    relations = parsed.get("relations")
    if not isinstance(caption, str) or not isinstance(relations, list):
        raise _fail("caption_interface", "unvalidated_caption")
    return {"caption": caption, "relations": list(relations)}


# --------------------------------------------------------------------------
# closed-graph channel (blind, and OCR-assisted)
# --------------------------------------------------------------------------
def ocr_assisted_graph_prompt(page_transcript: str) -> str:
    """The blind graph instrument plus the OCR transcript slot.

    Reuses ``structure_reader.PROMPT`` byte-for-byte so the reader arms differ
    from the blind arm only in what is supplied, never in what is asked.
    """
    if not isinstance(page_transcript, str):
        raise _fail("ocr_assisted_graph", "ocr_text_not_string")
    if not page_transcript.strip():
        raise _fail("ocr_assisted_graph", "empty_ocr_text")
    return (
        structure_reader.PROMPT
        + OCR_ASSISTED_SUFFIX
        + "OCR_TEXT="
        + page_transcript
    )


def ocr_assisted_graph_static_fingerprint() -> str:
    """Digest of everything in the reader prompt except the transcript slot."""
    sentinel = ocr_assisted_graph_prompt("<OCR_TEXT>")
    return hashlib.sha256(sentinel.encode("utf-8")).hexdigest()


def parse_graph_interface(raw: str) -> dict[str, Any]:
    """Parse a structure graph using the frozen closed schema.

    Delegates to ``structure_reader.parse_and_validate`` so the graph contract
    has exactly one implementation, and re-raises its stable reason string.
    """
    result = structure_reader.parse_and_validate(raw if isinstance(raw, str) else "")
    if not result["ok"]:
        detail = result.get("detail") or ""
        suffix = f":{detail}" if detail else ""
        raise _fail("graph_schema", f"{result['reason']}{suffix}")
    graph = result["graph"]
    assert isinstance(graph, dict)  # guaranteed by validate_graph
    return graph


def _normalise_bbox(bbox: list[Any], width: int, height: int) -> list[float]:
    """Convert the frozen 0--1000 full-image grid to planner [0,1] coordinates.

    ``width`` and ``height`` remain mandatory so callers must still prove that the
    source image is readable, but provider-side resizing cannot affect this grid.
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise _fail("graph_interface", "invalid_bbox")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox):
        raise _fail("graph_interface", "invalid_bbox")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not (0.0 <= x0 < x1 <= 1000.0 and 0.0 <= y0 < y1 <= 1000.0):
        raise _fail("graph_interface", "invalid_bbox")
    return [round(x0 / 1000.0, 6), round(y0 / 1000.0, 6),
            round(x1 / 1000.0, 6), round(y1 / 1000.0, 6)]


def interface_from_graph(graph: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    """Project a validated graph onto the planner-visible interface.

    Reader boxes use the fixed 0--1000 full-image grid and are converted to the
    normalized ``[0,1]`` space that ``guards.verify_visual_anchors`` replays
    against. ``confidence`` is dropped for the same reason as in the caption
    channel.
    """
    if isinstance(width, bool) or isinstance(height, bool):
        raise _fail("graph_interface", "invalid_image_size")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise _fail("graph_interface", "invalid_image_size")
    verdict = structure_reader.validate_graph(graph)
    if not verdict["ok"]:
        raise _fail("graph_interface", str(verdict["reason"]))
    return {
        "graph_type": graph["graph_type"],
        "nodes": [
            {
                "id": node["id"],
                "label": node["label"],
                "level": node["level"],
                "bbox": _normalise_bbox(node["bbox"], width, height),
            }
            for node in graph["nodes"]
        ],
        "edges": [
            {"src": edge["src"], "dst": edge["dst"], "kind": edge["kind"], "label": edge["label"]}
            for edge in graph["edges"]
        ],
    }
