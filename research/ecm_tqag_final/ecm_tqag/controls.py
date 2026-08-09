"""Frozen synthetic controls for validating answerer sensitivity.

These controls are created before the paid experiment and have known answers.
Positive controls require a value visible only in pixels. Negative controls put
the answer in the supplied text and use an irrelevant image. Images are derived
byte-for-byte from the fixture record, so the fixture JSON digest is sufficient
to freeze both content and rendering.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont

from .io import canonical, sha256_bytes

CONTROL_SCHEMA = "ecm-tqag.sensitivity-controls.v1"
CONTROL_TYPES = frozenset({"positive_visual", "negative_text_sufficient"})
EXPECTED_COUNT = 10


def _blocked(reason: str) -> ValueError:
    return ValueError(f"BLOCKED_CONTROLS:{reason}")


def load_controls(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise _blocked(f"unreadable:{type(exc).__name__}") from exc
    validate_controls(obj)
    return obj


def validate_controls(obj: Any) -> None:
    if not isinstance(obj, dict) or set(obj) != {"schema", "rendering", "controls"}:
        raise _blocked("invalid_top_level")
    if obj["schema"] != CONTROL_SCHEMA:
        raise _blocked("invalid_schema")
    rendering = obj["rendering"]
    if rendering != {"width": 720, "height": 300, "format": "PNG", "background": "white"}:
        raise _blocked("rendering_not_frozen")
    rows = obj["controls"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise _blocked(f"count:{len(rows) if isinstance(rows, list) else 'not_list'}")
    ids: set[str] = set()
    counts = {kind: 0 for kind in CONTROL_TYPES}
    required = {"control_id", "type", "question", "choices", "answer_index", "text", "image_lines"}
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise _blocked(f"invalid_record:{i}")
        cid = row["control_id"]
        if not isinstance(cid, str) or not cid or cid in ids:
            raise _blocked(f"invalid_or_duplicate_id:{i}")
        ids.add(cid)
        kind = row["type"]
        if kind not in CONTROL_TYPES:
            raise _blocked(f"invalid_type:{cid}")
        counts[kind] += 1
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise _blocked(f"invalid_question:{cid}")
        choices = row["choices"]
        if not isinstance(choices, list) or len(choices) != 4 or any(not isinstance(x, str) or not x for x in choices):
            raise _blocked(f"invalid_choices:{cid}")
        if len(set(x.casefold() for x in choices)) != 4:
            raise _blocked(f"duplicate_choices:{cid}")
        answer = row["answer_index"]
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4:
            raise _blocked(f"invalid_answer:{cid}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise _blocked(f"invalid_text:{cid}")
        lines = row["image_lines"]
        if not isinstance(lines, list) or not lines or any(not isinstance(x, str) or not x for x in lines):
            raise _blocked(f"invalid_image_lines:{cid}")
        answer_text = choices[answer]
        joined_pixels = " ".join(lines)
        if kind == "positive_visual":
            if answer_text.casefold() in row["text"].casefold():
                raise _blocked(f"positive_answer_leaks_to_text:{cid}")
            if answer_text.casefold() not in joined_pixels.casefold():
                raise _blocked(f"positive_answer_missing_from_pixels:{cid}")
        else:
            if answer_text.casefold() not in row["text"].casefold():
                raise _blocked(f"negative_answer_missing_from_text:{cid}")
    if counts != {"positive_visual": 5, "negative_text_sufficient": 5}:
        raise _blocked(f"type_balance:{counts}")


def control_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "question": row["question"],
        "choices": list(row["choices"]),
        "answer_index": row["answer_index"],
        "rationale": "Đáp án đã biết trước trong control fixture.",
        "distractor_faults": ["control distractor"] * 3,
    }


def render_control_png(row: Mapping[str, Any]) -> bytes:
    lines = row.get("image_lines")
    if not isinstance(lines, list) or not lines:
        raise _blocked("cannot_render_invalid_lines")
    image = Image.new("RGB", (720, 300), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((8, 8, 711, 291), outline="black", width=4)
    y = 55
    for line in lines:
        draw.text((45, y), str(line), fill="black", font=font)
        y += 52
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def control_image_data_url(row: Mapping[str, Any]) -> str:
    return "data:image/png;base64," + base64.b64encode(render_control_png(row)).decode("ascii")


def controls_commitment(obj: Mapping[str, Any]) -> str:
    validate_controls(dict(obj))
    rendered = [sha256_bytes(render_control_png(row)) for row in obj["controls"]]
    return sha256_bytes(canonical({"fixture": obj, "rendered_png_sha256": rendered}).encode("utf-8"))


__all__ = ["CONTROL_SCHEMA", "control_image_data_url", "control_item", "controls_commitment",
           "load_controls", "render_control_png", "validate_controls"]
