from __future__ import annotations

import json
import re
from typing import Any

from ..prompts import DECODING

# The code is rendered into the smoke image by the caller; it must never occur
# in the textual prompt, so a successful vision smoke tests actual pixel use.
SMOKE_CODE = "ECM-7Q4P"


def build_smoke_payload(*, vision_required: bool, image_data_url: str | None) -> dict[str, Any]:
    """Build a minimal, provider-neutral multimodal smoke request."""
    if vision_required and not isinstance(image_data_url, str):
        raise ValueError("BLOCKED_SMOKE:missing_image_payload")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                'Return JSON only with exactly two keys: status and code. '
                'Set status to "ok". For a vision role, read the short code '
                "printed in the supplied image and place it in code."
            ),
        }
    ]
    if vision_required:
        assert isinstance(image_data_url, str)  # narrowed by the fail-closed check above
        if not image_data_url.startswith("data:image/"):
            raise ValueError("BLOCKED_SMOKE:invalid_image_data_url")
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    else:
        content[0]["text"] += " If no image is supplied, set code to null."
    return {
        "messages": [{"role": "user", "content": content}],
        "temperature": DECODING["temperature"],
        "max_tokens": min(int(DECODING["max_tokens"]), 120),
    }


def parse_smoke_answer(value: str | dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON object returned either raw or in a fenced code block."""
    if isinstance(value, dict):
        parsed: Any = value
    elif isinstance(value, str):
        text = value.strip()
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
        if match:
            text = match.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("BLOCKED_SMOKE:invalid_json") from exc
    else:
        raise ValueError("BLOCKED_SMOKE:invalid_answer_type")
    if not isinstance(parsed, dict):
        raise ValueError("BLOCKED_SMOKE:answer_not_object")
    if set(parsed) != {"status", "code"}:
        raise ValueError("BLOCKED_SMOKE:unexpected_answer_keys")
    return parsed


def validate_smoke_answer(answer: dict[str, Any], *, vision_required: bool) -> bool:
    """Accept smoke output only if schema and, for vision, pixels are verified."""
    if not isinstance(answer, dict) or set(answer) != {"status", "code"}:
        raise ValueError("BLOCKED_SMOKE:unexpected_answer_keys")
    if answer.get("status") != "ok":
        raise ValueError("BLOCKED_SMOKE:invalid_status")
    if vision_required and answer.get("code") != SMOKE_CODE:
        raise ValueError("BLOCKED_SMOKE:visual_code_mismatch")
    if not vision_required and answer.get("code") is not None:
        raise ValueError("BLOCKED_SMOKE:text_code_must_be_null")
    return True
