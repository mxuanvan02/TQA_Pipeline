from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ecm_tqag.run.smoke import parse_smoke_answer, validate_smoke_answer


def assistant_content(body: dict[str, Any]) -> str:
    """Extract one assistant text response from an OpenAI-compatible envelope."""
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("BLOCKED_SMOKE:invalid_choice_count")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("BLOCKED_SMOKE:choice_not_object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("BLOCKED_SMOKE:message_not_object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("BLOCKED_SMOKE:content_missing")
    return content


def validate_smoke_sidecar(path: Path, *, vision_required: bool) -> bool:
    """Validate the persisted provider envelope without trusting HTTP status alone."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("BLOCKED_SMOKE:sidecar_not_object")
    answer = parse_smoke_answer(assistant_content(value))
    return validate_smoke_answer(answer, vision_required=vision_required)
