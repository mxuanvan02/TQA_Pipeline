from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from ecm_tqag.run.smoke import (
    SMOKE_CODE,
    build_smoke_payload,
    parse_smoke_answer,
    validate_smoke_answer,
)


def _png_data_url() -> str:
    image = Image.new("RGB", (64, 32), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_vision_smoke_payload_contains_pixels_but_no_local_path() -> None:
    payload = build_smoke_payload(vision_required=True, image_data_url=_png_data_url())
    encoded = json.dumps(payload)
    assert "image_url" in encoded
    assert "data:image/png;base64," in encoded
    assert "/media/" not in encoded
    assert payload["temperature"] == 0
    assert payload["max_tokens"] <= 120


def test_text_smoke_payload_has_no_image_part() -> None:
    payload = build_smoke_payload(vision_required=False, image_data_url=None)
    encoded = json.dumps(payload)
    assert "image_url" not in encoded
    assert SMOKE_CODE not in encoded  # code is only present inside test pixels


def test_smoke_answer_requires_exact_visual_code_for_vision_role() -> None:
    good = parse_smoke_answer(f'```json\n{{"status":"ok","code":"{SMOKE_CODE}"}}\n```')
    assert validate_smoke_answer(good, vision_required=True) is True
    with pytest.raises(ValueError, match="visual_code_mismatch"):
        validate_smoke_answer({"status": "ok", "code": "guessed"}, vision_required=True)


def test_text_smoke_requires_exact_schema() -> None:
    assert validate_smoke_answer({"status": "ok", "code": None}, vision_required=False)
    with pytest.raises(ValueError, match="invalid_status"):
        validate_smoke_answer({"status": "maybe", "code": None}, vision_required=False)
