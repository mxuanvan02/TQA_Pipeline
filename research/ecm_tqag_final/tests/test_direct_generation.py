from __future__ import annotations

import json

import pytest

from ecm_tqag.arms import ARM_BY_NAME
from ecm_tqag.direct import (
    DIRECT_INPUT_MODES,
    ITEM_KEYS,
    ITEM_SCHEMA_BLOCK,
    direct_prompt,
    direct_request,
    direct_static_fingerprint,
    parse_item,
)
from ecm_tqag.prompts import DECODING, realizer_prompt

PNG = "data:image/png;base64,AAAA"
TEXT = "Điều 1. Quốc hội là cơ quan quyền lực nhà nước cao nhất."


def _item() -> dict:
    return {
        "question": "Cơ quan nào được mô tả là quyền lực nhà nước cao nhất?",
        "choices": ["Quốc hội", "Chính phủ", "Toà án", "Viện kiểm sát"],
        "answer_index": 0,
        "rationale": "Điều 1 nêu Quốc hội là cơ quan quyền lực nhà nước cao nhất.",
        "distractor_faults": ["sai cấp", "sai chức năng", "sai chủ thể"],
    }


def _sealed() -> dict:
    return {
        "motif": "diagram_text_reconcile",
        "atoms": [{"atom_id": "a0", "op": "locate", "value": "v"}],
        "anchors": [{"atom_id": "a1", "excerpt": "e", "sentence_index": 0}],
        "visual_observations": [],
        "construction_hash": "deadbeef",
    }


# --------------------------------------------------------------- one template
def test_direct_input_modes_match_the_frozen_arm_design() -> None:
    assert DIRECT_INPUT_MODES == (
        ARM_BY_NAME["text_only"].input_mode,
        ARM_BY_NAME["direct"].input_mode,
    )


def test_text_only_and_multimodal_share_one_byte_identical_template() -> None:
    a = direct_prompt(TEXT, input_mode="ocr_only")
    b = direct_prompt(TEXT, input_mode="ocr_layout_pixels")
    prefix_a = a.replace("INPUT_MODE=ocr_only", "INPUT_MODE=<MODE>").split("TEXT=", 1)[0]
    prefix_b = b.replace("INPUT_MODE=ocr_layout_pixels", "INPUT_MODE=<MODE>").split("TEXT=", 1)[0]
    assert prefix_a == prefix_b
    assert a.split("TEXT=", 1)[1] == TEXT
    assert direct_static_fingerprint() == direct_static_fingerprint()


def test_direct_prompt_rejects_unknown_input_mode_and_empty_text() -> None:
    with pytest.raises(ValueError, match="direct_prompt:unknown_input_mode"):
        direct_prompt(TEXT, input_mode="pixels_only")
    with pytest.raises(ValueError, match="direct_prompt:empty_text"):
        direct_prompt("  ", input_mode="ocr_only")


def test_direct_item_schema_is_the_realizer_schema() -> None:
    # Every arm must emit the same item object, otherwise the arms are not
    # comparable at the item level.
    assert ITEM_SCHEMA_BLOCK in realizer_prompt(_sealed())
    assert ITEM_SCHEMA_BLOCK in direct_prompt(TEXT, input_mode="ocr_only")
    assert ITEM_KEYS == {"question", "choices", "answer_index", "rationale", "distractor_faults"}


# --------------------------------------------------------------- requests
def test_text_only_request_carries_no_pixels() -> None:
    payload = direct_request(TEXT, input_mode="ocr_only")
    assert set(payload) == {"messages", "temperature", "max_tokens"}
    assert payload["temperature"] == DECODING["temperature"]
    assert payload["max_tokens"] == DECODING["max_tokens"]
    content = payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text"]
    with pytest.raises(ValueError, match="direct_request:text_only_must_not_receive_images"):
        direct_request(TEXT, input_mode="ocr_only", image_data_urls=[PNG])


def test_multimodal_request_requires_at_least_one_valid_image() -> None:
    payload = direct_request(TEXT, input_mode="ocr_layout_pixels", image_data_urls=[PNG, PNG])
    content = payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "image_url"]
    assert content[1]["image_url"]["url"] == PNG
    with pytest.raises(ValueError, match="direct_request:missing_image_payload"):
        direct_request(TEXT, input_mode="ocr_layout_pixels")
    with pytest.raises(ValueError, match="direct_request:invalid_image_data_url"):
        direct_request(TEXT, input_mode="ocr_layout_pixels", image_data_urls=["/tmp/p.png"])


# --------------------------------------------------------------- item parser
def test_item_parser_accepts_the_closed_item_schema() -> None:
    assert parse_item(json.dumps(_item(), ensure_ascii=False)) == _item()
    fenced = "```json\n" + json.dumps(_item(), ensure_ascii=False) + "\n```"
    assert parse_item(fenced) == _item()


def test_item_parser_fails_closed_and_never_repairs() -> None:
    cases = {
        "item_schema:invalid_json": "nope",
        "item_schema:top_level_not_object": "[]",
        "item_schema:unexpected_keys": json.dumps(_item() | {"chunk_id": "c1"}),
        "item_schema:empty_question": json.dumps(_item() | {"question": " "}),
        "item_schema:choices_must_have_four_entries": json.dumps(
            _item() | {"choices": ["a", "b", "c"]}
        ),
        "item_schema:duplicate_choices": json.dumps(
            _item() | {"choices": ["Quốc hội", "Quốc hội", "Toà án", "Viện kiểm sát"]}
        ),
        "item_schema:answer_index_out_of_range": json.dumps(_item() | {"answer_index": 4}),
        "item_schema:answer_index_not_int": json.dumps(_item() | {"answer_index": True}),
        "item_schema:empty_rationale": json.dumps(_item() | {"rationale": ""}),
        "item_schema:distractor_faults_must_have_three_entries": json.dumps(
            _item() | {"distractor_faults": ["a", "b"]}
        ),
    }
    for reason, raw in cases.items():
        with pytest.raises(ValueError, match=reason):
            parse_item(raw)
