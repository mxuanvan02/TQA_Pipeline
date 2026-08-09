from __future__ import annotations

import inspect
import json

import pytest

from ecm_tqag import structure_reader
from ecm_tqag.interfaces import (
    CAPTION_MAX_CAPTION_WORDS,
    CAPTION_MAX_RELATIONS,
    CAPTION_PROMPT,
    caption_prompt,
    caption_request,
    interface_from_caption,
    interface_from_graph,
    ocr_assisted_graph_prompt,
    ocr_assisted_graph_static_fingerprint,
    parse_caption,
    parse_graph_interface,
)
from ecm_tqag.prompts import planner_prompt

PNG = "data:image/png;base64,AAAA"


def _caption_payload() -> dict:
    return {
        "caption": "Sơ đồ ba khối xếp dọc, khối trên có mũi tên xuống hai khối dưới",
        "relations": [
            "khối trên chỉ mũi tên xuống khối giữa",
            "khối giữa chỉ mũi tên xuống khối dưới",
        ],
        "confidence": 0.8,
    }


def _graph() -> dict:
    return {
        "graph_type": "TREE",
        "nodes": [
            {"id": "n1", "label": "Quốc hội", "level": 0, "bbox": [100, 50, 300, 120]},
            {"id": "n2", "label": "Chính phủ", "level": 1, "bbox": [100, 200, 300, 260]},
        ],
        "edges": [{"src": "n1", "dst": "n2", "kind": "ARROW", "label": ""}],
        "confidence": 0.7,
    }


# --------------------------------------------------------------- caption arm
def test_caption_extraction_is_pixels_only_by_construction() -> None:
    for fn in (caption_prompt, caption_request):
        params = list(inspect.signature(fn).parameters)
        bad = [
            p
            for p in params
            if any(tok in p.lower() for tok in structure_reader.FORBIDDEN_PARAM_TOKENS)
        ]
        assert bad == [], f"{fn.__name__} may not accept text-carrying parameters: {bad}"
    assert caption_prompt() == CAPTION_PROMPT


def test_caption_request_carries_exactly_one_image_and_no_model() -> None:
    payload = caption_request(PNG)
    assert set(payload) == {"messages", "temperature", "max_tokens", "response_format"}
    assert "model" not in payload
    assert payload["temperature"] == 0
    schema = payload["response_format"]["json_schema"]["schema"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["properties"]["relations"]["maxItems"] == CAPTION_MAX_RELATIONS
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": CAPTION_PROMPT}
    assert content[1] == {"type": "image_url", "image_url": {"url": PNG}}
    assert len(content) == 2


def test_caption_request_rejects_non_image_payload() -> None:
    with pytest.raises(ValueError, match="caption_request:invalid_image_data_url"):
        caption_request("https://example.invalid/page.png")


def test_caption_parser_accepts_closed_schema() -> None:
    payload = _caption_payload()
    assert parse_caption(json.dumps(payload, ensure_ascii=False)) == payload


def test_caption_parser_strips_fence() -> None:
    payload = _caption_payload()
    raw = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
    assert parse_caption(raw) == payload


def test_caption_parser_fails_closed() -> None:
    cases = {
        "caption_schema:invalid_json": "not json",
        "caption_schema:top_level_not_object": "[1,2]",
        "caption_schema:unexpected_keys": json.dumps(
            _caption_payload() | {"chunk_id": "c1"}, ensure_ascii=False
        ),
        "caption_schema:empty_caption": json.dumps(
            _caption_payload() | {"caption": "   "}, ensure_ascii=False
        ),
        "caption_schema:no_relations": json.dumps(
            _caption_payload() | {"relations": []}, ensure_ascii=False
        ),
        "caption_schema:invalid_relation": json.dumps(
            _caption_payload() | {"relations": ["ok relation here", ""]},
            ensure_ascii=False,
        ),
        "caption_schema:confidence_not_unit_interval": json.dumps(
            _caption_payload() | {"confidence": 1.5}, ensure_ascii=False
        ),
    }
    for reason, raw in cases.items():
        with pytest.raises(ValueError, match=reason):
            parse_caption(raw)


def test_caption_parser_rejects_page_length_transcript() -> None:
    long_caption = " ".join(["từ"] * (CAPTION_MAX_CAPTION_WORDS + 1))
    with pytest.raises(ValueError, match="caption_schema:caption_too_long"):
        parse_caption(json.dumps(_caption_payload() | {"caption": long_caption}))


def test_caption_interface_projection_drops_confidence() -> None:
    assert interface_from_caption(_caption_payload()) == {
        "caption": _caption_payload()["caption"],
        "relations": _caption_payload()["relations"],
    }


# ------------------------------------------------------- ocr-assisted reader
def test_ocr_assisted_prompt_reuses_the_blind_graph_instrument() -> None:
    prompt = ocr_assisted_graph_prompt("Điều 1. Quốc hội là cơ quan quyền lực cao nhất.")
    assert prompt.startswith(structure_reader.PROMPT)
    assert "OCR_TEXT=" in prompt
    head, tail = prompt.split("OCR_TEXT=", 1)
    assert tail == "Điều 1. Quốc hội là cơ quan quyền lực cao nhất."
    # The reader arms differ only by the OCR slot, never by the schema block.
    other = ocr_assisted_graph_prompt("khác hoàn toàn")
    assert other.split("OCR_TEXT=", 1)[0] == head
    assert ocr_assisted_graph_static_fingerprint() == ocr_assisted_graph_static_fingerprint()


def test_ocr_assisted_prompt_requires_real_ocr_text() -> None:
    for bad in ("", "   ", None, 3):
        with pytest.raises(ValueError, match="ocr_assisted_graph:"):
            ocr_assisted_graph_prompt(bad)  # type: ignore[arg-type]


def test_graph_interface_parser_shares_the_closed_graph_schema() -> None:
    graph = _graph()
    assert parse_graph_interface(json.dumps(graph, ensure_ascii=False)) == graph
    assert structure_reader.validate_graph(graph)["ok"] is True


def test_graph_interface_parser_reports_structure_reader_reasons() -> None:
    dangling = _graph()
    dangling["edges"][0]["dst"] = "n9"
    with pytest.raises(ValueError, match="graph_schema:dangling_edge_endpoint"):
        parse_graph_interface(json.dumps(dangling))
    with pytest.raises(ValueError, match="graph_schema:non_json_response"):
        parse_graph_interface("{oops")


def test_graph_interface_normalises_fixed_grid_bboxes_for_the_planner() -> None:
    iface = interface_from_graph(_graph(), width=400, height=400)
    assert set(iface) == {"graph_type", "nodes", "edges"}
    assert iface["nodes"][0]["bbox"] == [0.1, 0.05, 0.3, 0.12]
    assert all(0.0 <= v <= 1.0 for n in iface["nodes"] for v in n["bbox"])
    assert set(iface["nodes"][0]) == {"id", "label", "level", "bbox"}
    with pytest.raises(ValueError, match="graph_interface:invalid_bbox"):
        bad = _graph()
        bad["nodes"][0]["bbox"] = [-1, 0, 10, 10]
        interface_from_graph(bad, width=400, height=400)


def test_both_interfaces_feed_the_one_downstream_planner_template() -> None:
    graph_iface = interface_from_graph(_graph(), width=400, height=400)
    caption_iface = interface_from_caption(_caption_payload())
    a = planner_prompt("T", {"section": "s"}, graph_iface, interface_kind="closed_graph")
    b = planner_prompt("T", {"section": "s"}, caption_iface, interface_kind="caption")
    prefix_a = a.replace("INTERFACE_KIND=closed_graph", "INTERFACE_KIND=<KIND>").split("INTERFACE=", 1)[0]
    prefix_b = b.replace("INTERFACE_KIND=caption", "INTERFACE_KIND=<KIND>").split("INTERFACE=", 1)[0]
    assert prefix_a == prefix_b
