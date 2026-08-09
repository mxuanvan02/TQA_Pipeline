from __future__ import annotations

import json

import pytest

from ecm_tqag.prompts import planner_prompt, planner_program


def _valid_payload() -> dict:
    return {
        "motif": "diagram_text_reconcile",
        "steps": [
            {
                "op": "locate",
                "anchor": "Cơ quan A ở tầng trên",
                "result": "Cơ quan A nằm trên cơ quan B",
                "visual_node": 1,
                "bbox": [0.10, 0.10, 0.40, 0.30],
            },
            {
                "op": "locate",
                "anchor": "cơ quan cấp trên hướng dẫn cơ quan cấp dưới",
                "result": "cấp trên hướng dẫn cấp dưới",
            },
            {
                "op": "reconcile",
                "anchor": None,
                "result": "Cơ quan A có vai trò hướng dẫn cơ quan B",
            },
        ],
    }


def test_planner_returns_typed_program_consumable_by_compiler() -> None:
    raw = json.dumps(_valid_payload(), ensure_ascii=False)
    assert planner_program(raw) == _valid_payload()


def test_planner_abstention_is_preserved() -> None:
    assert planner_program('{"abstain":"no_cross_modal_dependency"}') == {
        "abstain": "no_cross_modal_dependency"
    }


def test_planner_rejects_legacy_or_unknown_schema() -> None:
    with pytest.raises(ValueError, match="planner_schema"):
        planner_program('{"relation_index":0,"text_anchor":"x","text_result":"y","derived_result":"z"}')
    bad = _valid_payload() | {"chunk_id": "secret"}
    with pytest.raises(ValueError, match="planner_schema"):
        planner_program(json.dumps(bad))


def test_graph_and_caption_share_identical_typed_program_instructions() -> None:
    a = planner_prompt("T", {"section": "s"}, {"graph_type": "FLOW"}, interface_kind="closed_graph")
    b = planner_prompt("T", {"section": "s"}, {"caption": "flow"}, interface_kind="caption")
    prefix_a = a.replace("INTERFACE_KIND=closed_graph", "INTERFACE_KIND=<KIND>").split("INTERFACE=", 1)[0]
    prefix_b = b.replace("INTERFACE_KIND=caption", "INTERFACE_KIND=<KIND>").split("INTERFACE=", 1)[0]
    assert prefix_a == prefix_b
    assert '"motif"' in a and '"steps"' in a
