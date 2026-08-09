from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.controls import (
    control_image_data_url,
    controls_commitment,
    load_controls,
    render_control_png,
)
from ecm_tqag.evaluation import (
    answerer_request,
    image_audit_request,
    judge_request,
    parse_answer,
    parse_image_audit,
    parse_judgement,
)
from ecm_tqag.freeze import build_freeze
from ecm_tqag.run.experiment import activate_secondary, build_phase_plan

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"


def _item() -> dict:
    return {"question": "Câu nào đúng?", "choices": ["A", "B", "C", "D"],
            "answer_index": 0, "rationale": "A đúng.",
            "distractor_faults": ["B sai", "C sai", "D sai"]}


def test_controls_are_fixed_balanced_and_render_deterministically() -> None:
    obj = load_controls(CONTROLS)
    rows = obj["controls"]
    assert len(rows) == 10
    assert [r["type"] for r in rows].count("positive_visual") == 5
    assert [r["type"] for r in rows].count("negative_text_sufficient") == 5
    assert render_control_png(rows[0]) == render_control_png(rows[0])
    assert control_image_data_url(rows[0]).startswith("data:image/png;base64,")
    assert controls_commitment(obj) == controls_commitment(load_controls(CONTROLS))


def test_freeze_commits_to_control_fixture_and_rendered_images() -> None:
    frozen = build_freeze(MANIFEST)
    expected = controls_commitment(load_controls(CONTROLS))
    assert frozen["sensitivity_controls"] == {
        "schema": "ecm-tqag.sensitivity-controls.v1",
        "count": 10,
        "commitment_sha256": expected,
    }


def test_answer_audit_and_blinded_judge_contracts() -> None:
    item = _item()
    req = answerer_request(item, text="Nguồn", image_data_urls=[])
    assert "model" not in req
    assert parse_answer('{"answer_index":0,"abstain":false,"confidence":0.9}')["answer_index"] == 0
    assert parse_answer('{"answer_index":null,"abstain":true,"confidence":0.2}')["abstain"] is True
    with pytest.raises(ValueError, match="answer_schema:keys"):
        parse_answer('{"answer_index":0,"abstain":false,"confidence":1,"extra":1}')

    audit = image_audit_request({"nodes": []}, "data:image/png;base64,AA==")
    assert "model" not in audit
    assert parse_image_audit('{"supported":true,"relation_supported":false,"notes":"x"}')["supported"]

    judge = judge_request(item)
    prompt = judge["messages"][0]["content"]
    assert "source_support" not in prompt
    assert "answer_index" not in prompt
    assert "rationale" not in prompt
    parsed = parse_judgement(
        '{"answerability":4,"single_best_answer":5,"clarity":3,"notes":"x"}'
    )
    assert parsed["clarity"] == 3
    with pytest.raises(ValueError, match="judge_schema:keys"):
        parse_judgement(
            '{"answerability":4,"single_best_answer":5,"source_support":3,"notes":"x"}'
        )


def test_phase_plan_exact_parity_and_floor_blocks_secondary() -> None:
    controls = load_controls(CONTROLS)
    ids = [row["control_id"] for row in controls["controls"]]
    chunks = [f"chunk-{i:02d}" for i in range(16)]
    plan = build_phase_plan(chunk_ids=chunks, image_count=18, control_ids=ids)
    assert plan["counts"] == {
        "role_smoke": 6,
        "extraction": 54,
        "construction": 128,
        "sensitivity_floor": 40,
        "secondary_probes": 160,
        "image_audit": 18,
        "judging": 80,
    }
    assert activate_secondary(plan, floor_passed=False) == []
    active = activate_secondary(plan, floor_passed=True)
    assert len(active) == 160
    assert {row["task_id"].split(":")[-1] for row in active} == {
        "control", "control_replicate", "label_permutation", "block_shuffle",
        "text_anchor_removal",
    }
