from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.freeze import build_freeze
from ecm_tqag.io import sha256_file
from ecm_tqag.run.live import (
    build_extraction_loader,
    build_judging_frame_from_results,
    evaluate_sensitivity_results,
    paid_ledger_limits,
)
from ecm_tqag.run.experiment import build_phase_plan
from ecm_tqag.manifest import load_corpus

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"


def _write_result(root: Path, task_id: str, body: dict) -> None:
    import hashlib
    digest = hashlib.sha256(json.dumps(task_id, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    path = root / "results" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def test_paid_ledger_subtracts_verified_smoke_attempts() -> None:
    freeze = build_freeze(MANIFEST)
    cap, reserve = paid_ledger_limits(freeze, {"http_attempts_used": 6})
    assert cap == freeze["full_call_budget"]["current_ledger_cap"] - 6
    assert reserve == freeze["full_call_budget"]["retry_reserve"]
    with pytest.raises(ValueError, match="smoke_attempt_count"):
        paid_ledger_limits(freeze, {"http_attempts_used": 5})


def test_extraction_loader_merges_all_images_for_chunk(tmp_path: Path) -> None:
    corpus = load_corpus(MANIFEST)
    package = next(row for row in corpus.tlv if len(row["evidence"]["images"]) >= 2)
    global_index = 0
    indexes = []
    expected_captions = []
    for row in corpus.tlv:
        for image in sorted(row["evidence"]["images"], key=lambda x: x["declared_order"]):
            global_index += 1
            if row["chunk_id"] == package["chunk_id"]:
                indexes.append(global_index)
                expected_captions.append(f"caption {image['declared_order']}")
                _write_result(tmp_path, f"extract:caption:{global_index:02d}", {
                    "interface": {"caption": f"caption {image['declared_order']}",
                                  "relations": [f"relation {image['declared_order']}"]}
                })
    loader = build_extraction_loader(corpus=corpus, run_dir=tmp_path)
    merged = loader({"chunk_id": package["chunk_id"], "extraction_kind": "caption"})
    assert merged["interface"]["image_count"] == len(indexes)
    assert merged["interface"]["caption"] == " ".join(expected_captions)
    assert indexes


def test_sensitivity_gate_uses_first_replicate_and_agreement() -> None:
    rows = []
    for family in ("answerer_a", "answerer_b"):
        for i in range(10):
            rows.append({"answerer_role": family,
                         "control_type": "positive_visual" if i < 5 else "negative_text_sufficient",
                         "correct": [True, True], "replicate_agreement": True})
    verdict = evaluate_sensitivity_results(rows)
    assert verdict["passed"] is True
    assert verdict["delta_perm_reportable"] is True


def test_judging_frame_is_built_only_from_complete_5x16_pool(tmp_path: Path) -> None:
    freeze = build_freeze(MANIFEST)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    corpus = load_corpus(MANIFEST)
    plan = build_phase_plan(chunk_ids=corpus.chunk_ids, image_count=18,
                            control_ids=[f"c{i}" for i in range(10)], freeze=freeze,
                            corpus=corpus)
    for task in plan["tasks"]:
        if task["phase"] != "construction" or task["arm"] == "gates_off":
            continue
        if task["construction_stage"] not in {"realizer", "direct"}:
            continue
        _write_result(tmp_path, task["task_id"], {
            "status": "PARSED", "arm": task["arm"], "chunk_id": task["chunk_id"],
            "item": {"question": "Q?", "choices": ["A", "B", "C", "D"], "answer_index": 0}
        })
    frame = build_judging_frame_from_results(
        plan=plan, run_dir=tmp_path, freeze_sha256=sha256_file(freeze_path))
    assert frame["frame_size"] == 40
    assert len(frame["private_frame"]) == 40
    assert all("item" in row for row in frame["private_frame"])


def test_extraction_loader_propagates_schema_rejection_without_partial_merge(tmp_path: Path) -> None:
    corpus = load_corpus(MANIFEST)
    first_chunk = corpus.tlv[0]["chunk_id"]
    _write_result(tmp_path, "extract:graph:01", {
        "status": "SCHEMA_REJECTED", "reason": "grid_bbox_out_of_range",
        "calls_used": 1, "response_sha256": "a" * 64})
    loader = build_extraction_loader(corpus=corpus, run_dir=tmp_path)
    value = loader({"chunk_id": first_chunk, "extraction_kind": "graph"})
    assert value["status"] == "NOT_APPLICABLE"
    assert value["reason"] == "upstream_unavailable"
    assert value["source_reason"] == "schema_rejected"
    assert value["source_task_ids"] == ["extract:graph:01"]


def test_judging_frame_uses_complete_common_chunks_and_keeps_40_items(tmp_path: Path) -> None:
    freeze = build_freeze(MANIFEST)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    corpus = load_corpus(MANIFEST)
    plan = build_phase_plan(chunk_ids=corpus.chunk_ids, image_count=18,
                            control_ids=[f"c{i}" for i in range(10)], freeze=freeze, corpus=corpus)
    dropped = corpus.chunk_ids[0]
    for task in plan["tasks"]:
        if (task["phase"] != "construction" or task["arm"] == "gates_off"
                or task["construction_stage"] not in {"realizer", "direct"}):
            continue
        if task["chunk_id"] == dropped and task["arm"] == "full":
            body = {"status": "NOT_APPLICABLE", "reason": "upstream_unavailable",
                    "arm": task["arm"], "chunk_id": task["chunk_id"]}
        else:
            body = {"status": "PARSED", "arm": task["arm"], "chunk_id": task["chunk_id"],
                    "item": {"question": "Q?", "choices": ["A", "B", "C", "D"], "answer_index": 0}}
        _write_result(tmp_path, task["task_id"], body)
    frame = build_judging_frame_from_results(plan=plan, run_dir=tmp_path,
                                              freeze_sha256=sha256_file(freeze_path))
    assert frame["frame_size"] == 40
    assert frame["candidate_pool_size"] == 75
    assert frame["common_chunk_count"] == 15
    assert dropped in frame["dropped_chunks"]
