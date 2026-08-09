from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.controls import load_controls
from ecm_tqag.freeze import build_freeze
from ecm_tqag.manifest import load_corpus
from ecm_tqag.run.experiment import build_phase_plan
from ecm_tqag.run.import_extraction import (
    ImportVerificationBlocked,
    validate_import_manifest,
    verify_extraction_imports,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(
    "/media/SAS/Van/DeTai2026/TQA_Pipeline/research/work_ecm24/"
    "ecm_inputs_final_v2.json"
)
ORIGIN_RUN = ROOT / "runs" / "paid_caption_v2"
ORIGIN_FREEZE = ROOT / "runs" / "freeze_caption_v2" / "FREEZE_MANIFEST.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"


def _current(tmp_path: Path):
    freeze = build_freeze(MANIFEST)
    freeze_path = tmp_path / "current-freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    corpus = load_corpus(MANIFEST)
    controls = load_controls(CONTROLS)
    plan = build_phase_plan(
        chunk_ids=corpus.chunk_ids,
        image_count=18,
        control_ids=[row["control_id"] for row in controls["controls"]],
        freeze=freeze,
        corpus=corpus,
    )
    return freeze_path, corpus, plan


def _verify(tmp_path: Path, *, origin_run: Path = ORIGIN_RUN,
            origin_freeze: Path = ORIGIN_FREEZE, task_ids=None):
    current, corpus, plan = _current(tmp_path)
    return verify_extraction_imports(
        origin_run=origin_run,
        origin_freeze_path=origin_freeze,
        current_freeze_path=current,
        corpus=corpus,
        current_plan=plan,
        task_ids=task_ids,
    )


def test_preserved_caption_run_verifies_exactly_25_paid_extractions(tmp_path: Path) -> None:
    out = _verify(tmp_path)
    assert out["status"] == "VERIFIED"
    assert (out["selected_count"], out["verified_count"], out["rejected_count"]) == (25, 25, 0)
    assert len(out["records"]["verified"]) == 25
    validate_import_manifest(out)


def test_payload_hash_drift_is_explicitly_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "origin"
    copied.mkdir()
    copied.joinpath("responses").symlink_to(ORIGIN_RUN / "responses", target_is_directory=True)
    rows = [json.loads(line) for line in (ORIGIN_RUN / "RUN_LEDGER.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    task_id = "extract:caption:01"
    for row in rows:
        if row.get("record_type") == "CALL_TERMINAL" and row.get("task_id") == task_id:
            row["payload_sha256"] = "0" * 64
    copied.joinpath("RUN_LEDGER.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    out = _verify(tmp_path, origin_run=copied, task_ids=[task_id])
    assert out["status"] == "VERIFIED_WITH_REJECTIONS"
    assert out["verified_count"] == 0
    assert out["records"]["rejected"] == [
        {"task_id": task_id, "reason": "payload_sha256_mismatch"}
    ]


def test_missing_returned_model_is_rejected_fail_closed(tmp_path: Path) -> None:
    copied = tmp_path / "origin-model"
    copied.mkdir()
    copied.joinpath("responses").symlink_to(ORIGIN_RUN / "responses", target_is_directory=True)
    rows = [json.loads(line) for line in (ORIGIN_RUN / "RUN_LEDGER.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    task_id = "extract:caption:01"
    for row in rows:
        if row.get("record_type") == "CALL_TERMINAL" and row.get("task_id") == task_id:
            row.pop("returned_model", None)
    copied.joinpath("RUN_LEDGER.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    out = _verify(tmp_path, origin_run=copied, task_ids=[task_id])
    assert out["records"]["rejected"][0]["reason"] == "returned_model_mismatch"


def test_cross_freeze_prompt_drift_blocks_whole_import(tmp_path: Path) -> None:
    current, corpus, plan = _current(tmp_path)
    freeze = json.loads(current.read_text(encoding="utf-8"))
    freeze["prompt_templates"]["planner"] = "tampered"
    current.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ImportVerificationBlocked, match="cross_freeze_mismatch:prompt_templates"):
        verify_extraction_imports(
            origin_run=ORIGIN_RUN,
            origin_freeze_path=ORIGIN_FREEZE,
            current_freeze_path=current,
            corpus=corpus,
            current_plan=plan,
        )


def test_manifest_commitment_tamper_is_rejected(tmp_path: Path) -> None:
    out = _verify(tmp_path, task_ids=["extract:caption:01"])
    out["records"]["verified"][0]["image_index"] = 99
    with pytest.raises(ImportVerificationBlocked, match="manifest_commitment_mismatch"):
        validate_import_manifest(out)
