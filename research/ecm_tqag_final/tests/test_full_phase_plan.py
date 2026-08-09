from __future__ import annotations

from pathlib import Path

from ecm_tqag.controls import load_controls
from ecm_tqag.manifest import load_corpus
from ecm_tqag.run.experiment import build_phase_plan
from ecm_tqag.run.offline import run_offline_plan

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("/media/SAS/Van/DeTai2026/TQA_Pipeline/research/work_ecm24/ecm_inputs_final_v2.json")


def _full_plan() -> dict:
    corpus = load_corpus(MANIFEST)
    controls = load_controls(ROOT / "fixtures" / "sensitivity_controls.json")
    return build_phase_plan(
        chunk_ids=corpus.chunk_ids,
        image_count=sum(len(row["evidence"]["images"]) for row in corpus.tlv),
        control_ids=[row["control_id"] for row in controls["controls"]],
    )


def test_frozen_plan_has_all_phases_and_call_parity() -> None:
    plan = _full_plan()
    assert plan["counts"] == {
        "role_smoke": 6,
        "extraction": 54,
        "construction": 128,
        "sensitivity_floor": 40,
        "secondary_probes": 160,
        "image_audit": 18,
        "judging": 80,
    }
    assert len(plan["tasks"]) == 482
    assert sum(t["calls"] == 0 for t in plan["tasks"]) == 16
    assert all(
        t["arm"] == "gates_off" and t["deterministic_rescore"] is True
        for t in plan["tasks"] if t["calls"] == 0
    )


def test_full_plan_offline_rehearsal_is_network_free_and_resumable(tmp_path: Path) -> None:
    plan = _full_plan()
    first = run_offline_plan(plan, tmp_path, freeze_sha256="a" * 64, stop_after=7)
    assert first["newly_completed"] == 7
    assert first["http_attempts"] == 0
    second = run_offline_plan(plan, tmp_path, freeze_sha256="a" * 64)
    assert second["complete"] is True
    assert second["completed"] == 482
    assert second["skipped_existing"] == 7
    assert second["http_attempts"] == 0
