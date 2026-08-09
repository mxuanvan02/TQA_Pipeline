from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.freeze import build_freeze
from ecm_tqag.io import sha256_file
from ecm_tqag.run.paid import preflight_paid_run

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"


def _files(tmp_path: Path) -> tuple[Path, Path]:
    freeze_path = tmp_path / "FREEZE_MANIFEST.json"
    freeze_path.write_text(
        json.dumps(build_freeze(MANIFEST), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    freeze_sha = sha256_file(freeze_path)
    smoke_path = tmp_path / "SMOKE_RESULTS.json"
    smoke_path.write_text(
        json.dumps(
            {
                "schema": "ecm-tqag.role-smoke.v1",
                "freeze_sha256": freeze_sha,
                "status": "PASS",
                "passed_roles": [
                    "answerer_a",
                    "answerer_b",
                    "generator",
                    "image_auditor",
                    "model_judge_a",
                    "model_judge_b",
                ],
                "http_attempts_used": 6,
                "results": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return freeze_path, smoke_path


def test_paid_preflight_binds_freeze_smoke_corpus_controls_and_budget(tmp_path: Path) -> None:
    freeze_path, smoke_path = _files(tmp_path)
    out = preflight_paid_run(
        freeze_path=freeze_path,
        smoke_path=smoke_path,
        manifest_path=MANIFEST,
        controls_path=CONTROLS,
        execute=True,
    )
    assert out["status"] == "AUTHORIZED"
    assert out["freeze_sha256"] == sha256_file(freeze_path)
    assert out["chunk_count"] == 16
    assert out["image_count"] == 18
    assert out["control_count"] == 10
    assert out["phase_counts"] == {
        "role_smoke": 6,
        "extraction": 54,
        "construction": 128,
        "sensitivity_floor": 40,
        "secondary_probes": 160,
        "image_audit": 18,
        "judging": 80,
    }
    assert out["current_ledger_cap"] == 431


def test_paid_preflight_requires_execute_without_network(tmp_path: Path) -> None:
    freeze_path, smoke_path = _files(tmp_path)
    with pytest.raises(ValueError, match="execute_flag_missing"):
        preflight_paid_run(
            freeze_path=freeze_path,
            smoke_path=smoke_path,
            manifest_path=MANIFEST,
            controls_path=CONTROLS,
            execute=False,
        )


def test_paid_preflight_rejects_smoke_from_other_freeze(tmp_path: Path) -> None:
    freeze_path, smoke_path = _files(tmp_path)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["freeze_sha256"] = "0" * 64
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke_freeze_mismatch"):
        preflight_paid_run(
            freeze_path=freeze_path,
            smoke_path=smoke_path,
            manifest_path=MANIFEST,
            controls_path=CONTROLS,
            execute=True,
        )


def test_paid_preflight_rejects_incomplete_smoke_roster(tmp_path: Path) -> None:
    freeze_path, smoke_path = _files(tmp_path)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["passed_roles"].remove("answerer_b")
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke_roster_mismatch"):
        preflight_paid_run(
            freeze_path=freeze_path,
            smoke_path=smoke_path,
            manifest_path=MANIFEST,
            controls_path=CONTROLS,
            execute=True,
        )
