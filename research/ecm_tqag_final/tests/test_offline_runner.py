from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ecm_tqag.run.offline import run_offline_plan


def _plan() -> dict:
    return {
        "schema": "ecm-tqag.phase-plan.v1",
        "tasks": [
            {"phase": "extraction", "task_id": "extract:graph:01", "calls": 1},
            {"phase": "construction", "task_id": "construct:full:c01:1", "calls": 1},
            {"phase": "judging", "task_id": "judge:a:01", "calls": 1},
        ],
    }


def test_offline_runner_is_resumable_and_does_not_make_http_calls(tmp_path: Path) -> None:
    first = run_offline_plan(_plan(), tmp_path, freeze_sha256="a" * 64, stop_after=2)
    assert first["completed"] == 2
    assert first["http_attempts"] == 0

    second = run_offline_plan(_plan(), tmp_path, freeze_sha256="a" * 64)
    assert second["completed"] == 3
    assert second["skipped_existing"] == 2
    assert second["http_attempts"] == 0
    records = [json.loads(line) for line in (tmp_path / "OFFLINE_RUN.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert {r["record_type"] for r in records} == {"TASK_TERMINAL"}
    assert all(len(r["result_sha256"]) == 64 for r in records)
    assert os.stat(tmp_path / "OFFLINE_RUN.jsonl").st_mode & 0o777 == 0o600


def test_offline_runner_blocks_freeze_mismatch(tmp_path: Path) -> None:
    run_offline_plan(_plan(), tmp_path, freeze_sha256="b" * 64)
    with pytest.raises(ValueError, match="BLOCKED_OFFLINE:freeze_mismatch"):
        run_offline_plan(_plan(), tmp_path, freeze_sha256="c" * 64)


def test_offline_runner_blocks_tampered_sidecar_on_resume(tmp_path: Path) -> None:
    run_offline_plan(_plan(), tmp_path, freeze_sha256="d" * 64, stop_after=1)
    record = json.loads((tmp_path / "OFFLINE_RUN.jsonl").read_text().splitlines()[0])
    sidecar = tmp_path / record["result_path"]
    sidecar.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="BLOCKED_OFFLINE:sidecar_hash_mismatch"):
        run_offline_plan(_plan(), tmp_path, freeze_sha256="d" * 64)


def test_offline_runner_blocks_checkpoint_metadata_drift(tmp_path: Path) -> None:
    run_offline_plan(_plan(), tmp_path, freeze_sha256="e" * 64, stop_after=1)
    path = tmp_path / "OFFLINE_RUN.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    record["phase"] = "wrong"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="BLOCKED_OFFLINE:checkpoint_metadata_mismatch"):
        run_offline_plan(_plan(), tmp_path, freeze_sha256="e" * 64)
