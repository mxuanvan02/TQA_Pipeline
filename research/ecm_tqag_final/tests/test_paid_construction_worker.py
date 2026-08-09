from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ecm_tqag.freeze import build_freeze
from ecm_tqag.manifest import load_corpus
from ecm_tqag.run.experiment import build_phase_plan
from ecm_tqag.run.paid_worker import PaidPhaseWorker, PaidWorkerBlocked
from ecm_tqag.run.transport import TransportConfig

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"


class FakeTransport:
    def __init__(self, bodies: list[dict[str, Any]]):
        self.bodies = list(bodies)
        self.calls: list[tuple[TransportConfig, dict[str, Any], dict[str, Any]]] = []

    def call(self, config, payload, *, metadata):
        self.calls.append((config, payload, metadata))
        return {"outcome": "OK", "response_path": "responses/fake.json", "response_sha256": "a" * 64}


def _setup(tmp_path: Path, bodies: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]] | None = None):
    corpus = load_corpus(MANIFEST)
    freeze = build_freeze(MANIFEST)
    key = tmp_path / "key"
    key.write_text("x", encoding="utf-8")
    key.chmod(0o600)
    role = freeze["roles"]["generator"]
    config = TransportConfig(
        provider=role["provider"], model=role["model"],
        base_url="https://x.invalid", api_key_file=key,
    )
    transport = FakeTransport(bodies)
    artifact_map = dict(artifacts or {})
    worker = PaidPhaseWorker(
        corpus=corpus, freeze=freeze, transport=transport,
        role_configs={"generator": config}, response_root=tmp_path,
        response_loader=lambda _root, _terminal: transport.bodies.pop(0),
        artifact_loader=lambda task_id: artifact_map[task_id],
    )
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    tasks = build_phase_plan(
        chunk_ids=corpus.chunk_ids, image_count=18,
        control_ids=[x["control_id"] for x in controls["controls"]],
        freeze=freeze, corpus=corpus,
    )["tasks"]
    return corpus, transport, worker, tasks


def test_construction_plan_has_explicit_stage_and_parent_links(tmp_path: Path):
    _corpus, _transport, _worker, tasks = _setup(tmp_path, [])
    construction = [t for t in tasks if t["phase"] == "construction"]
    assert len(construction) == 144  # 128 HTTP calls plus 16 free rescoring cells
    assert sum(t["calls"] for t in construction) == 128
    assert {t["construction_stage"] for t in construction} == {
        "planner", "realizer", "direct", "rescore"
    }
    planners = [t for t in construction if t["construction_stage"] == "planner"]
    realizers = [t for t in construction if t["construction_stage"] == "realizer"]
    assert len(planners) == len(realizers) == 48
    assert all(t["parent_task_id"] is None for t in planners)
    assert all(isinstance(t["parent_task_id"], str) for t in realizers)
    assert all(t["parent_task_id"].replace(":realizer", ":planner") in {p["task_id"] for p in planners}
               for t in realizers)


def test_gates_off_reuses_full_item_and_makes_no_call(tmp_path: Path):
    corpus = load_corpus(MANIFEST)
    chunk = corpus.chunk_ids[0]
    full_id = f"construct:full:{chunk}:realizer"
    full = {
        "status": "PARSED", "arm": "full", "chunk_id": chunk,
        "item": {"question": "Q?", "choices": ["A", "B", "C", "D"],
                 "answer_index": 0, "rationale": "R", "distractor_faults": ["b", "c", "d"]},
        "gates": {"G-CUE": {"ok": True}, "G-SUPPORT": {"ok": True},
                  "G-UNIQUE": {"ok": True}, "CONTRACT": {"ok": True}, "SEAL": {"ok": True}},
    }
    _corpus, transport, worker, tasks = _setup(tmp_path, [], {full_id: full})
    task = next(t for t in tasks if t["task_id"] == f"construct:gates_off:{chunk}:rescore")
    result = worker(task)
    assert result["calls_used"] == 0
    assert result["status"] == "DETERMINISTIC_RESCORE"
    assert result["source_task_id"] == full_id
    assert result["item"] == full["item"]
    assert transport.calls == []


def test_gates_off_blocks_if_full_result_is_missing_or_not_parsed(tmp_path: Path):
    corpus = load_corpus(MANIFEST)
    chunk = corpus.chunk_ids[0]
    _corpus, transport, worker, tasks = _setup(tmp_path, [])
    task = next(t for t in tasks if t["task_id"] == f"construct:gates_off:{chunk}:rescore")
    with pytest.raises(PaidWorkerBlocked, match="source_artifact_missing"):
        worker(task)
    assert transport.calls == []
