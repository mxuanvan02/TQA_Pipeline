from __future__ import annotations

import json
from pathlib import Path

from ecm_tqag.freeze import build_freeze
from ecm_tqag.manifest import load_corpus
from ecm_tqag.run.experiment import build_phase_plan

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"


def test_every_evaluation_task_has_explicit_closed_routing_metadata() -> None:
    corpus = load_corpus(MANIFEST)
    freeze = build_freeze(MANIFEST)
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))["controls"]
    plan = build_phase_plan(
        chunk_ids=corpus.chunk_ids,
        image_count=18,
        control_ids=[row["control_id"] for row in controls],
        freeze=freeze,
        corpus=corpus,
    )
    tasks = plan["tasks"]

    floor = [t for t in tasks if t["phase"] == "sensitivity_floor"]
    assert len(floor) == 20 and sum(t["calls"] for t in floor) == 40
    assert all(t["answerer_role"] in {"answerer_a", "answerer_b"} for t in floor)
    assert all(t["control_id"] in {r["control_id"] for r in controls} for t in floor)
    assert all(t["replicates"] == 2 for t in floor)

    probes = [t for t in tasks if t["phase"] == "secondary_probes"]
    assert len(probes) == 160
    assert all(t["answerer_role"] in {"answerer_a", "answerer_b"} for t in probes)
    assert all(t["probe_condition"] in {
        "control", "control_replicate", "label_permutation", "block_shuffle",
        "text_anchor_removal",
    } for t in probes)
    assert all(t["parent_task_id"] == f"construct:full:{t['chunk_id']}:realizer" for t in probes)
    assert all(isinstance(t["input_fingerprint"], str) and len(t["input_fingerprint"]) == 64 for t in probes)

    audits = [t for t in tasks if t["phase"] == "image_audit"]
    assert len(audits) == 18
    assert [t["image_index"] for t in audits] == list(range(1, 19))
    assert all(isinstance(t["image_sha256"], str) and len(t["image_sha256"]) == 64 for t in audits)
    assert all(t["parent_task_id"] == f"extract:graph:{t['image_index']:02d}" for t in audits)

    judging = [t for t in tasks if t["phase"] == "judging"]
    assert len(judging) == 80
    assert all(t["judge_role"] in {"model_judge_a", "model_judge_b"} for t in judging)
    assert {t["frame_index"] for t in judging} == set(range(1, 41))
