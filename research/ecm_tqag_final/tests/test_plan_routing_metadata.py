from __future__ import annotations

from pathlib import Path

from ecm_tqag.controls import load_controls
from ecm_tqag.manifest import load_corpus
from ecm_tqag.run.experiment import build_phase_plan

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"


def test_paid_tasks_carry_deterministic_input_routing_metadata() -> None:
    corpus = load_corpus(MANIFEST)
    controls = load_controls(CONTROLS)
    plan = build_phase_plan(
        chunk_ids=corpus.chunk_ids,
        image_count=sum(len(row["evidence"]["images"]) for row in corpus.tlv),
        control_ids=[row["control_id"] for row in controls["controls"]],
    )
    extraction = [t for t in plan["tasks"] if t["phase"] == "extraction"]
    assert len(extraction) == 54
    assert {t["extraction_kind"] for t in extraction} == {"graph", "caption", "ocr_graph"}
    assert {t["image_index"] for t in extraction} == set(range(1, 19))
    assert all(isinstance(t["image_sha256"], str) and len(t["image_sha256"]) == 64 for t in extraction)

    construction = [t for t in plan["tasks"] if t["phase"] == "construction" and t["calls"]]
    assert construction
    assert all(t["source_condition"] in {"T", "TLV"} for t in construction)
    assert all(isinstance(t["input_fingerprint"], str) and len(t["input_fingerprint"]) == 64 for t in construction)
