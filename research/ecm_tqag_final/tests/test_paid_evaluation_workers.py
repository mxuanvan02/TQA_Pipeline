from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ecm_tqag.controls import load_controls
from ecm_tqag.freeze import build_freeze
from ecm_tqag.manifest import load_corpus
from ecm_tqag.run.experiment import build_phase_plan
from ecm_tqag.run.paid_worker import PaidPhaseWorker, PaidWorkerBlocked
from ecm_tqag.run.transport import TransportConfig

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"


class FakeTransport:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.calls: list[tuple[TransportConfig, dict[str, Any], dict[str, Any]]] = []

    def call(self, config, payload, *, metadata):
        self.calls.append((config, payload, metadata))
        return {"outcome": "OK", "response_path": "responses/fake.json", "response_sha256": "a" * 64}


def _body(content: str) -> dict[str, Any]:
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def _item() -> dict[str, Any]:
    return {"question": "Câu nào đúng?", "choices": ["A", "B", "C", "D"],
            "answer_index": 0, "rationale": "A đúng", "distractor_faults": ["b", "c", "d"]}


def _setup(tmp_path: Path, contents: list[str], *, artifacts=None, frame=None):
    corpus = load_corpus(MANIFEST)
    freeze = build_freeze(MANIFEST)
    key = tmp_path / "key"
    key.write_text("secret", encoding="utf-8")
    key.chmod(0o600)
    configs = {}
    for name in ("answerer_a", "answerer_b", "image_auditor", "model_judge_a", "model_judge_b"):
        role = freeze["roles"][name]
        configs[name] = TransportConfig(provider=role["provider"], model=role["model"],
                                        base_url="https://x.invalid", api_key_file=key)
    transport = FakeTransport(contents)
    artifact_map = dict(artifacts or {})
    worker = PaidPhaseWorker(
        corpus=corpus, freeze=freeze, transport=transport, role_configs=configs,
        response_root=tmp_path,
        response_loader=lambda _root, _terminal: _body(transport.contents.pop(0)),
        artifact_loader=lambda task_id: artifact_map[task_id],
        controls=load_controls(CONTROLS),
        judging_frame=frame,
    )
    ids = [x["control_id"] for x in load_controls(CONTROLS)["controls"]]
    tasks = build_phase_plan(chunk_ids=corpus.chunk_ids, image_count=18,
                             control_ids=ids, freeze=freeze, corpus=corpus)["tasks"]
    return corpus, transport, worker, tasks


def test_sensitivity_worker_makes_exactly_two_closed_answer_calls(tmp_path: Path) -> None:
    answer = '{"answer_index":0,"abstain":false,"confidence":0.9}'
    _corpus, transport, worker, tasks = _setup(tmp_path, [answer, answer])
    task = next(t for t in tasks if t["task_id"] == "control:answerer_a:pos01")
    result = worker(task)
    assert result["status"] == "SCORED" and result["calls_used"] == 2
    assert result["control_id"] == "pos01" and result["correct"] == [True, True]
    assert result["replicate_agreement"] is True
    assert len(transport.calls) == 2
    assert all(call[2]["role"] == "answerer_a" for call in transport.calls)
    assert all(call[2]["phase"] == "sensitivity_floor" for call in transport.calls)
    assert all(any(part.get("type") == "image_url" for part in call[1]["messages"][0]["content"])
               for call in transport.calls)


def test_text_sufficient_floor_control_is_answered_without_pixels(tmp_path: Path) -> None:
    answer = '{"answer_index":0,"abstain":false,"confidence":0.9}'
    _corpus, transport, worker, tasks = _setup(tmp_path, [answer, answer])
    task = next(t for t in tasks if t["task_id"] == "control:answerer_b:neg01")
    result = worker(task)
    assert result["correct"] == [True, True]
    assert result["control_type"] == "negative_text_sufficient"
    contents = [call[1]["messages"][0]["content"] for call in transport.calls]
    assert all(not any(part.get("type") == "image_url" for part in content)
               for content in contents)


def test_image_audit_uses_graph_parent_and_frozen_image(tmp_path: Path) -> None:
    audit = '{"supported":true,"relation_supported":true,"notes":"ok"}'
    graph = {"status": "OK", "extraction_kind": "graph", "image_index": 1,
             "image_sha256": load_corpus(MANIFEST).tlv[0]["evidence"]["images"][0]["sha256"],
             "interface": {"graph_type": "FLOW", "nodes": [], "edges": []}}
    _corpus, transport, worker, tasks = _setup(
        tmp_path, [audit], artifacts={"extract:graph:01": graph})
    task = next(t for t in tasks if t["task_id"] == "audit:image:01")
    result = worker(task)
    assert result["status"] == "AUDITED" and result["calls_used"] == 1
    assert result["audit"]["supported"] is True
    assert transport.calls[0][2]["role"] == "image_auditor"


def test_judging_uses_only_preblinded_frame_payload(tmp_path: Path) -> None:
    judgement = '{"answerability":4,"single_best_answer":5,"clarity":3,"notes":"ok"}'
    frame = {"schema": "ecm-tqag.fixed-judging-frame.v1", "frame_size": 40,
             "private_frame": [{"judge_item_id": f"j{i:02d}", "item": _item()} for i in range(1, 41)]}
    _corpus, transport, worker, tasks = _setup(tmp_path, [judgement], frame=frame)
    task = next(t for t in tasks if t["task_id"] == "judge:model_judge_b:01")
    result = worker(task)
    assert result["status"] == "JUDGED" and result["judge_item_id"] == "j01"
    assert transport.calls[0][2]["role"] == "model_judge_b"
    prompt = transport.calls[0][1]["messages"][0]["content"]
    assert "answer_index" not in prompt and "rationale" not in prompt


def test_probe_is_zero_call_not_applicable_only_for_ineligible_full_item(tmp_path: Path) -> None:
    corpus = load_corpus(MANIFEST)
    chunk = corpus.chunk_ids[0]
    parent = f"construct:full:{chunk}:realizer"
    source = {"status": "PARSED", "arm": "full", "chunk_id": chunk,
              "item": _item(), "passes_confirmatory_gates": False}
    _corpus, transport, worker, tasks = _setup(tmp_path, [], artifacts={parent: source})
    task = next(t for t in tasks if t["task_id"] == f"probe:answerer_a:{chunk}:control")
    result = worker(task)
    assert result == {"status": "NOT_APPLICABLE", "reason": "full_item_not_eligible",
                      "calls_used": 0, "chunk_id": chunk,
                      "probe_condition": "control", "answerer_role": "answerer_a"}
    assert transport.calls == []


def test_evaluation_worker_blocks_missing_fixtures_before_transport(tmp_path: Path) -> None:
    corpus = load_corpus(MANIFEST)
    freeze = build_freeze(MANIFEST)
    class NoCall:
        def call(self, *args, **kwargs):
            raise AssertionError("network must not be reached")
    worker = PaidPhaseWorker(corpus=corpus, freeze=freeze, transport=NoCall(),
                             role_configs={}, response_root=tmp_path)
    with pytest.raises(PaidWorkerBlocked, match="controls_missing"):
        worker({"phase": "sensitivity_floor", "task_id": "x", "calls": 2,
                "answerer_role": "answerer_a", "control_id": "pos01", "replicates": 2})
