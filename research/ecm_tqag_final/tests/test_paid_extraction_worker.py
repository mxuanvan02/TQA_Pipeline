from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ecm_tqag.freeze import build_freeze
from ecm_tqag.interfaces import CAPTION_PROMPT, OCR_ASSISTED_SUFFIX
from ecm_tqag.manifest import load_corpus
from ecm_tqag.run.experiment import build_phase_plan
from ecm_tqag.run.paid_worker import PaidPhaseWorker, PaidWorkerBlocked
from ecm_tqag.run.transport import TransportConfig
from ecm_tqag.structure_reader import PROMPT

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT.parent / "work_ecm24" / "ecm_inputs_final_v2.json"
CONTROLS = ROOT / "fixtures" / "sensitivity_controls.json"


class FakeTransport:
    def __init__(self, bodies: list[dict[str, Any]]):
        self.bodies = list(bodies)
        self.calls: list[tuple[TransportConfig, dict[str, Any], dict[str, Any]]] = []

    def call(self, config: TransportConfig, payload: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((config, payload, metadata))
        return {"outcome": "OK", "response_path": "responses/fake.json", "response_sha256": "a" * 64}


def _graph_body() -> dict[str, Any]:
    graph = {
        "graph_type": "FLOW",
        "nodes": [{"id": "n1", "label": "A", "level": 0,
                   "bbox": [10, 10, 100, 100]}],
        "edges": [],
        "confidence": 0.9,
    }
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(graph)}}]}


def _out_of_grid_graph_body() -> dict[str, Any]:
    graph = {
        "graph_type": "FLOW",
        "nodes": [{"id": "n1", "label": "A", "level": 0,
                   "bbox": [752, 677, 1008, 772]}],
        "edges": [],
        "confidence": 0.9,
    }
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(graph)}}]}


def _caption_body() -> dict[str, Any]:
    caption = {"caption": "Khối A đứng riêng", "relations": ["A đứng riêng"], "confidence": 0.8}
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(caption, ensure_ascii=False)}}]}


def _overlong_caption_body() -> dict[str, Any]:
    caption = {
        "caption": " ".join(["từ"] * 61),
        "relations": ["A đứng riêng"],
        "confidence": 0.8,
    }
    return {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(caption, ensure_ascii=False)
    }}]}


def _setup(tmp_path: Path, bodies: list[dict[str, Any]]):
    corpus = load_corpus(MANIFEST)
    freeze = build_freeze(MANIFEST)
    key = tmp_path / "key"
    key.write_text("secret", encoding="utf-8")
    key.chmod(0o600)
    role = freeze["roles"]["generator"]
    config = TransportConfig(
        provider=role["provider"], model=role["model"],
        base_url="https://example.invalid/v1/chat/completions", api_key_file=key,
        allow_fallbacks=False,
    )
    transport = FakeTransport(bodies)
    worker = PaidPhaseWorker(
        corpus=corpus, freeze=freeze, transport=transport,
        role_configs={"generator": config}, response_root=tmp_path,
        response_loader=lambda _root, _terminal: transport.bodies.pop(0),
    )
    return corpus, freeze, transport, worker


def _tasks(corpus, freeze):
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    return build_phase_plan(
        chunk_ids=corpus.chunk_ids,
        image_count=sum(len(r["evidence"]["images"]) for r in corpus.tlv),
        control_ids=[r["control_id"] for r in controls["controls"]],
        freeze=freeze, corpus=corpus,
    )["tasks"]


def test_extraction_worker_routes_three_interfaces_and_parses_closed_schemas(tmp_path: Path) -> None:
    corpus, freeze, transport, worker = _setup(tmp_path, [_graph_body(), _caption_body(), _graph_body()])
    tasks = _tasks(corpus, freeze)
    selected = [next(t for t in tasks if t["task_id"] == f"extract:{kind}:01")
                for kind in ("graph", "caption", "ocr_graph")]

    results = [worker(task) for task in selected]

    assert [r["extraction_kind"] for r in results] == ["graph", "caption", "ocr_graph"]
    assert all(r["status"] == "OK" and r["calls_used"] == 1 for r in results)
    assert results[0]["interface"]["graph_type"] == "FLOW"
    assert results[1]["interface"] == {"caption": "Khối A đứng riêng", "relations": ["A đứng riêng"]}
    assert results[2]["interface"]["graph_type"] == "FLOW"
    prompts = [call[1]["messages"][0]["content"][0]["text"] for call in transport.calls]
    assert prompts[0] == PROMPT
    assert prompts[1] == CAPTION_PROMPT
    assert PROMPT in prompts[2] and OCR_ASSISTED_SUFFIX in prompts[2] and "OCR_TEXT=" in prompts[2]
    assert all(call[2]["phase"] == "extraction" for call in transport.calls)
    assert all(call[2]["role"] == "generator" for call in transport.calls)


def test_extraction_worker_rejects_roster_mismatch_before_transport(tmp_path: Path) -> None:
    corpus = load_corpus(MANIFEST)
    freeze = build_freeze(MANIFEST)
    key = tmp_path / "key"
    key.write_text("secret", encoding="utf-8")
    key.chmod(0o600)
    wrong = TransportConfig(
        provider="omniproxy", model="wrong", base_url="https://example.invalid/v1/chat/completions",
        api_key_file=key,
    )
    transport = FakeTransport([])
    with pytest.raises(PaidWorkerBlocked, match="role_config_mismatch:generator"):
        PaidPhaseWorker(
            corpus=corpus, freeze=freeze, transport=transport,
            role_configs={"generator": wrong}, response_root=tmp_path,
            response_loader=lambda _root, _terminal: {},
        )
    assert transport.calls == []


def test_extraction_worker_fails_closed_on_transport_or_schema_error(tmp_path: Path) -> None:
    corpus, freeze, transport, worker = _setup(tmp_path, [{"choices": [{"message": {"content": "not-json"}}]}])
    task = next(t for t in _tasks(corpus, freeze) if t["task_id"] == "extract:graph:01")
    with pytest.raises(PaidWorkerBlocked, match="extraction_parse_failed"):
        worker(task)
    assert len(transport.calls) == 1


def test_extraction_worker_checkpoints_paid_bbox_schema_rejection(tmp_path: Path) -> None:
    corpus, freeze, transport, worker = _setup(tmp_path, [_out_of_grid_graph_body()])
    task = next(t for t in _tasks(corpus, freeze) if t["task_id"] == "extract:graph:01")
    result = worker(task)
    assert result["status"] == "SCHEMA_REJECTED"
    assert result["reason"] == "grid_bbox_out_of_range"
    assert result["calls_used"] == 1
    assert result["extraction_kind"] == "graph"
    assert result["image_index"] == 1
    assert len(result["response_sha256"]) == 64
    assert "interface" not in result
    assert len(transport.calls) == 1


def test_extraction_worker_checkpoints_overlong_caption_as_schema_rejection(tmp_path: Path) -> None:
    corpus, freeze, transport, worker = _setup(tmp_path, [_overlong_caption_body()])
    task = next(t for t in _tasks(corpus, freeze) if t["task_id"] == "extract:caption:01")
    result = worker(task)
    assert result["status"] == "SCHEMA_REJECTED"
    assert result["reason"] == "caption_too_long:61>60"
    assert result["calls_used"] == 1
    assert result["extraction_kind"] == "caption"
    assert result["image_index"] == 1
    assert len(result["response_sha256"]) == 64
    assert "interface" not in result
    assert len(transport.calls) == 1
