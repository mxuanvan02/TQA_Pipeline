from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.run.ledger import RunLedger
from ecm_tqag.run.transport import BudgetExceeded, OpenAITransport, TransportConfig


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "model": "wanted-model",
            "choices": [{"message": {"content": "{\"ok\":true}"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }).encode()


def _config(tmp_path: Path) -> TransportConfig:
    key = tmp_path / "key"
    key.write_text("***", encoding="utf-8")
    key.chmod(0o600)
    return TransportConfig(
        provider="test", model="wanted-model", base_url="https://example.invalid/v1/chat/completions",
        api_key_file=key, allow_fallbacks=False,
    )


def test_transport_is_idempotent_and_append_only(tmp_path, monkeypatch):
    calls = []

    def fake_open(req, timeout):
        calls.append(req)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    ledger = RunLedger(tmp_path / "RUN_LEDGER.jsonl", freeze_sha256="f" * 64, cap=2, retry_reserve=0)
    transport = OpenAITransport(ledger)
    payload = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0, "max_tokens": 10}
    one = transport.call(_config(tmp_path), payload, metadata={"phase": "smoke", "role": "generator"})
    two = transport.call(_config(tmp_path), payload, metadata={"phase": "smoke", "role": "generator"})
    assert one["idempotency_key"] == two["idempotency_key"]
    assert len(calls) == 1
    records = [json.loads(x) for x in (tmp_path / "RUN_LEDGER.jsonl").read_text().splitlines()]
    assert [r["record_type"] for r in records] == ["CALL_STARTED", "CALL_TERMINAL"]
    assert "sk-ultra-secret" not in json.dumps(records)


def test_every_http_attempt_counts_against_hard_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse())
    ledger = RunLedger(tmp_path / "RUN_LEDGER.jsonl", freeze_sha256="f" * 64, cap=1, retry_reserve=0)
    tx = OpenAITransport(ledger)
    tx.call(_config(tmp_path), {"messages": [], "nonce": 1}, metadata={"phase": "x", "role": "r"})
    with pytest.raises(BudgetExceeded, match="BLOCKED_BUDGET"):
        tx.call(_config(tmp_path), {"messages": [], "nonce": 2}, metadata={"phase": "x", "role": "r"})


def test_openrouter_request_disables_fallbacks(tmp_path, monkeypatch):
    bodies = []

    def fake_open(req, timeout):
        bodies.append(json.loads(req.data))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    ledger = RunLedger(tmp_path / "RUN_LEDGER.jsonl", freeze_sha256="f" * 64, cap=1, retry_reserve=0)
    tx = OpenAITransport(ledger)
    cfg = _config(tmp_path)
    cfg = TransportConfig(**{**cfg.__dict__, "provider": "openrouter"})
    tx.call(cfg, {"messages": []}, metadata={"phase": "x", "role": "r"})
    assert bodies[0]["provider"]["allow_fallbacks"] is False


def test_response_model_mismatch_is_terminal_identity_violation(tmp_path, monkeypatch):
    class Wrong(FakeResponse):
        def read(self):
            return json.dumps({"model": "other-model", "choices": [{"message": {"content": "x"}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Wrong())
    ledger = RunLedger(tmp_path / "RUN_LEDGER.jsonl", freeze_sha256="f" * 64, cap=1, retry_reserve=0)
    out = OpenAITransport(ledger).call(
        _config(tmp_path), {"messages": []}, metadata={"phase": "x", "role": "r"}
    )
    assert out["outcome"] == "MODEL_IDENTITY_VIOLATION"
