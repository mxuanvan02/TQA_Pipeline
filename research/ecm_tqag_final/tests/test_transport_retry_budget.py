from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.run.ledger import RunLedger
from ecm_tqag.run.transport import OpenAITransport, TransportConfig


class _Response:
    status = 200

    def __init__(self, body: dict):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


def _config(tmp_path: Path, *, retries: int = 0, url: str = "https://example.invalid/v1/chat/completions") -> TransportConfig:
    key = tmp_path / "key"
    key.write_text("secret", encoding="utf-8")
    key.chmod(0o600)
    return TransportConfig(
        provider="test",
        model="frozen-model",
        base_url=url,
        api_key_file=key,
        max_retries=retries,
    )


def test_non_retry_calls_cannot_consume_retry_reserve(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _Response({
            "model": "frozen-model",
            "choices": [{"message": {"content": "ok"}}],
        }),
    )
    ledger = RunLedger(
        tmp_path / "RUN_LEDGER.jsonl",
        freeze_sha256="f" * 64,
        cap=3,
        retry_reserve=1,
    )
    transport = OpenAITransport(ledger)
    transport.call(_config(tmp_path), {"messages": [], "nonce": 1}, metadata={"phase": "x"})
    transport.call(_config(tmp_path), {"messages": [], "nonce": 2}, metadata={"phase": "x"})
    with pytest.raises(RuntimeError, match="retry_reserve_protected"):
        transport.call(_config(tmp_path), {"messages": [], "nonce": 3}, metadata={"phase": "x"})
    assert ledger.attempts_used == 2


def test_transient_terminal_is_retried_and_resume_reuses_success(tmp_path: Path, monkeypatch) -> None:
    calls = 0

    def fake_open(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary outage")
        return _Response({
            "model": "frozen-model",
            "choices": [{"message": {"content": "ok"}}],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    ledger = RunLedger(
        tmp_path / "RUN_LEDGER.jsonl",
        freeze_sha256="f" * 64,
        cap=4,
        retry_reserve=2,
    )
    transport = OpenAITransport(ledger)
    payload = {"messages": [], "nonce": 1}
    terminal = transport.call(
        _config(tmp_path, retries=1), payload, metadata={"phase": "x"}
    )
    assert terminal["outcome"] == "OK"
    assert terminal["retry_count"] == 1
    assert ledger.attempts_used == 2

    resumed = OpenAITransport(
        RunLedger(
            tmp_path / "RUN_LEDGER.jsonl",
            freeze_sha256="f" * 64,
            cap=4,
            retry_reserve=2,
        )
    ).call(_config(tmp_path, retries=1), payload, metadata={"phase": "x"})
    assert resumed["outcome"] == "OK"
    assert calls == 2


def test_transport_rejects_non_https_url_before_reading_secret(tmp_path: Path, monkeypatch) -> None:
    cfg = _config(tmp_path, url="http://example.invalid/v1/chat/completions")
    cfg.api_key_file.unlink()
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    ledger = RunLedger(
        tmp_path / "RUN_LEDGER.jsonl",
        freeze_sha256="f" * 64,
        cap=1,
        retry_reserve=0,
    )
    with pytest.raises(ValueError, match="base_url_must_be_https"):
        OpenAITransport(ledger).call(cfg, {"messages": []}, metadata={"phase": "x"})
    assert called is False
    assert ledger.attempts_used == 0
