from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecm_tqag.run.ledger import RunLedger
from ecm_tqag.run.transport import (
    OpenAITransport,
    TransportConfig,
    load_response_sidecar,
)


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps({
            "model": "frozen-model",
            "choices": [{"message": {"content": "{\"ok\":true}"}}],
        }).encode("utf-8")


def _config(tmp_path: Path) -> TransportConfig:
    key = tmp_path / "key"
    key.write_text("secret", encoding="utf-8")
    key.chmod(0o600)
    return TransportConfig(
        provider="test",
        model="frozen-model",
        base_url="https://example.invalid/v1/chat/completions",
        api_key_file=key,
    )


def test_response_sidecar_is_private_and_hash_verified(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())
    ledger = RunLedger(tmp_path / "RUN_LEDGER.jsonl", freeze_sha256="f" * 64,
                       cap=2, retry_reserve=0)
    terminal = OpenAITransport(ledger).call(
        _config(tmp_path), {"messages": []}, metadata={"phase": "x", "role": "r"}
    )
    sidecar = tmp_path / terminal["response_path"]
    assert sidecar.stat().st_mode & 0o077 == 0
    body = load_response_sidecar(tmp_path, terminal)
    assert body["model"] == "frozen-model"

    sidecar.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="response_hash_mismatch"):
        load_response_sidecar(tmp_path, terminal)


def test_response_loader_rejects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-response.json"
    outside.write_text("{}", encoding="utf-8")
    terminal = {
        "response_path": "../outside-response.json",
        "response_sha256": "0" * 64,
        "outcome": "OK",
    }
    with pytest.raises(ValueError, match="response_path_invalid"):
        load_response_sidecar(tmp_path, terminal)


def test_transport_rejects_non_private_key_before_network(tmp_path: Path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    cfg.api_key_file.chmod(0o644)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    ledger = RunLedger(tmp_path / "RUN_LEDGER.jsonl", freeze_sha256="f" * 64,
                       cap=1, retry_reserve=0)
    with pytest.raises(ValueError, match="api_key_permissions"):
        OpenAITransport(ledger).call(
            cfg, {"messages": []}, metadata={"phase": "x", "role": "r"}
        )
    assert called is False
    assert ledger.attempts_used == 0
