from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..io import canonical, sha256_bytes
from .ledger import RunLedger


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class TransportConfig:
    provider: str
    model: str
    base_url: str
    api_key_file: Path
    allow_fallbacks: bool = False
    timeout_sec: int = 180
    max_retries: int = 0


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redact(value: str, secret: str) -> str:
    text = value.replace(secret, "[REDACTED]") if secret else value
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{6,}\b", "[REDACTED_KEY]", text)
    return text[:800]


def _write_private(path: Path, payload: bytes) -> None:
    """Write a response sidecar durably with owner-only permissions."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def _validate_secret_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise ValueError("BLOCKED_EXECUTION:api_key_missing")
    if path.stat().st_mode & 0o077:
        raise ValueError("BLOCKED_EXECUTION:api_key_permissions_must_be_0600")
    secret = path.read_text(encoding="utf-8").strip()
    if not secret:
        raise ValueError("BLOCKED_EXECUTION:empty_api_key")
    return secret


def load_response_sidecar(root: Path, terminal: dict[str, Any]) -> dict[str, Any]:
    """Load a transport sidecar only after exact path and digest validation."""
    relative_raw = terminal.get("response_path")
    digest = terminal.get("response_sha256")
    if not isinstance(relative_raw, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("BLOCKED_EXECUTION:response_sidecar_metadata")
    relative = Path(relative_raw)
    if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2
            or relative.parts[0] != "responses"
            or relative.name != f"{digest}.json"):
        raise ValueError("BLOCKED_EXECUTION:response_path_invalid")
    path = Path(root) / relative
    if not path.is_file():
        raise ValueError("BLOCKED_EXECUTION:response_missing")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("BLOCKED_EXECUTION:response_hash_mismatch")
    try:
        body = json.loads(payload)
    except Exception as exc:
        raise ValueError("BLOCKED_EXECUTION:response_invalid_json") from exc
    if not isinstance(body, dict):
        raise ValueError("BLOCKED_EXECUTION:response_not_object")
    return body


def _key(config: TransportConfig, payload: dict[str, Any], metadata: dict[str, Any], freeze: str) -> str:
    stable = {
        "freeze_sha256": freeze,
        "provider": config.provider,
        "model": config.model,
        "payload_sha256": sha256_bytes(canonical(payload).encode("utf-8")),
        "metadata": metadata,
    }
    return sha256_bytes(canonical(stable).encode("utf-8"))


def _validate_base_url(value: str) -> None:
    parsed = urlparse(value)
    if not parsed.hostname:
        raise ValueError("BLOCKED_EXECUTION:base_url_must_have_host")
    # Paid calls require TLS for remote endpoints.  The local omniproxy is an
    # explicitly trusted loopback service, so HTTP is allowed only on loopback.
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (loopback and parsed.scheme == "http"):
        raise ValueError("BLOCKED_EXECUTION:base_url_must_be_https_or_loopback_http")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("BLOCKED_EXECUTION:base_url_credentials_forbidden")


def _retryable(terminal: dict[str, Any]) -> bool:
    if terminal.get("outcome") == "TRANSPORT_ERROR":
        status = terminal.get("http_status")
        return status is None or status == 429 or (isinstance(status, int) and 500 <= status < 600)
    return False


class OpenAITransport:
    """Single OpenAI-compatible egress point used by every paid-run phase."""

    def __init__(self, ledger: RunLedger):
        self.ledger = ledger

    def call(self, config: TransportConfig, payload: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
        _validate_base_url(config.base_url)
        if (isinstance(config.max_retries, bool) or not isinstance(config.max_retries, int)
                or not 0 <= config.max_retries <= 8):
            raise ValueError("BLOCKED_EXECUTION:invalid_max_retries")

        request_payload = dict(payload)
        request_payload["model"] = config.model
        if config.provider == "openrouter":
            request_payload["provider"] = {"allow_fallbacks": bool(config.allow_fallbacks)}
        key = _key(config, request_payload, metadata, self.ledger.freeze_sha256)
        cached = self.ledger.terminal(key)
        if cached is not None and cached.get("outcome") == "OK":
            return cached
        if cached is not None and not _retryable(cached):
            return cached

        retry_count = int(cached.get("retry_count", 0)) + 1 if cached is not None else 0
        if retry_count > config.max_retries:
            return cached  # type: ignore[return-value]

        secret = _validate_secret_file(config.api_key_file)
        prompt_sha = sha256_bytes(canonical(request_payload.get("messages", [])).encode("utf-8"))

        while True:
            started_at = _utc()
            started_mono = time.monotonic()
            common = {
                "schema": "ecm-tqag.run-ledger.v1",
                "idempotency_key": key,
                "provider": config.provider,
                "model": config.model,
                "prompt_sha256": prompt_sha,
                "payload_sha256": sha256_bytes(canonical(request_payload).encode("utf-8")),
                "started_at": started_at,
                **metadata,
            }
            try:
                self.ledger.reserve_attempt(common, retry=retry_count > 0)
            except RuntimeError as exc:
                raise BudgetExceeded(str(exc)) from exc

            req = urllib.request.Request(
                config.base_url,
                data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {secret}"},
                method="POST",
            )
            body: dict[str, Any] | None = None
            status: int | None = None
            outcome = "TRANSPORT_ERROR"
            reason: str | None = None
            try:
                with urllib.request.urlopen(req, timeout=config.timeout_sec) as response:
                    status = int(getattr(response, "status", 200))
                    raw_bytes = response.read()
                parsed = json.loads(raw_bytes.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("provider_envelope_not_object")
                body = parsed
                returned_identity = body.get("model")
                if returned_identity is not None and returned_identity != config.model:
                    outcome = "MODEL_IDENTITY_VIOLATION"
                    reason = "response_model_mismatch"
                elif not isinstance(body.get("choices"), list) or not body["choices"]:
                    outcome = "MALFORMED_ENVELOPE"
                    reason = "missing_choices"
                else:
                    outcome = "OK"
            except urllib.error.HTTPError as exc:
                status = exc.code
                detail = _redact(exc.read().decode("utf-8", "replace"), secret)
                reason = f"http_{exc.code}:{detail}"
            except Exception as exc:
                reason = f"{type(exc).__name__}:{_redact(str(exc), secret)}"

            ended_at = _utc()
            elapsed = round(time.monotonic() - started_mono, 6)
            response_sha = None
            response_path = None
            usage = None
            returned_model = None
            system_fingerprint = None
            if body is not None:
                raw = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
                response_sha = hashlib.sha256(raw).hexdigest()
                response_dir = self.ledger.path.parent / "responses"
                response_dir.mkdir(parents=True, exist_ok=True)
                sidecar = response_dir / f"{response_sha}.json"
                if not sidecar.exists():
                    _write_private(sidecar, raw)
                elif sidecar.stat().st_mode & 0o077:
                    os.chmod(sidecar, 0o600)
                response_path = str(sidecar.relative_to(self.ledger.path.parent))
                usage = body.get("usage")
                returned_model = body.get("model")
                system_fingerprint = body.get("system_fingerprint")

            terminal = self.ledger.finish({
                **common,
                "ended_at": ended_at,
                "elapsed_sec": elapsed,
                "http_status": status,
                "outcome": outcome,
                "reason": reason,
                "usage": usage,
                "returned_model": returned_model,
                "system_fingerprint": system_fingerprint,
                "response_sha256": response_sha,
                "response_path": response_path,
                "retry_count": retry_count,
            })
            if outcome == "OK" or not _retryable(terminal) or retry_count >= config.max_retries:
                return terminal
            retry_count += 1
