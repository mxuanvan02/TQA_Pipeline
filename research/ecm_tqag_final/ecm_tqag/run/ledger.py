from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class LedgerIntegrityError(RuntimeError):
    pass


class RunLedger:
    """Append-only write-ahead ledger with deterministic replay.

    A CALL_STARTED record is itself a spent HTTP-attempt token. This is conservative
    after a crash: an orphan is never forgotten merely because no response was saved.
    """

    def __init__(self, path: Path, *, freeze_sha256: str, cap: int, retry_reserve: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.freeze_sha256 = freeze_sha256
        self.cap = int(cap)
        self.retry_reserve = int(retry_reserve)
        self._lock = threading.Lock()
        self._records = self._read_records()
        seen_freezes = {r.get("freeze_sha256") for r in self._records}
        if seen_freezes - {freeze_sha256}:
            raise LedgerIntegrityError("BLOCKED_RESUME:freeze_mismatch")
        self._terminals: dict[str, dict[str, Any]] = {}
        for record in self._records:
            if record.get("record_type") == "CALL_TERMINAL":
                key = record.get("idempotency_key")
                if not isinstance(key, str) or not key:
                    raise LedgerIntegrityError("BLOCKED_RESUME:terminal_missing_idempotency_key")
                previous = self._terminals.get(key)
                if previous is not None and previous.get("outcome") == "OK":
                    raise LedgerIntegrityError("BLOCKED_RESUME:terminal_after_success")
                self._terminals[key] = record

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise LedgerIntegrityError(f"BLOCKED_RESUME:invalid_json_line:{line_no}") from exc
            if not isinstance(value, dict):
                raise LedgerIntegrityError(f"BLOCKED_RESUME:non_object_line:{line_no}")
            records.append(value)
        return records

    @property
    def attempts_used(self) -> int:
        return sum(r.get("record_type") == "CALL_STARTED" for r in self._records)

    def terminal(self, key: str) -> dict[str, Any] | None:
        value = self._terminals.get(key)
        return dict(value) if value is not None else None

    def reserve_attempt(self, record: dict[str, Any], *, retry: bool = False) -> None:
        with self._lock:
            if self.attempts_used >= self.cap:
                raise RuntimeError(f"BLOCKED_BUDGET:http_cap_exhausted:{self.attempts_used}/{self.cap}")
            base_limit = self.cap - self.retry_reserve
            if not retry and self.attempts_used >= base_limit:
                raise RuntimeError(
                    f"BLOCKED_BUDGET:retry_reserve_protected:{self.attempts_used}/{base_limit}"
                )
            self._append({**record, "record_type": "CALL_STARTED", "freeze_sha256": self.freeze_sha256})

    def finish(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            key = str(record["idempotency_key"])
            previous = self._terminals.get(key)
            if previous is not None and previous.get("outcome") == "OK":
                raise LedgerIntegrityError("BLOCKED_RESUME:terminal_after_success")
            final = {**record, "record_type": "CALL_TERMINAL", "freeze_sha256": self.freeze_sha256}
            self._append(final)
            self._terminals[key] = final
            return dict(final)

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(self.path, flags, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        self._records.append(record)
