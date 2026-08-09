#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ecm_tqag.freeze import validate_execution_gate, validate_pre_smoke_gate
from ecm_tqag.io import sha256_file, write_json
from ecm_tqag.run.ledger import RunLedger
from ecm_tqag.run.smoke import SMOKE_CODE, build_smoke_payload
from ecm_tqag.run.smoke_result import validate_smoke_sidecar
from ecm_tqag.run.transport import OpenAITransport, TransportConfig


def _smoke_image_data_url() -> str:
    image = Image.new("RGB", (640, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 628, 228), outline="black", width=5)
    draw.text((85, 95), f"VERIFICATION CODE: {SMOKE_CODE}", fill="black")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _completion_url(base: str) -> str:
    """Accept an API root or an exact chat-completions endpoint."""
    value = base.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


def _config(
    role: dict[str, Any], *, openrouter_key: Path, omniproxy_key: Path,
    openrouter_url: str, omniproxy_url: str,
) -> TransportConfig:
    provider = role["provider"]
    if provider == "openrouter":
        return TransportConfig(provider=provider, model=role["model"], base_url=_completion_url(openrouter_url),
                               api_key_file=openrouter_key, allow_fallbacks=False)
    if provider == "omniproxy":
        return TransportConfig(provider=provider, model=role["model"], base_url=_completion_url(omniproxy_url),
                               api_key_file=omniproxy_key, allow_fallbacks=False)
    raise ValueError(f"BLOCKED_SMOKE:unsupported_provider:{provider}")


def current_run_ledger_cap(freeze: dict[str, Any]) -> int:
    """Return the cap available to this run after prior attempts are carried forward."""
    try:
        budget = freeze["full_call_budget"]
        cap = budget["current_ledger_cap"]
        operational = freeze["operational_http_cap"]
    except (KeyError, TypeError) as exc:
        raise ValueError("BLOCKED_SMOKE:current_ledger_cap_missing") from exc
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0 or cap > operational:
        raise ValueError(f"BLOCKED_SMOKE:invalid_current_ledger_cap:{cap!r}")
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the six frozen role-specific smoke calls.")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--openrouter-key", type=Path, required=True)
    parser.add_argument("--omniproxy-key", type=Path, required=True)
    parser.add_argument("--openrouter-url", required=True)
    parser.add_argument("--omniproxy-url", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validate_pre_smoke_gate(freeze, execute=args.execute)
    freeze_sha = sha256_file(args.freeze)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(
        args.run_dir / "RUN_LEDGER.jsonl",
        freeze_sha256=freeze_sha,
        cap=current_run_ledger_cap(freeze),
        retry_reserve=int(freeze["full_call_budget"]["retry_reserve"]),
    )
    transport = OpenAITransport(ledger)
    image_data_url = _smoke_image_data_url()
    passed: set[str] = set()
    rows: list[dict[str, Any]] = []

    for role_name in sorted(freeze["roles"]):
        role = freeze["roles"][role_name]
        vision = bool(role["vision_required"])
        terminal = transport.call(
            _config(
                role,
                openrouter_key=args.openrouter_key,
                omniproxy_key=args.omniproxy_key,
                openrouter_url=args.openrouter_url,
                omniproxy_url=args.omniproxy_url,
            ),
            build_smoke_payload(
                vision_required=vision,
                image_data_url=image_data_url if vision else None,
            ),
            metadata={"phase": "role_smoke", "role": role_name, "vision_required": vision},
        )
        if terminal.get("outcome") != "OK":
            raise ValueError(f"BLOCKED_SMOKE:{role_name}:{terminal.get('outcome')}:{terminal.get('reason')}")
        response_rel = terminal.get("response_path")
        if not isinstance(response_rel, str):
            raise ValueError(f"BLOCKED_SMOKE:{role_name}:missing_response_sidecar")
        validate_smoke_sidecar(args.run_dir / response_rel, vision_required=vision)
        passed.add(role_name)
        rows.append({
            "role": role_name,
            "provider": role["provider"],
            "model": role["model"],
            "vision_required": vision,
            "status": "PASS",
            "idempotency_key": terminal["idempotency_key"],
            "response_sha256": terminal["response_sha256"],
            "usage": terminal.get("usage"),
        })

    validate_execution_gate(freeze, execute=args.execute, smoke_passed_roles=passed)
    result = {
        "schema": "ecm-tqag.role-smoke.v1",
        "freeze_sha256": freeze_sha,
        "status": "PASS",
        "passed_roles": sorted(passed),
        "http_attempts_used": ledger.attempts_used,
        "results": rows,
    }
    write_json(args.run_dir / "SMOKE_RESULTS.json", result)
    print(json.dumps({
        "status": result["status"],
        "passed_roles": result["passed_roles"],
        "http_attempts_used": result["http_attempts_used"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
