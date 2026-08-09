from __future__ import annotations

import pytest

from run_smoke import current_run_ledger_cap


def _freeze() -> dict:
    return {
        "operational_http_cap": 550,
        "full_call_budget": {
            "current_ledger_cap": 536,
            "retry_reserve": 50,
            "worst_case_http_calls": 542,
        },
    }


def test_smoke_uses_current_run_ledger_cap_not_operational_cap() -> None:
    assert current_run_ledger_cap(_freeze()) == 536


def test_smoke_blocks_missing_or_overlarge_current_ledger_cap() -> None:
    missing = _freeze()
    del missing["full_call_budget"]["current_ledger_cap"]
    with pytest.raises(ValueError, match="current_ledger_cap"):
        current_run_ledger_cap(missing)

    oversized = _freeze()
    oversized["full_call_budget"]["current_ledger_cap"] = 551
    with pytest.raises(ValueError, match="current_ledger_cap"):
        current_run_ledger_cap(oversized)
