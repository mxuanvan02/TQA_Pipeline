from __future__ import annotations

import pytest

from ecm_tqag import item_gates


def _item() -> dict:
    return {
        "chunk_id": "c1", "condition": "full", "status": "PARSED",
        "question": "Câu hỏi?",
        "choices": ["Đúng", "Sai một", "Sai hai", "Sai ba"],
        "answer_index": 0,
        "anchors": ["nguồn quy định"],
        "observations": ["một quan hệ cụ thể nhìn thấy rõ"],
        "atom_values": ["Đúng"],
        "contract_gate": {"ok": False, "reason": "contract_fixture"},
        "seal_gate": {"ok": True, "reason": None},
    }


def test_evaluate_item_uses_all_five_confirmatory_gates(monkeypatch):
    for name in ("gate_novelty", "gate_cue", "gate_support", "gate_unique"):
        monkeypatch.setattr(item_gates, name, lambda *a, **k: {"ok": True, "reason": None})

    result = item_gates.evaluate_item(_item(), "nguồn", {}, 1, 0.95)
    assert result["gates"]["CONTRACT"]["ok"] is False
    assert result["confirmatory_failed_gates"] == ["CONTRACT"]
    assert result["passes_confirmatory_gates"] is False


def test_evaluate_item_fails_closed_when_contract_or_seal_missing(monkeypatch):
    for name in ("gate_novelty", "gate_cue", "gate_support", "gate_unique"):
        monkeypatch.setattr(item_gates, name, lambda *a, **k: {"ok": True, "reason": None})
    item = _item()
    item.pop("contract_gate")
    item.pop("seal_gate")
    with pytest.raises(ValueError, match="BLOCKED_GATES:missing_confirmatory_gate:CONTRACT"):
        item_gates.evaluate_item(item, "nguồn", {}, 1, 0.95)
