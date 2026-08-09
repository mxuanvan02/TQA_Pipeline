from __future__ import annotations

from ecm_tqag import item_gates


def _item() -> dict:
    return {
        "chunk_id": "fixture",
        "condition": "TLV",
        "status": "PARSED",
        "question": "Câu hỏi trung tính?",
        "choices": ["Đáp án đúng", "Sai một", "Sai hai", "Sai ba"],
        "answer_index": 0,
        "anchors": ["Đáp án đúng"],
        "observations": ["quan hệ hình học mới"],
        "atom_values": ["Đáp án đúng"],
        "contract_gate": {"ok": True, "reason": None},
        "seal_gate": {"ok": True, "reason": None},
    }


def test_novelty_is_exploratory_and_cannot_fail_confirmatory_yield(monkeypatch) -> None:
    monkeypatch.setattr(item_gates, "gate_novelty", lambda *a, **k: {"ok": False, "reason": "exploratory_fixture"})
    monkeypatch.setattr(item_gates, "gate_cue", lambda *a, **k: {"ok": True, "reason": None})
    monkeypatch.setattr(item_gates, "gate_support", lambda *a, **k: {"ok": True, "reason": None})
    monkeypatch.setattr(item_gates, "gate_unique", lambda *a, **k: {"ok": True, "reason": None})

    result = item_gates.evaluate_item(_item(), "nguồn", {}, 1, 0.95)
    assert result["gates"]["G-NOVELTY"]["ok"] is False
    assert result["exploratory_failed_gates"] == ["G-NOVELTY"]
    assert result["confirmatory_failed_gates"] == []
    assert result["passes_confirmatory_gates"] is True
    assert result["passes_all_gates"] is True
