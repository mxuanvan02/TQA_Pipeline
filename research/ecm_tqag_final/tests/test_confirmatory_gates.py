from ecm_tqag.item_gates import confirmatory_gate_verdict


def test_confirmatory_verdict_composes_five_gates_and_excludes_novelty():
    gates = {
        "G-CUE": {"ok": True},
        "G-SUPPORT": {"ok": True},
        "G-UNIQUE": {"ok": True},
        "CONTRACT": {"ok": True},
        "SEAL": {"ok": False, "reason": "seal_failed"},
        "G-NOVELTY": {"ok": False},
    }
    verdict = confirmatory_gate_verdict(gates)
    assert verdict["required"] == ["G-CUE", "G-SUPPORT", "G-UNIQUE", "CONTRACT", "SEAL"]
    assert verdict["failed"] == ["SEAL"]
    assert verdict["ok"] is False
    assert "G-NOVELTY" not in verdict["required"]


def test_confirmatory_verdict_fails_closed_on_missing_gate():
    try:
        confirmatory_gate_verdict({"G-CUE": {"ok": True}})
    except ValueError as exc:
        assert str(exc) == "BLOCKED_GATES:missing_confirmatory_gate:G-SUPPORT"
    else:
        raise AssertionError("missing confirmatory gate was accepted")
