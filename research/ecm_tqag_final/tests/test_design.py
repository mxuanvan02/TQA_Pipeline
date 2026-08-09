from __future__ import annotations

import json
from pathlib import Path

from ecm_tqag.arms import ARMS, assert_design
from ecm_tqag.io import sha256_file
from ecm_tqag.prompts import planner_prompt, planner_static_fingerprint
from ecm_tqag.stats.paired import exact_mcnemar, minimum_unidirectional_discordance


def test_arm_design_and_budget() -> None:
    assert_design()
    assert len(ARMS) == 6
    assert sum(a.construction_calls_per_chunk for a in ARMS) == 8
    assert sum(a.construction_calls_per_chunk for a in ARMS) * 16 == 128
    assert len(ARMS) * 16 == 96


def test_primary_pair_and_static_template() -> None:
    assert [a.name for a in ARMS if a.confirmatory] == ["full", "caption_mediated"]
    assert planner_static_fingerprint()
    common = {"section": "x"}
    graph = {"nodes": [], "edges": []}
    caption = {"caption": "x"}
    a = planner_prompt("T", common, graph, interface_kind="closed_graph")
    b = planner_prompt("T", common, caption, interface_kind="caption")
    assert a.replace("INTERFACE_KIND=closed_graph", "INTERFACE_KIND=<KIND>").split("INTERFACE=")[0] == b.replace("INTERFACE_KIND=caption", "INTERFACE_KIND=<KIND>").split("INTERFACE=")[0]


def test_mcnemar_reproduces_four_three_table() -> None:
    # 1 both, 4 left-only, 3 right-only, 8 neither.
    left = [True] + [True] * 4 + [False] * 3 + [False] * 8
    right = [True] + [False] * 4 + [True] * 3 + [False] * 8
    result = exact_mcnemar(left, right)
    assert result["n"] == 16
    assert result["left_only"] == 4
    assert result["right_only"] == 3
    assert result["discordant"] == 7
    assert result["p_two_sided_exact"] == 1.0
    assert minimum_unidirectional_discordance() == 6


def test_manifest_identity_if_present() -> None:
    path = Path(__file__).parents[2] / "fixtures" / "manifest.json"
    if path.is_file():
        obj = json.loads(path.read_text(encoding="utf-8"))
        assert obj["schema"].startswith("ecm-tqag.")
        assert sha256_file(path)
