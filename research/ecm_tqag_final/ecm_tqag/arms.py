from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    name: str
    construction_calls_per_chunk: int
    input_mode: str
    path: str
    confirmatory: bool = False
    deterministic_rescore: bool = False


ARMS = (
    Arm("full", 2, "closed_graph_from_pixels_only", "planner_realizer", True),
    Arm("caption_mediated", 2, "frozen_caption_from_pixels_only", "planner_realizer", True),
    Arm("text_only", 1, "ocr_only", "direct_generation"),
    Arm("text_assisted_reader", 2, "closed_graph_from_pixels_and_ocr", "planner_realizer"),
    Arm("direct", 1, "ocr_layout_pixels", "direct_generation"),
    Arm("gates_off", 0, "reuse_full_outputs", "deterministic_rescore", deterministic_rescore=True),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}


def assert_design() -> None:
    if len(ARMS) != 6 or len(ARM_BY_NAME) != 6:
        raise AssertionError("design must contain six uniquely named arms")
    primary = [a.name for a in ARMS if a.confirmatory]
    if primary != ["full", "caption_mediated"]:
        raise AssertionError(f"unexpected primary pair: {primary}")
    if sum(a.construction_calls_per_chunk for a in ARMS) != 8:
        raise AssertionError("construction budget must be eight calls per chunk")
    if ARM_BY_NAME["gates_off"].construction_calls_per_chunk != 0:
        raise AssertionError("gates_off must be a free deterministic rescore")


assert_design()
