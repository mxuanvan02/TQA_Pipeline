"""ECM--TQAG publication instrument.

Importing this package performs no network I/O. Every network-capable command is a
dry run unless the caller supplies ``--execute`` and a frozen experiment record.
"""

METHOD = "ecm_tqag.construct"
PROMPT_RELEASE = "ecm_tqag.planner.release"
GRAPH_SCHEMA = "ecm-tqag.structure-graph.release"
RESULT_SCHEMA = "ecm-tqag.construct-result.release"

ARMS = (
    "full",
    "caption_mediated",
    "text_only",
    "text_assisted_reader",
    "direct",
    "gates_off",
)
