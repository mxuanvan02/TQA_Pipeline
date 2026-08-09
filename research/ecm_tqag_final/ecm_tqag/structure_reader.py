#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structure_read_v1.py -- STAGE A of the v7 protocol: BLIND STRUCTURE READ.

Design contract (the whole reason this module exists as its own file):

    read_structure() accepts an IMAGE PATH and an OPTIONAL BBOX. Nothing else.
    It is structurally impossible to hand it the chunk OCR text.

Why that matters. The v3 protocol showed the model OCR(I) and I in the same
pass and asked for a "visual observation". The observation came back as a
measurable function of the text (IDF coverage mean 0.973). With T = OCR(I),
sigma(T) subset sigma(I); if o = g(T) then a = f(o, r) = h(T) and the data
processing inequality forces I(a; I | T) = 0. Visual-dependence rate 0 was
therefore a theorem about the protocol, not a measurement of the model.

The only way to break that implication is to make the observation channel
text-free BY CONSTRUCTION. Hence: no text parameter. A module-level assertion
enforces it at import time, and test_structure_read_v1.py re-checks it via
inspect.signature so the guarantee cannot silently rot.

The model is asked for a STRUCTURE GRAPH under a CLOSED schema. Validation is
strict and never repairs: unknown keys, missing keys, dangling edge endpoints,
unknown enum members, or non-JSON all REJECT with an explicit reason string.

Stdlib only (+ Pillow, used solely for optional bbox cropping).
No network I/O happens at import time.
"""
from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "ecm-tqag.structure-read.release"

# Transport identity is supplied only by the approved execution roster.  Keeping
# these unset prevents a direct function call from silently inheriting a provider.
BASE_URL: str | None = None
MODEL: str | None = None
TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Closed schema
# ---------------------------------------------------------------------------
GRAPH_TYPES = frozenset({"TREE", "FLOW", "CYCLE", "MATRIX", "CHART", "OTHER"})
EDGE_KINDS = frozenset({"ARROW", "LINE", "CONTAINMENT"})

TOP_KEYS = frozenset({"graph_type", "nodes", "edges", "confidence"})
NODE_KEYS = frozenset({"id", "label", "level", "bbox"})
EDGE_KEYS = frozenset({"src", "dst", "kind", "label"})

# Explicit, stable reason strings. Tests assert on these exact values.
R_NON_JSON = "non_json_response"
R_NOT_OBJECT = "top_level_not_object"
R_UNKNOWN_TOP_KEY = "unknown_top_level_key"
R_MISSING_TOP_KEY = "missing_top_level_key"
R_UNKNOWN_GRAPH_TYPE = "unknown_graph_type"
R_NODES_NOT_LIST = "nodes_not_list"
R_EDGES_NOT_LIST = "edges_not_list"
R_NODE_NOT_OBJECT = "node_not_object"
R_EDGE_NOT_OBJECT = "edge_not_object"
R_UNKNOWN_NODE_KEY = "unknown_node_key"
R_MISSING_NODE_KEY = "missing_node_key"
R_UNKNOWN_EDGE_KEY = "unknown_edge_key"
R_MISSING_EDGE_KEY = "missing_edge_key"
R_UNKNOWN_EDGE_KIND = "unknown_edge_kind"
R_DANGLING_EDGE = "dangling_edge_endpoint"
R_DUPLICATE_NODE_ID = "duplicate_node_id"
R_BAD_LEVEL = "level_not_int"
R_BAD_BBOX = "bbox_not_4_numbers"
R_BAD_CONFIDENCE = "confidence_not_unit_interval"
R_EMPTY_NODES = "no_nodes"

PROMPT = """\
You are reading a scanned page image from a Vietnamese law textbook.

Your ONLY job is to transcribe the GEOMETRIC AND RELATIONAL STRUCTURE of the
diagram, flowchart or chart on the page. You are NOT given any text transcript
and you must NOT summarise the prose of the page.

Return a STRUCTURE GRAPH as STRICT JSON. No markdown fence, no commentary,
no trailing text. Exactly these four keys, exactly these value types:

{
  "graph_type": "TREE|FLOW|CYCLE|MATRIX|CHART|OTHER",
  "nodes": [{"id": "n1", "label": "<text inside the box>", "level": 0,
             "bbox": [x0, y0, x1, y1]}],
  "edges": [{"src": "n1", "dst": "n2", "kind": "ARROW|LINE|CONTAINMENT",
             "label": ""}],
  "confidence": 0.0
}

Rules:
- "id" must be unique, of the form n1, n2, n3, ...
- "level" is the hierarchical depth of the node: the topmost / root row is 0,
  its children are 1, their children are 2, and so on. For a flow, level is the
  step index. For a chart, use 0 for every node.
- "bbox" is [x0, y0, x1, y1] on a FIXED 0--1000 GRID over the full image:
  left=0, top=0, right=1000, bottom=1000. Use numbers in [0,1000], with
  x0 < x1 and y0 < y1. This grid is independent of pixels and provider resizing.
- Every "src" and "dst" MUST be an id that appears in "nodes".
- "kind" is ARROW when the connector has an arrowhead, LINE when it is a plain
  connector, CONTAINMENT when one box is drawn INSIDE another.
- "label" on an edge is the text written ON the connector, or "" when there is
  none.
- "confidence" is your own confidence in the extracted structure, 0.0 to 1.0.
- If the page has no diagram at all, return graph_type "OTHER" with an empty
  edges list and one node per visible block.

Emit ONLY the JSON object.
"""

PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()

# Any parameter name that could smuggle the OCR transcript into Stage A.
FORBIDDEN_PARAM_TOKENS = (
    "text", "ocr", "chunk", "prose", "evidence", "transcript",
    "caption", "passage", "context", "doc",
)


def _assert_blind_signature(fn) -> None:
    """Fail LOUDLY at import time if the Stage A entry point ever grows a
    parameter that could carry the chunk text."""
    params = list(inspect.signature(fn).parameters)
    bad = [p for p in params
           if any(tok in p.lower() for tok in FORBIDDEN_PARAM_TOKENS)]
    if bad:
        raise AssertionError(
            f"structure_read_v1.read_structure has text-carrying parameters "
            f"{bad}; Stage A must be blind by construction"
        )


# ---------------------------------------------------------------------------
# Validation (pure, no I/O -- this is what the tests hammer)
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_fence(text: str) -> tuple[str, bool]:
    """Remove a markdown fence if present. Returns (payload, was_fenced)."""
    m = _FENCE_RE.match(text or "")
    if m:
        return m.group(1), True
    return (text or ""), False


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_graph(obj: Any) -> dict:
    """Strict closed-schema validation.

    Returns {"ok": bool, "reason": str|None, "detail": str}. NEVER repairs,
    NEVER defaults. The first violation found is reported.
    """
    def bad(reason: str, detail: str = "") -> dict:
        return {"ok": False, "reason": reason, "detail": detail}

    if not isinstance(obj, dict):
        return bad(R_NOT_OBJECT, f"type={type(obj).__name__}")

    keys = set(obj)
    unknown = sorted(keys - TOP_KEYS)
    if unknown:
        return bad(R_UNKNOWN_TOP_KEY, f"unknown={unknown}")
    missing = sorted(TOP_KEYS - keys)
    if missing:
        return bad(R_MISSING_TOP_KEY, f"missing={missing}")

    if obj["graph_type"] not in GRAPH_TYPES:
        return bad(R_UNKNOWN_GRAPH_TYPE, f"graph_type={obj['graph_type']!r}")

    if not isinstance(obj["nodes"], list):
        return bad(R_NODES_NOT_LIST, f"type={type(obj['nodes']).__name__}")
    if not isinstance(obj["edges"], list):
        return bad(R_EDGES_NOT_LIST, f"type={type(obj['edges']).__name__}")
    if not obj["nodes"]:
        return bad(R_EMPTY_NODES, "nodes=[]")

    if not _is_num(obj["confidence"]) or not (0.0 <= float(obj["confidence"]) <= 1.0):
        return bad(R_BAD_CONFIDENCE, f"confidence={obj['confidence']!r}")

    ids: set[str] = set()
    for i, n in enumerate(obj["nodes"]):
        if not isinstance(n, dict):
            return bad(R_NODE_NOT_OBJECT, f"nodes[{i}]")
        nk = set(n)
        u = sorted(nk - NODE_KEYS)
        if u:
            return bad(R_UNKNOWN_NODE_KEY, f"nodes[{i}] unknown={u}")
        m = sorted(NODE_KEYS - nk)
        if m:
            return bad(R_MISSING_NODE_KEY, f"nodes[{i}] missing={m}")
        if not isinstance(n["id"], str) or not n["id"]:
            return bad(R_NODE_NOT_OBJECT, f"nodes[{i}].id={n['id']!r}")
        if n["id"] in ids:
            return bad(R_DUPLICATE_NODE_ID, f"id={n['id']!r}")
        ids.add(n["id"])
        if isinstance(n["level"], bool) or not isinstance(n["level"], int):
            return bad(R_BAD_LEVEL, f"nodes[{i}].level={n['level']!r}")
        bb = n["bbox"]
        if not isinstance(bb, list) or len(bb) != 4 or not all(_is_num(v) for v in bb):
            return bad(R_BAD_BBOX, f"nodes[{i}].bbox={bb!r}")

    for j, e in enumerate(obj["edges"]):
        if not isinstance(e, dict):
            return bad(R_EDGE_NOT_OBJECT, f"edges[{j}]")
        ek = set(e)
        u = sorted(ek - EDGE_KEYS)
        if u:
            return bad(R_UNKNOWN_EDGE_KEY, f"edges[{j}] unknown={u}")
        m = sorted(EDGE_KEYS - ek)
        if m:
            return bad(R_MISSING_EDGE_KEY, f"edges[{j}] missing={m}")
        if e["kind"] not in EDGE_KINDS:
            return bad(R_UNKNOWN_EDGE_KIND, f"edges[{j}].kind={e['kind']!r}")
        if e["src"] not in ids:
            return bad(R_DANGLING_EDGE, f"edges[{j}].src={e['src']!r} not in nodes")
        if e["dst"] not in ids:
            return bad(R_DANGLING_EDGE, f"edges[{j}].dst={e['dst']!r} not in nodes")

    return {"ok": True, "reason": None, "detail": ""}


def parse_and_validate(raw_text: str) -> dict:
    """Text -> validated graph, or an explicit rejection."""
    payload, fenced = strip_fence(raw_text)
    try:
        obj = json.loads(payload)
    except Exception as exc:
        return {"ok": False, "reason": R_NON_JSON,
                "detail": f"{type(exc).__name__}: {exc}",
                "graph": None, "fence_stripped": fenced}
    v = validate_graph(obj)
    return {"ok": v["ok"], "reason": v["reason"], "detail": v["detail"],
            "graph": obj if v["ok"] else None, "fence_stripped": fenced}


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def load_api_key(key_file: str) -> str:
    """Read the bearer token. The value is never logged or returned upward."""
    return Path(key_file).read_text(encoding="utf-8").strip()


def _data_url_from_bytes(blob: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(blob).decode("ascii")


def _image_payload(image_path: str, bbox: tuple[int, int, int, int] | None):
    """Return (data_url, width, height, cropped: bool)."""
    p = Path(image_path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    if bbox is None:
        blob = p.read_bytes()
        try:
            from PIL import Image
            with Image.open(io.BytesIO(blob)) as im:
                w, h = im.size
        except Exception:
            w = h = None
        return _data_url_from_bytes(blob, mime), w, h, False
    from PIL import Image
    with Image.open(p) as im:
        crop = im.crop(tuple(int(v) for v in bbox))
        buf = io.BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=95)
        w, h = crop.size
    return _data_url_from_bytes(buf.getvalue(), "image/jpeg"), w, h, True


def _post(payload: dict, api_key: str, base_url: str, timeout_sec: int) -> dict:
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_with_retry(payload: dict, api_key: str, base_url: str,
                    timeout_sec: int, retries: int) -> dict:
    """Exponential backoff on 429 / 5xx / URLError. Errors recorded verbatim."""
    attempts: list[str] = []
    for attempt in range(retries + 1):
        try:
            body = _post(payload, api_key, base_url, timeout_sec)
            return {"ok": True, "body": body, "retry_count": attempt,
                    "error": None, "attempts": attempts}
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:800]
            except Exception:
                detail = ""
            msg = f"HTTPError {exc.code}: {exc.reason} :: {detail}"
            attempts.append(msg)
            transient = exc.code == 429 or 500 <= exc.code < 600
            if not transient or attempt >= retries:
                return {"ok": False, "body": None, "retry_count": attempt,
                        "error": msg, "attempts": attempts}
            time.sleep(min(2.0 * (2 ** attempt), 30.0))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            attempts.append(msg)
            if attempt >= retries:
                return {"ok": False, "body": None, "retry_count": attempt,
                        "error": msg, "attempts": attempts}
            time.sleep(min(2.0 * (2 ** attempt), 30.0))
    return {"ok": False, "body": None, "retry_count": retries,
            "error": "retries_exhausted", "attempts": attempts}


# ---------------------------------------------------------------------------
# STAGE A entry point -- BLIND BY CONSTRUCTION
# ---------------------------------------------------------------------------
def read_structure(image_path: str,
                   bbox: tuple[int, int, int, int] | None = None,
                   *,
                   api_key_file: str | None = None,
                   model: str | None = MODEL,
                   base_url: str | None = BASE_URL,
                   timeout_sec: int = 180,
                   retries: int = 4,
                   execute: bool = False) -> dict:
    """Read the STRUCTURE GRAPH of one image. Image path (+ optional bbox) ONLY.

    There is deliberately NO parameter through which the chunk OCR text, page
    prose, or any textual context can enter. See _assert_blind_signature.
    """
    started = time.time()
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "image_path": image_path,
        "bbox": list(bbox) if bbox else None,
        "model": model,
        "base_url": base_url,
        "temperature": TEMPERATURE,
        "prompt_sha256": PROMPT_SHA256,
        "status": None,
        "reason": None,
        "detail": "",
        "graph": None,
        "usage": None,
        "retry_count": 0,
        "elapsed_sec": None,
        "raw_response": None,
        "error": None,
    }
    if not os.path.exists(image_path):
        out.update(status="REJECT", reason="image_missing",
                   detail=image_path, elapsed_sec=round(time.time() - started, 3))
        return out

    try:
        durl, w, h, cropped = _image_payload(image_path, bbox)
    except Exception as exc:
        out.update(status="REJECT", reason="image_unreadable",
                   detail=f"{type(exc).__name__}: {exc}",
                   elapsed_sec=round(time.time() - started, 3))
        return out
    out["image_width"] = w
    out["image_height"] = h
    out["cropped"] = cropped

    if not execute:
        out.update(status="DRYRUN", elapsed_sec=round(time.time() - started, 3))
        return out

    if not model or not base_url or not api_key_file:
        out.update(status="BLOCKED", reason="BLOCKED_EXECUTION_ROSTER",
                   detail="model, base_url, and api_key_file are required for execution",
                   elapsed_sec=round(time.time() - started, 3))
        return out

    payload = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": durl}},
        ]}],
    }
    api_key = load_api_key(api_key_file)
    call = call_with_retry(payload, api_key, base_url, timeout_sec, retries)
    out["retry_count"] = call["retry_count"]
    out["attempts"] = call["attempts"]
    if not call["ok"]:
        out.update(status="ERROR", reason="api_error", error=call["error"],
                   elapsed_sec=round(time.time() - started, 3))
        return out

    body = call["body"]
    out["usage"] = body.get("usage")
    try:
        raw = body["choices"][0]["message"]["content"]
    except Exception as exc:
        out.update(status="ERROR", reason="malformed_api_envelope",
                   error=f"{type(exc).__name__}: {exc}",
                   raw_response=json.dumps(body)[:2000],
                   elapsed_sec=round(time.time() - started, 3))
        return out
    out["raw_response"] = raw

    pv = parse_and_validate(raw)
    out["fence_stripped"] = pv.get("fence_stripped")
    if not pv["ok"]:
        out.update(status="REJECT", reason=pv["reason"], detail=pv["detail"],
                   elapsed_sec=round(time.time() - started, 3))
        return out
    out.update(status="OK", graph=pv["graph"],
               elapsed_sec=round(time.time() - started, 3))
    return out


# Enforce the blindness contract at import time.
_assert_blind_signature(read_structure)


def graph_stats(graph: dict) -> dict:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    levels = sorted({n["level"] for n in nodes})
    kinds: dict[str, int] = {}
    for e in edges:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    return {
        "graph_type": graph.get("graph_type"),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "levels": levels,
        "max_level": max(levels) if levels else None,
        "edge_kinds": kinds,
        "confidence": graph.get("confidence"),
    }
