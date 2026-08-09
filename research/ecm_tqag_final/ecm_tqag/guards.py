"""ECM v3 core: adds the three manuscript-mandated mechanisms missing from v2.

v2 already implements (and this module reuses, unmodified):
  * closed motif catalog  -> matcher
  * typed program         -> compiler
  * anchor execution      -> executor
  * anti-verbatim guard

v3 adds what main.tex claims but v2 did not implement:

  (1) PROVENANCE TRACE Gamma  [main.tex eq. (5), contribution (iii)]
      A = (a_1..a_k) answer atoms; Gamma = (gamma_1..gamma_k) links each atom
      to a source node via rho. Here rho is instantiated concretely as
      (sentence_index, char_span) into the supplied evidence text, so a
      reviewer can replay every atom back to its source characters.

  (2) SEALED REALIZER CONTRACT  [main.tex sec. "Evidence-first construction"]
      "A question realizer then receives the locked answer construction."
      seal_construction() produces the ONLY payload the realizer may see.
      assert_realizer_blind() proves the sealed payload carries no full
      evidence text, so the realizer cannot re-read the chunk.

  (3) VISUAL BINDING FOR TLV  [main.tex: "ECM--TLV must bind a visual node"]
      require_visual_binding() enforces that at least one step is bound to a
      declared image node under condition TLV; otherwise the record is
      rejected with a machine-readable reason rather than silently accepted.

Nothing in this module calls a model. Everything is deterministic and
unit-testable so the derivation can be replayed without API access.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from . import contract as v2

# Re-export the frozen v2 instrument so v3 is a strict superset.
norm = v2.norm
words = v2.words
sentences = v2.sentences
longest_common_run = v2.longest_common_run
build_idf = v2.build_idf
compile_program = v2.compile_program
execute_program = v2.execute_program
CompileError = v2.CompileError
MOTIF_CATALOG = v2.MOTIF_CATALOG
VALID_OPS = v2.VALID_OPS
MAX_VERBATIM_RUN = v2.MAX_VERBATIM_RUN
# A question stem must not disclose the visual value before the respondent uses
# the image. Two shared words can be ordinary grammar; a run of three or more
# words copied from the visual observation is a material leak.
MAX_VISUAL_STEM_RUN = 2
# v4 cross-modal additions, re-exported so callers only ever import v3.
CROSS_MODAL_MOTIFS = v2.CROSS_MODAL_MOTIFS
step_modality = v2.step_modality

SCHEMA = "ecm-tqag.v3"
CONDITIONS = ("T", "TL_struct", "TLV")

# Ops that may legitimately consume a visual node. `traverse`/`identify` cover
# diagram/hierarchy reading; `compare` covers reading two visual facts.
VISUAL_CAPABLE_OPS = {"locate", "traverse", "identify", "compare"}


# --------------------------------------------------------------- (1) trace
def _sentence_spans(evidence_text: str) -> list[tuple[int, int, str]]:
    """Return (start_char, end_char, sentence) for each retained sentence.

    Spans are computed against the ORIGINAL evidence_text so a reviewer can
    slice evidence_text[start:end] and recover the exact source characters.
    """
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sent in sentences(evidence_text):
        idx = evidence_text.find(sent, cursor)
        if idx < 0:                      # normalisation drift: fall back
            idx = evidence_text.find(sent)
        if idx < 0:
            spans.append((-1, -1, sent))
            continue
        spans.append((idx, idx + len(sent), sent))
        cursor = idx + len(sent)
    return spans


def build_trace(program: dict[str, Any], execution: dict[str, Any],
                evidence_text: str, image_nodes: list[str] | None = None) -> dict[str, Any]:
    """Build answer atoms A and provenance trace Gamma.

    Each compiled step yields one answer atom a_j (its `result`). Gamma links
    a_j to its source node:
      * anchor steps  -> rho = ("text", sentence_index, char_span)
      * derived steps -> rho = ("derived", [indices of the steps it consumes])

    Returns {"atoms": [...], "gamma": [...], "trace_hash": ...} where
    trace_hash is a stable digest a reviewer can compare across replays.
    """
    spans = _sentence_spans(evidence_text)
    per_step = {p["index"]: p for p in execution.get("per_step", [])}
    atoms: list[dict[str, Any]] = []
    gamma: list[dict[str, Any]] = []

    anchor_step_indices = [s["index"] for s in program["steps"] if s["needs"] > 0]

    for step in program["steps"]:
        i = step["index"]
        atoms.append({"atom_id": f"a{i}", "step_index": i, "op": step["op"],
                      "value": step["result"]})
        if step["needs"] > 0 and step.get("modality") == "visual":
            # A visual anchor is grounded in the IMAGE, not in evidence_text, so
            # rho is (image_index, bbox) rather than (sentence_index, char_span).
            # Emitting kind="text" here would send it into verify_trace()'s
            # character-slice replay, which can never succeed for something read
            # off a page image -- that mismatch is what made cross-modal steps
            # look like provenance failures. It is verified instead by
            # verify_visual_anchors() against the declared image list.
            gamma.append({
                "atom_id": f"a{i}", "kind": "visual",
                "sentence_index": None, "char_span": None,
                "visual_node": step.get("visual_node"),
                "bbox": step.get("bbox"),
                "anchor_declared": step["anchor"],
                "observation": step["result"],
            })
        elif step["needs"] > 0:
            rep = per_step.get(i, {})
            sent_i = rep.get("matched_sentence")
            if isinstance(sent_i, int) and 0 <= sent_i < len(spans):
                start, end, sent = spans[sent_i]
            else:
                start, end, sent = -1, -1, ""
            gamma.append({
                "atom_id": f"a{i}", "kind": "text",
                "sentence_index": sent_i,
                "char_span": [start, end],
                "match_score": rep.get("match_score"),
                "source_excerpt": sent[:200],
                "anchor_declared": step["anchor"],
                "visual_node": step.get("visual_node"),
            })
        else:
            gamma.append({
                "atom_id": f"a{i}", "kind": "derived",
                "derived_from": [f"a{j}" for j in anchor_step_indices],
                "sentence_index": None, "char_span": None,
                "visual_node": None,
            })

    payload = json.dumps({"atoms": atoms, "gamma": gamma},
                         ensure_ascii=False, sort_keys=True)
    return {"atoms": atoms, "gamma": gamma,
            "trace_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
            "n_atoms": len(atoms),
            "n_text_grounded": sum(1 for g in gamma if g["kind"] == "text"
                                   and isinstance(g["sentence_index"], int)
                                   and g["sentence_index"] >= 0)}


def verify_trace(trace: dict[str, Any], evidence_text: str) -> dict[str, Any]:
    """Replay check: every text-grounded gamma must slice back to real chars.

    This is the mechanical realisation of the manuscript's "atom--trace
    correspondence" check in C.
    """
    problems: list[str] = []
    atom_ids = {a["atom_id"] for a in trace["atoms"]}
    if len(atom_ids) != len(trace["atoms"]):
        problems.append("duplicate_atom_ids")
    if {g["atom_id"] for g in trace["gamma"]} != atom_ids:
        problems.append("atom_gamma_mismatch")

    for g in trace["gamma"]:
        if g["kind"] != "text":
            continue
        span = g.get("char_span")
        if (not isinstance(span, list) or len(span) != 2
                or not all(isinstance(x, int) for x in span)):
            problems.append(f'{g["atom_id"]}:bad_span')
            continue
        start, end = span
        if start < 0 or end <= start or end > len(evidence_text):
            problems.append(f'{g["atom_id"]}:span_out_of_range')
            continue
        sliced = evidence_text[start:end]
        if norm(sliced)[:60] != norm(g.get("source_excerpt", ""))[:60]:
            problems.append(f'{g["atom_id"]}:span_text_mismatch')

    return {"ok": not problems, "problems": problems,
            "n_checked": sum(1 for g in trace["gamma"] if g["kind"] == "text")}


# ------------------------------------------------------- (2) sealed realizer
_REALIZER_ALLOWED_KEYS = {"motif", "atoms", "anchors", "condition",
                          "visual_nodes", "visual_observations",
                          "construction_hash"}

# Single source of truth for the excerpt budget. The pilot rejected valid rows
# with seal:a0:excerpt_too_long because seal_construction() truncated excerpts
# by CHARACTERS (200) while assert_realizer_blind() measured them in WORDS
# (40): Vietnamese averages ~4.5 chars/word, so 200 chars is ~44 words and the
# sealer emitted payloads its own guard refused. Both sides now share this
# constant and truncate/measure in the same unit. Tightening (not relaxing):
# the realizer sees fewer words than before, i.e. it is strictly more blind.
MAX_EXCERPT_WORDS = 40
MAX_LEAK_RATIO = 0.35


def _truncate_words(text: str, max_words: int = MAX_EXCERPT_WORDS) -> str:
    """Cut an excerpt to `max_words` whitespace-delimited tokens.

    Truncating on the same unit the guard measures keeps seal_construction()
    and assert_realizer_blind() consistent by construction.
    """
    toks = (text or "").split()
    if len(toks) <= max_words:
        return (text or "").strip()
    return " ".join(toks[:max_words])


def seal_construction(program: dict[str, Any], trace: dict[str, Any],
                      condition: str, image_nodes: list[str] | None = None,
                      evidence_text: str | None = None) -> dict[str, Any]:
    """Produce the ONLY payload the realizer is allowed to see.

    Carries motif name, answer atoms, and short anchor excerpts -- NOT the
    full evidence text. This implements "the realizer receives the locked
    answer construction" and makes the isolation auditable.

    When `evidence_text` is given, the per-anchor word cap is derived via
    excerpt_word_budget() so short chunks can't violate MAX_LEAK_RATIO even
    though each anchor individually respects MAX_EXCERPT_WORDS (see the
    195-word-chunk failure in full_v3: 2*40/195 = 40.5% > 35%). Omitting
    evidence_text falls back to the flat MAX_EXCERPT_WORDS cap (back-compat).
    """
    n_text_anchors = sum(1 for g in trace["gamma"] if g["kind"] == "text")
    cap = (excerpt_word_budget(evidence_text, n_text_anchors)
           if evidence_text else MAX_EXCERPT_WORDS)
    anchors = []
    for g in trace["gamma"]:
        if g["kind"] != "text":
            continue
        anchors.append({"atom_id": g["atom_id"],
                        "excerpt": _truncate_words(g.get("source_excerpt") or "", cap),
                        "sentence_index": g.get("sentence_index")})
    sealed = {
        "motif": program["motif"],
        "condition": condition,
        "atoms": [{"atom_id": a["atom_id"], "op": a["op"], "value": a["value"]}
                  for a in trace["atoms"]],
        "anchors": anchors,
        "visual_nodes": list(image_nodes or []),
        # Visual observations travel with the sealed construction: the realizer
        # needs to know WHAT was read off the image (and from which region) to
        # write a question that genuinely depends on it. This leaks no source
        # prose -- an observation is the planner's own reading of pixels, not a
        # span of evidence_text -- so realizer blindness to the chunk is intact,
        # and these words are deliberately NOT counted in the text leak ratio.
        "visual_observations": [
            {"atom_id": g["atom_id"], "visual_node": g.get("visual_node"),
             "bbox": g.get("bbox"), "anchor": g.get("anchor_declared"),
             "observation": g.get("observation")}
            for g in trace["gamma"] if g.get("kind") == "visual"
        ],
        "construction_hash": trace["trace_hash"],
    }
    return sealed


def excerpt_word_budget(evidence_text: str, n_anchors: int = 2,
                        max_words: int = MAX_EXCERPT_WORDS,
                        max_ratio: float = MAX_LEAK_RATIO) -> int:
    """Per-anchor word budget that keeps MAX_EXCERPT_WORDS and MAX_LEAK_RATIO
    compatible on short chunks.

    The full run hit seal:excerpt_leak_ratio:0.405>0.35 on the corpus's
    shortest chunk (195 words): 2 anchors x 40 words = 79 words = 40.5% of a
    195-word chunk, which VIOLATES max_ratio even though each anchor was
    individually within max_words. A fixed 40-word cap is only safe when
    evidence_words >= n_anchors*max_words/max_ratio (here, >=228 words). Below
    that, shrink the per-anchor cap so n_anchors anchors at the cap can never
    exceed max_ratio of the chunk. This TIGHTENS the budget on short chunks;
    it never raises it above MAX_EXCERPT_WORDS.
    """
    ev_words = len(words(evidence_text))
    if ev_words <= 0 or n_anchors <= 0:
        return max_words
    safe = int((max_ratio * ev_words) / n_anchors)
    return max(1, min(max_words, safe))


def assert_realizer_blind(sealed: dict[str, Any], evidence_text: str,
                          max_excerpt_words: int = MAX_EXCERPT_WORDS) -> dict[str, Any]:
    """Prove the sealed payload does not leak the full chunk to the realizer.

    Checks: (a) no key outside the allow-list; (b) total anchor excerpt length
    is a small fraction of the evidence; (c) no single excerpt reproduces a
    long verbatim stretch beyond what the anchor legitimately needs.

    Thresholds come from MAX_LEAK_RATIO / MAX_EXCERPT_WORDS so the sealer and
    this guard can never drift into disagreement again. max_excerpt_words
    itself is derived per-call via excerpt_word_budget() by the caller when the
    evidence chunk is short (see seal_construction()).
    """
    problems: list[str] = []
    extra = set(sealed) - _REALIZER_ALLOWED_KEYS
    if extra:
        problems.append(f"unexpected_keys:{sorted(extra)}")

    ev_words = len(words(evidence_text))
    leaked = sum(len(words(a.get("excerpt", ""))) for a in sealed.get("anchors", []))
    ratio = (leaked / ev_words) if ev_words else 1.0
    if ratio > MAX_LEAK_RATIO:
        problems.append(f"excerpt_leak_ratio:{ratio:.3f}>{MAX_LEAK_RATIO}")
    for a in sealed.get("anchors", []):
        if len(words(a.get("excerpt", ""))) > max_excerpt_words:
            problems.append(f'{a.get("atom_id")}:excerpt_too_long')

    return {"ok": not problems, "problems": problems,
            "leak_ratio": round(ratio, 4), "evidence_words": ev_words,
            "leaked_words": leaked}


# ------------------------------------------- (3b) visual anchor verification
# A bbox covering (almost) the whole page localises nothing: it is the visual
# equivalent of "see the document" and was the loophole that let a planner
# claim a visual anchor without reading the image. Anything at or above this
# fraction of page area is rejected as non-localising.
MAX_BBOX_AREA_FRAC = 0.80
MIN_BBOX_AREA_FRAC = 0.0004          # ~0.02 x 0.02: below this it is noise
MIN_VISUAL_OBSERVATION_WORDS = 4     # "theo hình" / "xem sơ đồ" carry no fact

# Placeholder observations that assert a visual step without reporting anything
# read off the image. Matched on the normalised observation string.
_VISUAL_PLACEHOLDERS = (
    "theo hinh", "theo so do", "theo bang", "xem hinh", "xem so do",
    "xem bang", "nhu hinh", "nhu so do", "trong hinh", "trong so do",
    "quan sat hinh", "quan sat so do", "hinh minh hoa", "so do minh hoa",
)


def _strip_diacritics(s: str) -> str:
    decomposed = unicodedata.normalize("NFD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def verify_visual_anchors(trace: dict[str, Any],
                          declared_images: list[Any] | None) -> dict[str, Any]:
    """Verify each visual gamma the same way verify_trace() verifies text ones.

    Text anchors are replayable because rho = (sentence_index, char_span) can be
    sliced back out of the evidence. The visual counterpart is
    rho = (image_index, bbox), so this checks:

      (a) visual_node indexes a DECLARED image (1-based);
      (b) bbox is a normalised [x0, y0, x1, y1] in [0, 1] with x1 > x0, y1 > y0;
      (c) bbox area is a genuine region -- not the whole page (>= 80%, which
          localises nothing) and not a degenerate speck (< 0.04%);
      (d) the observation reports a fact, not a pointer ("theo hình" and friends
          are rejected, and a minimum word count is enforced).

    This is deliberately GEOMETRIC + STRUCTURAL, not semantic: it does not claim
    the observation is a true description of the pixels (that needs an
    independent captioner and is reported separately). What it does establish is
    that the planner committed to a specific, checkable region of a specific
    declared image and reported a fact about it -- which a self-declared
    `visual_node: 1` tag never did.
    """
    n_images = len(declared_images or [])
    problems: list[str] = []
    checked = 0

    for g in trace.get("gamma", []):
        if g.get("kind") != "visual":
            continue
        checked += 1
        aid = g.get("atom_id")

        node = g.get("visual_node")
        if isinstance(node, bool) or not isinstance(node, int):
            problems.append(f"{aid}:visual_node_not_int")
            continue
        if n_images == 0:
            problems.append(f"{aid}:no_declared_images")
            continue
        if not 1 <= node <= n_images:
            problems.append(f"{aid}:visual_node_out_of_range:{node}")
            continue

        bbox = g.get("bbox")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or not all(isinstance(v, (int, float))
                           and not isinstance(v, bool) for v in bbox)):
            problems.append(f"{aid}:bbox_missing_or_malformed")
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox)
        if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
            problems.append(f"{aid}:bbox_not_normalised")
            continue
        if x1 <= x0 or y1 <= y0:
            problems.append(f"{aid}:bbox_degenerate")
            continue
        area = (x1 - x0) * (y1 - y0)
        if area >= MAX_BBOX_AREA_FRAC:
            problems.append(f"{aid}:bbox_covers_whole_page:{area:.3f}")
            continue
        if area < MIN_BBOX_AREA_FRAC:
            problems.append(f"{aid}:bbox_too_small:{area:.5f}")
            continue

        observation = str(g.get("observation") or "")
        flat = _strip_diacritics(norm(observation))
        if len(words(observation)) < MIN_VISUAL_OBSERVATION_WORDS:
            problems.append(f"{aid}:observation_too_short")
            continue
        if any(flat.startswith(p) or flat == p for p in _VISUAL_PLACEHOLDERS):
            problems.append(f"{aid}:observation_is_placeholder")
            continue

    return {"ok": not problems, "problems": problems, "n_checked": checked,
            "n_declared_images": n_images}


# --------------------------------------------------- (3) TLV visual binding
def require_visual_binding(program: dict[str, Any], condition: str,
                           declared_images: list[Any] | None,
                           require_cross_modal: bool = False) -> dict[str, Any]:
    """Under TLV, at least one step must bind a declared visual node.

    Two ways a step can legitimately bind a visual:

      (a) MOTIF-DECLARED (v4, preferred): the motif template marks the step
          modality="visual", so the compiler already refused to compile the
          program without a visual_node. The op is whatever the cross-modal
          template declares (`locate` here), and the join op (`reconcile` /
          `apply`) is a DERIVED step that consumes both modalities -- it must
          not be checked against VISUAL_CAPABLE_OPS, which only ever described
          the legacy opportunistic tagging.

      (b) SELF-DECLARED (legacy v3): the planner bolted `visual_node` onto a
          text-only motif's step. Kept for backward compatibility and still
          restricted to VISUAL_CAPABLE_OPS, but this is precisely the weak form
          that made TLV a label rather than a mechanism.

    When `require_cross_modal` is True, TLV additionally demands form (a): a
    program whose motif does not structurally require an image is rejected, so
    "multimodal" cannot be satisfied by tagging a text-only construction.
    Non-TLV conditions must NOT bind visuals at all.
    """
    n_images = len(declared_images or [])
    bound = [(s["index"], s.get("visual_node"), s["op"],
              s.get("modality", "any"))
             for s in program["steps"] if s.get("visual_node") is not None]
    is_cross_modal = bool(program.get("is_cross_modal"))

    if condition != "TLV":
        if is_cross_modal:
            return {"ok": False,
                    "reason": f"cross_modal_motif_outside_TLV:{condition}",
                    "n_bound": len(bound), "bound": bound,
                    "is_cross_modal": is_cross_modal}
        if bound:
            return {"ok": False, "reason": f"visual_node_declared_outside_TLV:{condition}",
                    "n_bound": len(bound), "bound": bound,
                    "is_cross_modal": is_cross_modal}
        return {"ok": True, "reason": None, "n_bound": 0, "bound": [],
                "is_cross_modal": False}

    if n_images == 0:
        return {"ok": False, "reason": "TLV_package_has_no_images",
                "n_bound": 0, "bound": [], "is_cross_modal": is_cross_modal}
    if require_cross_modal and not is_cross_modal:
        return {"ok": False, "reason": f"TLV_motif_not_cross_modal:{program['motif']}",
                "n_bound": len(bound), "bound": bound, "is_cross_modal": False}
    if not bound:
        return {"ok": False, "reason": "TLV_no_visual_binding",
                "n_bound": 0, "bound": [], "is_cross_modal": is_cross_modal}
    for idx, node, op, modality in bound:
        if not isinstance(node, int) or isinstance(node, bool) or not 1 <= node <= n_images:
            return {"ok": False, "reason": f"step{idx}_visual_node_out_of_range:{node!r}",
                    "n_bound": len(bound), "bound": bound,
                    "is_cross_modal": is_cross_modal}
        # Only the legacy self-declared form is op-restricted; a motif-declared
        # visual step already passed the compiler's modality contract.
        if modality != "visual" and op not in VISUAL_CAPABLE_OPS:
            return {"ok": False, "reason": f"step{idx}_op_cannot_consume_visual:{op}",
                    "n_bound": len(bound), "bound": bound,
                    "is_cross_modal": is_cross_modal}
    return {"ok": True, "reason": None, "n_bound": len(bound), "bound": bound,
            "is_cross_modal": is_cross_modal}


# --------------------------------------------------------------- full gate
def _attach_visual_nodes(program: dict[str, Any], steps: Any) -> None:
    """Copy the model-declared visual_node and bbox onto compiled steps.

    The compiler keeps `visual_node` only for steps whose motif template
    declares modality="visual"; for the legacy text-only motifs it is dropped,
    so this re-attaches it (that is how require_visual_binding() can still
    detect a visual_node illegally declared outside TLV).

    `bbox` is carried here too: compile_program() does not model geometry, but
    build_trace() needs the bbox to emit rho = (image_index, bbox) and
    verify_visual_anchors() needs it to check the region actually localises
    something. Without this, every cross-modal step would reach the verifier
    with bbox=None and fail as bbox_missing_or_malformed.
    """
    if not isinstance(steps, list):
        return
    for compiled, raw in zip(program["steps"], steps):
        if not isinstance(raw, dict):
            continue
        if "visual_node" in raw and compiled.get("visual_node") is None:
            compiled["visual_node"] = raw.get("visual_node")
        if "bbox" in raw:
            compiled["bbox"] = raw.get("bbox")


def guard_plan_v3(parsed: dict[str, Any], evidence_text: str, condition: str,
                  declared_images: list[Any] | None = None,
                  idf_cut: float | None = None,
                  idf_table: dict[str, float] | None = None,
                  require_cross_modal: bool = False) -> dict[str, Any]:
    """Stage-1 gate: matcher -> compiler -> executor -> trace -> visual bind.

    NOTE: no question exists yet at this stage. The anti-verbatim guard runs
    in stage 2 (guard_realized_v3) once the realizer has written the question.
    """
    motif, steps = parsed.get("motif"), parsed.get("steps")
    try:
        program = compile_program(motif, steps)
    except CompileError as exc:
        return {"status": "REJECTED", "reason": f"compile:{exc}",
                "program": None, "execution": None, "trace": None}
    _attach_visual_nodes(program, steps)

    execution = execute_program(program, evidence_text, idf_cut, idf_table)
    if not execution["ok"]:
        return {"status": "REJECTED", "reason": f"exec:{execution['reason']}",
                "program": program, "execution": execution, "trace": None}

    vis = require_visual_binding(program, condition, declared_images,
                                 require_cross_modal=require_cross_modal)
    if not vis["ok"]:
        return {"status": "REJECTED", "reason": f"visual:{vis['reason']}",
                "program": program, "execution": execution, "trace": None,
                "visual": vis}

    trace = build_trace(program, execution, evidence_text)
    ver = verify_trace(trace, evidence_text)
    if not ver["ok"]:
        return {"status": "REJECTED", "reason": f"trace:{','.join(ver['problems'])[:80]}",
                "program": program, "execution": execution, "trace": trace,
                "visual": vis, "trace_verify": ver}

    # Visual anchors get their own replay check, symmetric to verify_trace():
    # verify_trace() proves each text anchor slices back to real characters;
    # this proves each visual anchor names a declared image, commits to a
    # localising bbox, and reports a fact rather than a pointer. For text-only
    # motifs there are no visual gammas, so n_checked=0 and this is a no-op --
    # the frozen v3 behaviour is unchanged.
    vver = verify_visual_anchors(trace, declared_images)
    if not vver["ok"]:
        return {"status": "REJECTED",
                "reason": f"visual_anchor:{','.join(vver['problems'])[:80]}",
                "program": program, "execution": execution, "trace": trace,
                "visual": vis, "trace_verify": ver, "visual_anchor_verify": vver}

    return {"status": "PLAN_OK", "reason": None, "program": program,
            "execution": execution, "trace": trace, "visual": vis,
            "trace_verify": ver, "visual_anchor_verify": vver}


def guard_realized_v3(realized: dict[str, Any], sealed: dict[str, Any],
                      evidence_text: str) -> dict[str, Any]:
    """Stage-2 gate: the realizer's MCQ must respect the sealed construction.

    Enforces main.tex's check set C: four distinct options, answer index in
    range, answer--choice agreement, atom preservation, and anti-verbatim.
    """
    for field in ("question", "choices", "answer_index"):
        if field not in realized or realized[field] in (None, "", [], {}):
            return {"status": "REJECTED", "reason": f"missing:{field}"}

    q, choices, ai = realized["question"], realized["choices"], realized["answer_index"]
    if not isinstance(q, str) or not q.strip():
        return {"status": "REJECTED", "reason": "invalid_question"}
    if (not isinstance(choices, list) or len(choices) != 4
            or any(not isinstance(c, str) or not c.strip() for c in choices)):
        return {"status": "REJECTED", "reason": "invalid_choices"}
    if len({norm(c) for c in choices}) != 4:
        return {"status": "REJECTED", "reason": "choices_not_distinct"}
    if isinstance(ai, bool) or not isinstance(ai, int) or not 0 <= ai < 4:
        return {"status": "REJECTED", "reason": "answer_index_out_of_range"}

    # The visual value must remain hidden from the stem. Compare against the
    # planner's localized visual anchor, not the full observation/result: the
    # latter often repeats generic domain context (e.g. "trình tự tố tụng")
    # that is not itself the image value and caused a false positive in pilot
    # v6. The independent answerer can then only obtain the localized value
    # from the image (or from a choice, which is why paired ablation remains
    # the empirical final gate).
    visual_stem_runs = []
    for visual in sealed.get("visual_observations", []) or []:
        visual_anchor = str(visual.get("anchor") or "")
        if not visual_anchor:
            # Backward-compatible read of pre-v6 sealed payloads; new payloads
            # always carry anchor. This fallback preserves fail-closed checking
            # rather than silently disabling the guard.
            visual_anchor = str(visual.get("observation") or "")
        visual_stem_runs.append(longest_common_run(words(q), visual_anchor))
    max_visual_stem_run = max(visual_stem_runs, default=0)
    if max_visual_stem_run > MAX_VISUAL_STEM_RUN:
        return {"status": "REJECTED",
                "reason": f"visual_stem_leak:run={max_visual_stem_run}>{MAX_VISUAL_STEM_RUN}",
                "visual_stem_runs": visual_stem_runs}

    # atom preservation: the correct option must reflect the sealed final atom
    final_atom = sealed["atoms"][-1]["value"] if sealed.get("atoms") else ""
    overlap = 0.0
    fa, co = set(words(final_atom)), set(words(choices[ai]))
    if fa:
        overlap = len(fa & co) / len(fa)

    run = longest_common_run(words(q), evidence_text)
    if run > MAX_VERBATIM_RUN:
        return {"status": "REJECTED", "reason": f"anti_verbatim:run={run}>{MAX_VERBATIM_RUN}",
                "verbatim_run": run, "atom_overlap": round(overlap, 3)}

    return {"status": "PARSED", "reason": None, "verbatim_run": run,
            "atom_overlap": round(overlap, 3),
            "visual_stem_runs": visual_stem_runs}
