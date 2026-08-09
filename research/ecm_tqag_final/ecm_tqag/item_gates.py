#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
item_gates_v1.py -- deterministic, API-free ITEM-LEVEL gates for ECM-TQAG.

Four gates, all pure stdlib:

  G-NOVELTY  reject when the claimed visual observation is already contained in
             the chunk OCR text (IDF-weighted content-word coverage >= tau).
             reason: "obs_covered_by_text"
  G-SUPPORT  every content token of the key must appear in the union of
             (anchor excerpts + visual observations + derived atom values).
             Numbers/years are matched EXACTLY (token equality, never substring).
             reason: "key_unsupported"
  G-UNIQUE   if >= 2 choices are supported by the anchor sentences at or above
             the key's support level, the item is not single-best-answer.
             reason: "multi_correct"
  G-CUE      reject stems carrying a generic image-deixis phrase, which is a
             formal cue rather than real visual dependence.
             reason: "generic_image_cue"

HISTORICAL PITFALL (v5): a lexical-novelty guard based on consecutive run-length
matching was rolled back because it rejected the one genuine VISUAL_DEPENDENT
item in runs/vd_v1. That item is therefore wired in here as a REGRESSION FIXTURE
(load_vd_fixture) and calibrate_tau() reports, for every candidate tau, whether
the fixture survives. No tau is recommended unless the fixture survives it.

Nothing in this module performs network I/O. run_ecm_v3.py and ecm_v3_core.py
are not imported and not modified.
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter

SCHEMA = "ecm-tqag.item-gates.v1"

# The confirmatory endpoint is determined by exactly these five gates. Lexical
# novelty remains available in the gate ledger, but is exploratory and must not
# alter the confirmatory verdict.
CONFIRMATORY_GATES = ("G-CUE", "G-SUPPORT", "G-UNIQUE", "CONTRACT", "SEAL")


def confirmatory_gate_verdict(gates: dict) -> dict:
    """Compose the frozen five-gate endpoint and fail closed on missing data."""
    for name in CONFIRMATORY_GATES:
        gate = gates.get(name)
        if not isinstance(gate, dict) or "ok" not in gate:
            raise ValueError(f"BLOCKED_GATES:missing_confirmatory_gate:{name}")
        if not isinstance(gate["ok"], bool):
            raise ValueError(f"BLOCKED_GATES:invalid_confirmatory_gate:{name}")
    failed = [name for name in CONFIRMATORY_GATES if not gates[name]["ok"]]
    return {"required": list(CONFIRMATORY_GATES), "failed": failed, "ok": not failed}

# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------
# [^\W_]+ = runs of Unicode word characters EXCLUDING underscore, so Vietnamese
# diacritics survive while punctuation, quotes, brackets and underscores act as
# separators. Digits are kept as their own tokens so years can be compared.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Explicit Vietnamese stopword list (function words, deixis, copulas,
# quantifiers). Declared here in full so the gate is auditable and reproducible.
STOPWORDS = frozenset("""
và hoặc là của cho với trong trên dưới ngoài từ đến về theo tại bởi bằng như
vào ra lên xuống qua sang cùng giữa sau trước khi mà thì nên nếu vì do bởi_vì
các những mọi mỗi một hai này đó ấy kia đây nào gì ai sao vậy thế
được bị có không chưa đã đang sẽ vẫn còn cũng rất quá lắm hơn nhất
để phải cần nên việc sự cái con chiếc người
tôi ta chúng họ nó mình bạn anh chị em
ở nữa chỉ thêm cả tất_cả đều song tuy nhưng vả lại thậm chí
hay khác trở nhằm chẳng hạn ví dụ tức
""".split())


def strip_accents_lower(s: str) -> str:
    """Lowercase + NFC normalise. Accents are PRESERVED (they are meaningful in
    Vietnamese); this only guards against NFD/NFC mismatch between files."""
    return unicodedata.normalize("NFC", s).lower()


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, return the token sequence."""
    if not text:
        return []
    return _TOKEN_RE.findall(strip_accents_lower(text))


def is_stopword(tok: str) -> bool:
    return tok in STOPWORDS


def content_tokens(text: str) -> list[str]:
    """Token sequence with stopwords removed (order and duplicates preserved)."""
    return [t for t in tokenize(text) if not is_stopword(t)]


def is_numeric(tok: str) -> bool:
    return tok.isdigit()


def _uniq(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# --------------------------------------------------------------------------
# IDF over the chunk corpus
# --------------------------------------------------------------------------
def build_idf(docs: list[str]) -> dict[str, float]:
    """Smoothed IDF:  idf(t) = ln((N + 1) / (df(t) + 1)).

    Non-negative for every observed token because df(t) <= N.
    """
    n = len(docs)
    df = Counter()
    for d in docs:
        for t in set(tokenize(d)):
            df[t] += 1
    return {t: math.log((n + 1.0) / (c + 1.0)) for t, c in df.items()}


def idf_of(idf: dict[str, float], tok: str, n_docs: int) -> float:
    """IDF of a token; unseen tokens get the maximum value ln(N + 1)."""
    if tok in idf:
        return idf[tok]
    return math.log((n_docs + 1.0) / 1.0)


# --------------------------------------------------------------------------
# G-NOVELTY
# --------------------------------------------------------------------------
def novelty_coverage(observation: str, chunk_text: str,
                     idf: dict[str, float], n_docs: int) -> dict:
    """IDF-weighted fraction of the observation's content vocabulary that also
    occurs in the chunk OCR text.

    coverage = sum(idf(t) for covered t) / sum(idf(t) for all t)

    coverage == 1.0 means the "visual" claim adds no lexical information beyond
    the text. An observation with no content tokens is treated as fully covered
    (it carries no novel information by construction).
    """
    obs_toks = _uniq(content_tokens(observation))
    text_toks = set(tokenize(chunk_text))
    if not obs_toks:
        return {"coverage": 1.0, "n_content": 0, "n_covered": 0,
                "missing": [], "covered": [],
                "idf_total": 0.0, "idf_covered": 0.0}
    idf_total = 0.0
    idf_cov = 0.0
    missing, covered = [], []
    for t in obs_toks:
        w = idf_of(idf, t, n_docs)
        idf_total += w
        if t in text_toks:
            idf_cov += w
            covered.append(t)
        else:
            missing.append(t)
    cov = (idf_cov / idf_total) if idf_total > 0 else 1.0
    return {"coverage": cov, "n_content": len(obs_toks),
            "n_covered": len(covered), "missing": missing, "covered": covered,
            "idf_total": idf_total, "idf_covered": idf_cov}


def gate_novelty(observations: list[str], chunk_text: str,
                 idf: dict[str, float], tau: float, n_docs: int) -> dict:
    """Reject when the visual observation is redundant with the OCR text.

    With several observations the ITEM coverage is the MINIMUM per-observation
    coverage, i.e. the item is only rejected when EVERY declared observation is
    redundant. That asymmetry is deliberate: over-rejection is exactly what got
    v5 rolled back, so a single genuinely novel observation rescues the item.
    """
    per = [novelty_coverage(o, chunk_text, idf, n_docs) for o in observations]
    if not per:
        # No visual observation at all: nothing for this gate to measure.
        return {"gate": "G-NOVELTY", "ok": True, "reason": None,
                "coverage": None, "tau": tau, "per_observation": [],
                "note": "no_visual_observation"}
    cov = min(p["coverage"] for p in per)
    ok = cov < tau
    return {"gate": "G-NOVELTY", "ok": ok,
            "reason": None if ok else "obs_covered_by_text",
            "coverage": cov, "tau": tau, "per_observation": per}


# --------------------------------------------------------------------------
# G-CUE
# --------------------------------------------------------------------------
IMAGE_CUE_PHRASES = (
    "dựa vào hình",
    "dựa vào ảnh",
    "theo hình",
    "xem sơ đồ",
    "trong hình",
    "hình ảnh",
)


def normalise_for_phrase(text: str) -> str:
    """Lowercased, punctuation-stripped, whitespace-collapsed form used for
    phrase matching, so 'trong Hình 3,' matches the cue 'trong hình'."""
    return " ".join(tokenize(text))


def gate_cue(stem: str) -> dict:
    norm = normalise_for_phrase(stem)
    matched = [p for p in IMAGE_CUE_PHRASES if p in norm]
    ok = not matched
    return {"gate": "G-CUE", "ok": ok,
            "reason": None if ok else "generic_image_cue",
            "matched": matched, "normalised_stem": norm}


# --------------------------------------------------------------------------
# G-SUPPORT
# --------------------------------------------------------------------------
def gate_support(key_text: str, anchors: list[str], observations: list[str],
                 atom_values: list[str]) -> dict:
    """Every content token of the key must occur in the evidence union.

    Matching is token-level, so numbers and years are compared EXACTLY:
    the token "100" is not supported by the token "1002".

    Two verdicts are returned:
      ok         -- union = anchors + observations + derived atom values
                    (the contract as specified)
      ok_strict  -- union = anchors + observations only, i.e. atom values are
                    not allowed to launder a fact the model itself invented.
    """
    key_toks = _uniq(content_tokens(key_text))
    strict_union = set()
    for s in list(anchors) + list(observations):
        strict_union |= set(tokenize(s))
    full_union = set(strict_union)
    for s in atom_values:
        full_union |= set(tokenize(s))

    offending = [t for t in key_toks if t not in full_union]
    offending_strict = [t for t in key_toks if t not in strict_union]
    ok = not offending
    return {
        "gate": "G-SUPPORT",
        "ok": ok,
        "reason": None if ok else "key_unsupported",
        "offending": offending,
        "offending_numeric": [t for t in offending if is_numeric(t)],
        "ok_strict": not offending_strict,
        "offending_strict": offending_strict,
        "offending_numeric_strict": [t for t in offending_strict if is_numeric(t)],
        "n_key_content_tokens": len(key_toks),
    }


# --------------------------------------------------------------------------
# G-UNIQUE
# --------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def choice_support(choice: str, anchor_sentences: list[str]) -> float:
    """Best per-sentence fraction of the choice's content tokens present in an
    anchor sentence. 0.0 when the choice has no content tokens."""
    toks = _uniq(content_tokens(choice))
    if not toks:
        return 0.0
    best = 0.0
    for s in anchor_sentences:
        stoks = set(tokenize(s))
        hit = sum(1 for t in toks if t in stoks)
        best = max(best, hit / float(len(toks)))
    return best


def gate_unique(choices: list[str], answer_index: int,
                anchors: list[str]) -> dict:
    """Reject when >= 2 choices reach the key's anchor-support level."""
    sents = []
    for a in anchors:
        sents.extend(_split_sentences(a) or [a])
    if not sents:
        sents = []
    support = [choice_support(c, sents) for c in choices]
    if not (0 <= answer_index < len(choices)):
        return {"gate": "G-UNIQUE", "ok": False, "reason": "bad_answer_index",
                "support": support, "key_support": None,
                "supported_at_or_above": []}
    key_sup = support[answer_index]
    eps = 1e-9
    at_or_above = [i for i, s in enumerate(support) if s >= key_sup - eps]
    ok = len(at_or_above) < 2
    return {"gate": "G-UNIQUE", "ok": ok,
            "reason": None if ok else "multi_correct",
            "support": support, "key_support": key_sup,
            "supported_at_or_above": sorted(at_or_above)}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_chunk_texts(manifest_path: str) -> dict[str, str]:
    """chunk_id -> evidence text, from the manifest packages (16 unique chunks)."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        man = json.load(fh)
    out = {}
    for p in man.get("packages", []):
        cid = p.get("chunk_id")
        txt = (p.get("evidence") or {}).get("text")
        if cid and txt and cid not in out:
            out[cid] = txt
    return out


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def item_view(row: dict) -> dict | None:
    """Flatten a PARSED result row into the fields the gates need."""
    pr = row.get("parsed_response") or {}
    sealed = row.get("sealed") or {}
    if not pr.get("choices"):
        return None
    return {
        "chunk_id": row.get("chunk_id"),
        "condition": row.get("condition"),
        "status": row.get("status"),
        "question": pr.get("question") or "",
        "choices": pr.get("choices") or [],
        "answer_index": pr.get("answer_index"),
        "anchors": [a.get("excerpt", "") for a in sealed.get("anchors", [])],
        "observations": [o.get("observation", "")
                         for o in sealed.get("visual_observations", [])],
        "atom_values": [a.get("value", "") for a in sealed.get("atoms", [])],
    }


def load_vd_fixture(vd_verdicts_path: str, source_results_path: str) -> dict:
    """The single genuine VISUAL_DEPENDENT item from runs/vd_v1, resolved back to
    its generating row in the source results file. This is the regression
    fixture that killed v5."""
    with open(vd_verdicts_path, "r", encoding="utf-8") as fh:
        v = json.load(fh)
    vds = [x for x in v.get("verdicts", []) if x.get("verdict") == "VISUAL_DEPENDENT"]
    if len(vds) != 1:
        raise RuntimeError(f"expected exactly 1 VISUAL_DEPENDENT, got {len(vds)}")
    vd = vds[0]
    rows = read_jsonl(source_results_path)
    cands = [r for r in rows
             if r.get("chunk_id") == vd["chunk_id"]
             and r.get("status") == "PARSED"
             and (r.get("parsed_response") or {}).get("choices")]
    if not cands:
        raise RuntimeError(f"no PARSED row for fixture chunk {vd['chunk_id']}")
    iv = item_view(cands[0])
    if iv is None:
        raise RuntimeError(f"fixture row for {vd['chunk_id']} has no choices")
    iv.update({"verdict": vd["verdict"],
               "item_fingerprint": vd.get("item_fingerprint"),
               "accuracy_with_image": vd.get("accuracy_with_image"),
               "accuracy_without_image": vd.get("accuracy_without_image"),
               "source_results": source_results_path,
               "n_candidate_rows": len(cands)})
    return iv


# --------------------------------------------------------------------------
# tau calibration
# --------------------------------------------------------------------------
TAU_GRID = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def calibrate_tau(manifest_path: str, v6_results_path: str,
                  vd_verdicts_path: str, v4_results_path: str,
                  tau_grid=TAU_GRID) -> dict:
    """Scan tau. For each tau report how many of the PARSED (degenerate) items
    G-NOVELTY rejects and whether the vd_v1 fixture survives.

    Recommendation rule: the LARGEST tau that rejects >= 4 of the degenerate
    items AND keeps the fixture. If no tau does both, recommended_tau is None
    and exists_tau_satisfying_both is False -- that is a real finding, not a
    failure of the scan.
    """
    chunks = load_chunk_texts(manifest_path)
    idf = build_idf(list(chunks.values()))
    n_docs = len(chunks)

    rows = read_jsonl(v6_results_path)
    parsed = [item_view(r) for r in rows if r.get("status") == "PARSED"]
    parsed = [p for p in parsed if p]

    fix = load_vd_fixture(vd_verdicts_path, v4_results_path)
    fix_cov = min(novelty_coverage(o, chunks[fix["chunk_id"]], idf, n_docs)["coverage"]
                  for o in fix["observations"]) if fix["observations"] else None

    deg_cov = []
    for p in parsed:
        txt = chunks.get(p["chunk_id"], "")
        c = (min(novelty_coverage(o, txt, idf, n_docs)["coverage"]
                 for o in p["observations"]) if p["observations"] else None)
        deg_cov.append({"chunk_id": p["chunk_id"], "coverage": c})

    scan = []
    for tau in tau_grid:
        n_rej = sum(1 for d in deg_cov
                    if d["coverage"] is not None and d["coverage"] >= tau)
        fixture_survives = (fix_cov is None) or (fix_cov < tau)
        scan.append({
            "tau": tau,
            "n_degenerate_total": len(deg_cov),
            "n_degenerate_rejected": n_rej,
            "fixture_survives": fixture_survives,
            "satisfies_both": (n_rej >= 4) and fixture_survives,
        })

    ok_taus = [r["tau"] for r in scan if r["satisfies_both"]]
    rec = max(ok_taus) if ok_taus else None
    return {
        "tau_grid": list(tau_grid),
        "scan": scan,
        "recommended_tau": rec,
        "exists_tau_satisfying_both": bool(ok_taus),
        "criterion": "reject >= 4 of 5 degenerate items AND keep vd_v1 fixture",
        "fixture": {
            "chunk_id": fix["chunk_id"],
            "verdict": fix["verdict"],
            "item_fingerprint": fix["item_fingerprint"],
            "observation": fix["observations"][0] if fix["observations"] else None,
            "coverage": fix_cov,
            "question": fix["question"],
        },
        "degenerate_coverage": deg_cov,
        "n_chunks_in_idf_corpus": n_docs,
    }


# --------------------------------------------------------------------------
# Full evaluation
# --------------------------------------------------------------------------
def evaluate_item(iv: dict, chunk_text: str, idf: dict, n_docs: int,
                  tau: float) -> dict:
    """Evaluate one item against the frozen endpoint.

    ``CONTRACT`` and ``SEAL`` are upstream deterministic verdicts produced by the
    construction path.  They must be carried explicitly in ``iv``; inferring them
    from a loosely named status would make a missing guard silently pass.
    """
    key = ""
    ai = iv.get("answer_index")
    if isinstance(ai, int) and 0 <= ai < len(iv["choices"]):
        key = iv["choices"][ai]
    g_nov = gate_novelty(iv["observations"], chunk_text, idf, tau, n_docs)
    g_cue = gate_cue(iv["question"])
    g_sup = gate_support(key, iv["anchors"], iv["observations"], iv["atom_values"])
    g_uni = gate_unique(iv["choices"], ai if isinstance(ai, int) else -1,
                        iv["anchors"])
    gates = {"G-NOVELTY": g_nov, "G-CUE": g_cue,
             "G-SUPPORT": g_sup, "G-UNIQUE": g_uni}
    for field, name in (("contract_gate", "CONTRACT"), ("seal_gate", "SEAL")):
        gate = iv.get(field)
        if not isinstance(gate, dict) or "ok" not in gate:
            raise ValueError(f"BLOCKED_GATES:missing_confirmatory_gate:{name}")
        if not isinstance(gate["ok"], bool):
            raise ValueError(f"BLOCKED_GATES:invalid_confirmatory_gate:{name}")
        gates[name] = dict(gate)

    verdict = confirmatory_gate_verdict(gates)
    exploratory_failed = ["G-NOVELTY"] if not g_nov["ok"] else []
    return {
        "chunk_id": iv["chunk_id"],
        "condition": iv["condition"],
        "status": iv["status"],
        "question": iv["question"],
        "key": key,
        "answer_index": ai,
        "gates": gates,
        "confirmatory_gate_verdict": verdict,
        # Backward-compatible aggregate fields refer to the frozen five-gate
        # endpoint. Lexical novelty remains visible but exploratory.
        "failed_gates": list(verdict["failed"]),
        "confirmatory_failed_gates": list(verdict["failed"]),
        "exploratory_failed_gates": exploratory_failed,
        "reasons": [gates[k].get("reason") for k in verdict["failed"]],
        "passes_confirmatory_gates": verdict["ok"],
        "passes_all_gates": verdict["ok"],
    }


def evaluate_all(manifest_path: str, v6_results_path: str,
                 vd_verdicts_path: str, v4_results_path: str,
                 out_path: str) -> dict:
    chunks = load_chunk_texts(manifest_path)
    idf = build_idf(list(chunks.values()))
    n_docs = len(chunks)

    cal = calibrate_tau(manifest_path, v6_results_path,
                        vd_verdicts_path, v4_results_path)
    if cal["recommended_tau"] is not None:
        tau_applied = cal["recommended_tau"]
        tau_note = "recommended tau from calibration"
    else:
        tau_applied = max(cal["tau_grid"])
        tau_note = ("NO tau satisfies both constraints; applying the largest tau "
                    "in the grid, which is the most fixture-protective choice. "
                    "G-NOVELTY verdicts below are reported at this tau and the "
                    "per-tau yield table is given separately.")

    rows = read_jsonl(v6_results_path)
    per_row = []
    for i, r in enumerate(rows):
        iv = item_view(r)
        if iv is None:
            per_row.append({
                "row_index": i,
                "chunk_id": r.get("chunk_id"),
                "condition": r.get("condition"),
                "status": r.get("status"),
                "contract_reason": r.get("reason"),
                "gates": None,
                "failed_gates": ["CONTRACT"],
                "reasons": ["not_parsed_by_construction_contract"],
                "passes_all_gates": False,
            })
            continue
        res = evaluate_item(iv, chunks.get(iv["chunk_id"], ""), idf, n_docs,
                            tau_applied)
        res["row_index"] = i
        res["contract_reason"] = r.get("reason")
        per_row.append(res)

    parsed_rows = [x for x in per_row if x["status"] == "PARSED"]
    yield_all = sum(1 for x in per_row if x["passes_all_gates"])
    yield_parsed = sum(1 for x in parsed_rows if x["passes_all_gates"])

    # yield as a function of tau (other three gates held fixed)
    yield_by_tau = []
    for tau in cal["tau_grid"]:
        n = 0
        for r in rows:
            iv = item_view(r)
            if iv is None:
                continue
            res = evaluate_item(iv, chunks.get(iv["chunk_id"], ""), idf,
                                n_docs, tau)
            if res["passes_all_gates"]:
                n += 1
        yield_by_tau.append({"tau": tau, "rows_passing_all_gates_of_16": n})

    gate_fail_counts = Counter()
    for x in parsed_rows:
        for g in x["failed_gates"]:
            gate_fail_counts[g] += 1

    out = {
        "schema": SCHEMA,
        "inputs": {
            "manifest": os.path.abspath(manifest_path),
            "v6_results": os.path.abspath(v6_results_path),
            "vd_v1_verdicts": os.path.abspath(vd_verdicts_path),
            "vd_v1_source_results": os.path.abspath(v4_results_path),
        },
        "corpus": {"n_chunks": n_docs, "idf_formula": "ln((N+1)/(df+1))",
                   "n_stopwords": len(STOPWORDS)},
        "tau_calibration": cal,
        "tau_applied": tau_applied,
        "tau_applied_note": tau_note,
        "aggregate": {
            "n_rows_total": len(rows),
            "n_rows_parsed_by_contract": len(parsed_rows),
            "n_rows_rejected_by_contract": len(rows) - len(parsed_rows),
            "true_yield_all_gates_of_16": yield_all,
            "true_yield_string": f"{yield_all}/16",
            "n_parsed_items_passing_all_gates": yield_parsed,
            "parsed_gate_failure_counts": dict(gate_fail_counts),
        },
        "yield_by_tau": yield_by_tau,
        "per_row": per_row,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    return out


def _default_paths(here: str) -> dict:
    return {
        "manifest": os.path.join(here, "ecm_inputs_final_v2.json"),
        "v6": os.path.join(here, "runs", "full_v6_dependency", "results.jsonl"),
        "vd": os.path.join(here, "runs", "vd_v1", "verdicts.json"),
        "v4": os.path.join(here, "runs", "full_v4", "results.jsonl"),
        "out": os.path.join(here, "audit", "item_gates_v1_eval.json"),
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = _default_paths(here)
    res = evaluate_all(p["manifest"], p["v6"], p["vd"], p["v4"], p["out"])
    cal = res["tau_calibration"]
    print(f"wrote {p['out']}")
    print(f"exists_tau_satisfying_both = {cal['exists_tau_satisfying_both']}")
    print(f"recommended_tau            = {cal['recommended_tau']}")
    print(f"fixture coverage           = {cal['fixture']['coverage']:.4f}")
    print(f"true yield                 = {res['aggregate']['true_yield_string']}")
