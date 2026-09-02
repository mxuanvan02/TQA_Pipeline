"""
Quality signals for AI-generated QA records (dataset triage / tuning)
=====================================================================
Who:    Imported by scripts/research/audit_qa_quality.py and (optionally) by
        Stage 3/4 to attach a quality stamp alongside ``_provenance``.
Where:  src/quality_signals.py
Why:    Provenance answers "who made this and was it tampered?".
        THIS module answers "is this record good, and if not, WHERE is it weak?"
        so the dataset can be filtered, hand-fixed, or regenerated to raise
        downstream benchmark accuracy.

Design:
    - Pure standard library + existing repo helpers (mcq_utils, language_sanity).
    - Deterministic and cheap: NO GPU, NO LLM call. Runs on a laptop over the
      full dataset in seconds, so it is safe to run every build / in CI.
    - Every record gets a list of machine-readable ``flags`` (severity-tagged)
      plus a single ``triage`` decision: keep | review | regenerate | drop.
    - The LLM-judge scores (Stage 4) are complementary; when present they are
      folded in, but this module does not require them.

Attached under the reserved key ``_quality`` so it sits beside ``_provenance``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reuse the repo's own resolvers so triage matches how the paper scores answers.
_HERE = Path(__file__).resolve().parent
_RESEARCH = _HERE.parent / "scripts" / "research"
for _p in (str(_RESEARCH), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from mcq_utils import resolve_ground_truth_index, extract_choice_label, canonicalize_text
except Exception:  # pragma: no cover - fallback if layout differs
    resolve_ground_truth_index = None  # type: ignore
    extract_choice_label = None  # type: ignore
    canonicalize_text = None  # type: ignore

try:
    from language_sanity import qa_artifact_reasons
except Exception:  # pragma: no cover
    try:
        from src.language_sanity import qa_artifact_reasons  # type: ignore
    except Exception:
        qa_artifact_reasons = None  # type: ignore

QUALITY_SCHEMA_VERSION = "tqa.quality.v1"
QUALITY_KEY = "_quality"

# Severity ranking (higher = worse). Drives the triage decision.
SEV_INFO = "info"
SEV_WARN = "warn"
SEV_ERROR = "error"
_SEV_RANK = {SEV_INFO: 0, SEV_WARN: 1, SEV_ERROR: 2}

# Tunable thresholds — conservative defaults, override per audit run.
@dataclass
class QualityThresholds:
    expected_candidates: int = 4
    min_question_chars: int = 12
    min_rationale_chars: int = 40
    min_context_chars: int = 80
    # Rationale should show the legal-syllogism skeleton the prompt asked for.
    syllogism_markers: tuple[str, ...] = ("tiền đề", "kết luận")


def _text(x: Any) -> str:
    return "" if x is None else str(x)


def _flag(flags: list[dict[str, str]], code: str, severity: str, detail: str = "") -> None:
    flags.append({"code": code, "severity": severity, "detail": detail})


def compute_quality_signals(
    record: dict[str, Any],
    thresholds: QualityThresholds | None = None,
) -> dict[str, Any]:
    """
    Compute deterministic quality signals for one QA record.

    Returns a dict (also suitable to attach under ``_quality``) with:
      - flags:   list of {code, severity, detail}
      - metrics: raw measurements (lengths, candidate count, gt match method)
      - triage:  keep | review | regenerate | drop
    """
    th = thresholds or QualityThresholds()
    flags: list[dict[str, str]] = []
    metrics: dict[str, Any] = {}

    question = _text(record.get("question_content"))
    rationale = _text(record.get("legal_rationale"))
    context = _text(record.get("context_text"))
    ground_truth = _text(record.get("ground_truth"))
    candidates = record.get("candidate_answers")
    candidates = candidates if isinstance(candidates, list) else []

    metrics["question_chars"] = len(question.strip())
    metrics["rationale_chars"] = len(rationale.strip())
    metrics["context_chars"] = len(context.strip())
    metrics["num_candidates"] = len(candidates)

    # 1) Ground-truth must resolve to exactly one candidate (the #1 accuracy killer).
    if resolve_ground_truth_index is not None:
        res = resolve_ground_truth_index(candidates, ground_truth)
        metrics["gt_match_method"] = res.get("method")
        metrics["gt_matched_index"] = res.get("matched_index")
        if res.get("matched_index") is None:
            _flag(flags, "gt_unresolved", SEV_ERROR,
                  f"ground_truth does not map to any candidate (method={res.get('method')})")
        elif res.get("method") == "fuzzy_payload":
            _flag(flags, "gt_fuzzy_only", SEV_WARN,
                  "ground_truth matches a candidate only by fuzzy similarity")
    else:
        metrics["gt_match_method"] = "resolver_unavailable"

    # 2) Candidate count / duplicates / labeling.
    if len(candidates) != th.expected_candidates:
        _flag(flags, "candidate_count", SEV_ERROR if len(candidates) < 2 else SEV_WARN,
              f"expected {th.expected_candidates}, got {len(candidates)}")
    norm_cands = []
    if canonicalize_text is not None:
        norm_cands = [canonicalize_text(c) for c in candidates]
        nonempty = [c for c in norm_cands if c]
        if len(set(nonempty)) < len(nonempty):
            _flag(flags, "duplicate_candidates", SEV_ERROR, "two candidates are identical after normalization")
    if extract_choice_label is not None and candidates:
        labels = [extract_choice_label(c) for c in candidates]
        if any(l is None for l in labels):
            _flag(flags, "candidate_unlabeled", SEV_WARN, "a candidate is missing its A/B/C/D label")

    # 3) Foreign-language / prompt-leak artifacts (reuse repo detector).
    if qa_artifact_reasons is not None:
        reasons = qa_artifact_reasons(question, candidates, ground_truth, rationale)
        if reasons:
            metrics["artifact_reasons"] = reasons
            sev = SEV_ERROR if ("cjk_chars" in reasons or "prompt_leak" in reasons) else SEV_WARN
            _flag(flags, "language_artifact", sev, ",".join(reasons))

    # 4) Degenerate lengths.
    if metrics["question_chars"] < th.min_question_chars:
        _flag(flags, "question_too_short", SEV_WARN, f"{metrics['question_chars']} chars")
    if metrics["rationale_chars"] < th.min_rationale_chars:
        _flag(flags, "rationale_too_short", SEV_WARN, f"{metrics['rationale_chars']} chars")
    if metrics["context_chars"] < th.min_context_chars:
        _flag(flags, "context_too_short", SEV_WARN, f"{metrics['context_chars']} chars")

    # 5) Legal-syllogism skeleton presence (prompt explicitly asks for it).
    low = rationale.lower()
    if not any(m in low for m in th.syllogism_markers):
        _flag(flags, "missing_syllogism", SEV_WARN, "no 'tiền đề'/'kết luận' markers in rationale")

    # 6) Multimodal consistency.
    is_mm = bool(record.get("is_multimodal", False))
    visuals = record.get("context_visuals") or []
    if is_mm and not visuals:
        _flag(flags, "multimodal_no_visual", SEV_WARN, "is_multimodal=True but context_visuals is empty")

    # 7) Fold in Stage-4 judge scores if they were persisted on the record.
    scores = record.get("eval_scores") or {}
    if scores:
        metrics["judge"] = {
            "groundedness": scores.get("groundedness"),
            "legal_fluency": scores.get("legal_fluency"),
            "multimodal_alignment": scores.get("multimodal_alignment"),
            "overall": scores.get("overall"),
        }
        if scores.get("groundedness") == 0.0:
            _flag(flags, "judge_ungrounded", SEV_ERROR, "Stage-4 judge marked answer unsupported by context")
        if scores.get("legal_fluency") == 0.0:
            _flag(flags, "judge_illogical", SEV_WARN, "Stage-4 judge marked rationale illogical")

    # Triage decision from worst severity + which codes fired.
    worst = max((_SEV_RANK[f["severity"]] for f in flags), default=-1)
    codes = {f["code"] for f in flags}
    if "gt_unresolved" in codes or "duplicate_candidates" in codes or "judge_ungrounded" in codes:
        triage = "regenerate"
    elif worst >= _SEV_RANK[SEV_ERROR]:
        triage = "regenerate"
    elif worst == _SEV_RANK[SEV_WARN]:
        triage = "review"
    else:
        triage = "keep"

    return {
        "schema": QUALITY_SCHEMA_VERSION,
        "flags": flags,
        "metrics": metrics,
        "triage": triage,
    }


def stamp_quality(record: dict[str, Any], thresholds: QualityThresholds | None = None) -> dict[str, Any]:
    """Attach quality signals under ``_quality`` (in place) and return the record."""
    record[QUALITY_KEY] = compute_quality_signals(record, thresholds)
    return record


if __name__ == "__main__":
    # Minimal self-test on synthetic records.
    good = {
        "question_content": "Theo Bộ luật Tố tụng Hình sự, ai có thẩm quyền phê chuẩn lệnh bắt?",
        "candidate_answers": ["A. Viện kiểm sát.", "B. Tòa án.", "C. Cơ quan điều tra.", "D. Ủy ban nhân dân."],
        "ground_truth": "A. Viện kiểm sát.",
        "legal_rationale": "Đại tiền đề: Theo luật... Tiểu tiền đề: Trong trường hợp này... Kết luận: Vì vậy Viện kiểm sát phê chuẩn.",
        "context_text": "Điều 113. Việc bắt bị can để tạm giam phải có phê chuẩn của Viện kiểm sát cùng cấp trước khi thi hành..." * 1,
        "is_multimodal": False,
    }
    bad = {
        "question_content": "?",
        "candidate_answers": ["A. x", "A. x"],
        "ground_truth": "C. một đáp án không có trong danh sách",
        "legal_rationale": "根据法律规定，答案是C。",
        "context_text": "short",
        "is_multimodal": True,
        "context_visuals": [],
    }
    for name, rec in (("good", good), ("bad", bad)):
        q = compute_quality_signals(rec)
        print(name, "→ triage:", q["triage"], "| flags:", [f["code"] for f in q["flags"]])
    assert compute_quality_signals(good)["triage"] in ("keep", "review")
    assert compute_quality_signals(bad)["triage"] == "regenerate"
    print("quality_signals self-test OK")
