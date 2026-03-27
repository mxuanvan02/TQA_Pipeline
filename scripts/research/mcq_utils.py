#!/usr/bin/env python3
"""
Utilities for normalizing MCQ candidates and resolving ground-truth answers.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


_CHOICE_LABEL_RE = re.compile(r"^\s*([A-D])[\s\.\):\-]+", re.IGNORECASE)
_GROUND_TRUTH_TAG_RE = re.compile(r"<ground_truth>(.*?)(?:</ground_truth>|$)", re.IGNORECASE | re.DOTALL)
_TRAILING_NOTE_RE = re.compile(
    r"\s*\((?:full correct answer text|đây là đáp án.*?|.*?correct answer text.*?)\)\s*$",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def extract_choice_label(text: str) -> str | None:
    match = _CHOICE_LABEL_RE.match(str(text).strip())
    if not match:
        return None
    return match.group(1).upper()


def strip_choice_label(text: str) -> str:
    return _CHOICE_LABEL_RE.sub("", str(text).strip(), count=1).strip()


def strip_trailing_notes(text: str) -> str:
    return _TRAILING_NOTE_RE.sub("", str(text).strip()).strip()


def extract_embedded_ground_truth(text: str) -> str:
    value = str(text).strip()
    matches = _GROUND_TRUTH_TAG_RE.findall(value)
    if matches:
        value = matches[-1].strip()
    return value.replace("</ground_truth>", "").strip()


def canonicalize_text(text: str) -> str:
    return normalize_text(strip_trailing_notes(strip_choice_label(extract_embedded_ground_truth(text))))


def build_ground_truth_variants(ground_truth: str) -> list[str]:
    raw = str(ground_truth).strip()
    variants: list[str] = []

    for candidate in (
        raw,
        extract_embedded_ground_truth(raw),
        strip_trailing_notes(raw),
        strip_trailing_notes(extract_embedded_ground_truth(raw)),
        strip_choice_label(raw),
        strip_choice_label(extract_embedded_ground_truth(raw)),
        canonicalize_text(raw),
    ):
        candidate = str(candidate).strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def resolve_ground_truth_index(
    candidate_answers: list[str] | object,
    ground_truth: str,
    *,
    allow_fuzzy: bool = True,
    similarity_threshold: float = 0.96,
    similarity_margin: float = 0.05,
) -> dict:
    if not isinstance(candidate_answers, list):
        return {"matched_index": None, "method": "candidates_not_list", "candidate_count": 0}

    candidates = [str(x).strip() for x in candidate_answers]
    candidate_count = len(candidates)
    if not candidates:
        return {"matched_index": None, "method": "no_candidates", "candidate_count": 0}

    full_norm = [normalize_text(x) for x in candidates]
    payload_norm = [canonicalize_text(x) for x in candidates]
    label_to_index = {}
    for idx, cand in enumerate(candidates):
        label = extract_choice_label(cand)
        if label and label not in label_to_index:
            label_to_index[label] = idx

    variants = build_ground_truth_variants(ground_truth)
    for variant in variants:
        norm_variant = normalize_text(variant)
        if norm_variant in full_norm:
            return {
                "matched_index": full_norm.index(norm_variant),
                "method": "exact_full",
                "candidate_count": candidate_count,
            }

    for variant in variants:
        norm_variant = canonicalize_text(variant)
        if norm_variant in payload_norm:
            return {
                "matched_index": payload_norm.index(norm_variant),
                "method": "payload_match",
                "candidate_count": candidate_count,
            }

    # Fallback to label-only only if the ground truth is effectively just the label.
    raw = str(ground_truth).strip()
    raw_label = extract_choice_label(raw)
    raw_payload = canonicalize_text(raw)
    if raw_label and raw_label in label_to_index and raw_payload == "":
        return {
            "matched_index": label_to_index[raw_label],
            "method": "label_only",
            "candidate_count": candidate_count,
        }

    if allow_fuzzy:
        for variant in variants:
            norm_variant = canonicalize_text(variant)
            if not norm_variant:
                continue
            sims = [SequenceMatcher(None, norm_variant, payload).ratio() for payload in payload_norm]
            best_idx = max(range(len(sims)), key=lambda i: sims[i])
            best_score = sims[best_idx]
            second_score = sorted(sims, reverse=True)[1] if len(sims) > 1 else 0.0
            if best_score >= similarity_threshold and (best_score - second_score) >= similarity_margin:
                return {
                    "matched_index": best_idx,
                    "method": "fuzzy_payload",
                    "candidate_count": candidate_count,
                }

    return {"matched_index": None, "method": "unresolved", "candidate_count": candidate_count}
