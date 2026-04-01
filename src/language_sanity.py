"""
Helpers for conservative Vietnamese-only release sanitation.

The generation prompt asks for Vietnamese outputs, but small multilingual or
prompt-leak artifacts can still appear in synthetic fields. These helpers
remove known harmless scaffolding and flag records that still contain foreign
language or instruction leakage after cleanup.
"""

from __future__ import annotations

import re
from typing import Iterable

_SCAFFOLD_NOTE_RE = re.compile(
    r"\s*\((?:full correct answer text|correct answer text)\)\s*",
    re.IGNORECASE,
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_META_RE = re.compile(
    r"\b(correct answer is|because|therefore|in this case)\b",
    re.IGNORECASE,
)
_PROMPT_LEAK_RE = re.compile(
    r"(<qa_pair>|Bloom's Taxonomy|only the xml tags|XML格式|根据提供的背景信息)",
    re.IGNORECASE,
)


def sanitize_generated_text(text: str) -> str:
    """Remove known English scaffolding and normalize whitespace."""
    cleaned = _SCAFFOLD_NOTE_RE.sub("", str(text))
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    return cleaned.strip()


def detect_foreign_artifacts(text: str) -> list[str]:
    """Return conservative artifact reasons after sanitation."""
    reasons: list[str] = []
    cleaned = sanitize_generated_text(text)
    if _CJK_RE.search(cleaned):
        reasons.append("cjk_chars")
    if _EN_META_RE.search(cleaned):
        reasons.append("english_meta")
    if _PROMPT_LEAK_RE.search(cleaned):
        reasons.append("prompt_leak")
    return reasons


def qa_artifact_reasons(
    question: str,
    candidate_answers: Iterable[str],
    ground_truth: str,
    legal_rationale: str,
) -> list[str]:
    """Aggregate artifact reasons over the main synthetic QA fields."""
    merged = " ".join(
        [
            sanitize_generated_text(question),
            sanitize_generated_text(ground_truth),
            sanitize_generated_text(legal_rationale),
            *(sanitize_generated_text(x) for x in candidate_answers),
        ]
    )
    seen = set()
    reasons: list[str] = []
    for reason in detect_foreign_artifacts(merged):
        if reason not in seen:
            reasons.append(reason)
            seen.add(reason)
    return reasons
