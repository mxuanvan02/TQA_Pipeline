"""Generic OpenAI-compatible response extraction.

Every arm reads its model output through this one function, so an envelope that a
provider returned in an unexpected shape is rejected in exactly one place with a
stable reason string instead of being silently coerced somewhere downstream.

Two shapes are accepted, because OpenAI-compatible providers disagree:

  * ``choices[0].message.content`` as a plain string;
  * the same field as a list of content parts, in which case the ``text`` parts
    are concatenated in order.

Everything else fails closed. In particular, ``finish_reason == "length"`` is a
rejection by default: a truncated completion usually still parses as text but is
not a complete JSON object, and treating it as valid output is how a silent
schema failure becomes a silent data failure. The caller can opt into keeping the
partial body with ``allow_truncated=True`` when it only wants to log it.

Pure: no network I/O, no ledger writes, no transport state.
"""
from __future__ import annotations

from typing import Any

TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})


def _fail(reason: str) -> ValueError:
    return ValueError(f"envelope:{reason}")


def _first_choice(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise _fail(f"not_object:{type(body).__name__}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _fail("missing_choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise _fail(f"choice_not_object:{type(choice).__name__}")
    return choice


def response_content(body: Any, *, allow_truncated: bool = False) -> str:
    """Return the assistant text from an OpenAI-compatible chat completion.

    Raises ``ValueError('envelope:<reason>')`` for every malformed shape; the
    reason strings are stable and asserted on by the tests so the ledger can
    record which contract a provider broke.
    """
    choice = _first_choice(body)

    finish_reason = choice.get("finish_reason")
    if (
        not allow_truncated
        and isinstance(finish_reason, str)
        and finish_reason in TRUNCATION_FINISH_REASONS
    ):
        raise _fail(f"truncated_response:{finish_reason}")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise _fail("missing_message")

    raw = message.get("content")
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        parts: list[str] = []
        for part in raw:
            if not isinstance(part, dict):
                continue
            fragment = part.get("text")
            if isinstance(fragment, str):
                parts.append(fragment)
        if not parts:
            raise _fail("content_not_text:no_text_parts")
        text = "".join(parts)
    else:
        raise _fail(f"content_not_text:{type(raw).__name__}")

    if not text.strip():
        raise _fail("empty_content")
    return text


def response_usage(body: Any) -> dict[str, Any] | None:
    """Return the usage object when the provider supplied one as an object.

    Usage is accounting metadata, not output, so a provider that omits it or
    returns it in a non-object form must not break a run: this reports ``None``
    rather than raising.
    """
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    return dict(usage) if isinstance(usage, dict) else None


def response_model(body: Any) -> str | None:
    """Return the model identity the provider claims it served, if any."""
    if not isinstance(body, dict):
        return None
    model = body.get("model")
    return model if isinstance(model, str) and model else None


__all__ = [
    "TRUNCATION_FINISH_REASONS",
    "response_content",
    "response_model",
    "response_usage",
]
