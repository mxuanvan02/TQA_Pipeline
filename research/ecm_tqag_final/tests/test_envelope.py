from __future__ import annotations

import pytest

from ecm_tqag.run.envelope import response_content, response_usage


def _body(content, **extra):
    choice = {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
    choice.update(extra)
    return {"id": "x", "model": "m", "choices": [choice]}


def test_extracts_plain_string_content() -> None:
    assert response_content(_body('{"status":"ok"}')) == '{"status":"ok"}'


def test_extracts_multipart_content_by_concatenating_text_parts() -> None:
    body = _body([{"type": "text", "text": '{"a":'}, {"type": "text", "text": "1}"}])
    assert response_content(body) == '{"a":1}'


def test_rejects_truncated_completions_by_default() -> None:
    body = _body('{"a": 1', finish_reason="length")
    with pytest.raises(ValueError, match="envelope:truncated_response"):
        response_content(body)
    assert response_content(body, allow_truncated=True) == '{"a": 1'


def test_fails_closed_on_every_malformed_envelope_shape() -> None:
    cases = {
        "envelope:not_object": [],
        "envelope:missing_choices": {"id": "x"},
        "envelope:missing_choices": {"id": "x", "choices": []},
        "envelope:choice_not_object": {"choices": ["text"]},
        "envelope:missing_message": {"choices": [{"finish_reason": "stop"}]},
        "envelope:content_not_text": _body(17),
        "envelope:empty_content": _body("   "),
    }
    for reason, body in cases.items():
        with pytest.raises(ValueError, match=reason):
            response_content(body)


def test_multipart_content_without_text_parts_is_rejected() -> None:
    with pytest.raises(ValueError, match="envelope:content_not_text"):
        response_content(_body([{"type": "image_url", "image_url": {"url": "x"}}]))


def test_usage_is_returned_only_when_it_is_an_object() -> None:
    body = _body("ok")
    assert response_usage(body) is None
    body["usage"] = {"prompt_tokens": 3, "completion_tokens": 4}
    assert response_usage(body) == {"prompt_tokens": 3, "completion_tokens": 4}
    body["usage"] = "3 tokens"
    assert response_usage(body) is None
