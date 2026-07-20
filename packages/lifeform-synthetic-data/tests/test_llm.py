from __future__ import annotations

import io
import json
import urllib.error

import pytest

from lifeform_synthetic_data.llm import (
    LLMAuthenticationError,
    LLMQuotaError,
    LLMResponseError,
    OpenAICompatibleConfig,
    OpenAICompatibleJsonClient,
    RateCard,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.headers = {"x-request-id": "request-1"}

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback

    def read(self) -> bytes:
        return self._body


def _client(*, attempts: int = 3) -> OpenAICompatibleJsonClient:
    return OpenAICompatibleJsonClient(
        OpenAICompatibleConfig(
            base_url="https://example.invalid/v1",
            api_key="test-secret",
            model_id="test-model",
            rate_card=RateCard(1.0, 2.0),
            max_attempts=attempts,
            initial_backoff_seconds=0.0,
        )
    )


def _success() -> FakeHTTPResponse:
    return FakeHTTPResponse(
        {
            "id": "completion-1",
            "model": "test-model",
            "choices": [{"message": {"content": json.dumps({"trajectory_id": "t", "slots": []})}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )


def _http_error(code: int, body: dict[str, object]) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.invalid/v1/chat/completions",
        code=code,
        msg="failure",
        hdrs={},
        fp=io.BytesIO(json.dumps(body).encode("utf-8")),
    )


def test_transient_http_error_retries_then_succeeds(monkeypatch) -> None:
    responses = [
        _http_error(503, {"error": {"type": "service_unavailable"}}),
        _success(),
    ]

    def fake_urlopen(request, timeout):
        del request, timeout
        value = responses.pop(0)
        if isinstance(value, urllib.error.HTTPError):
            raise value
        return value

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    completion = _client().complete_json(
        system_prompt="system",
        user_prompt="user",
    )

    assert completion.request_id == "request-1"
    assert completion.usage.total_tokens == 15
    assert responses == []


def test_authentication_error_does_not_retry_or_hide(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        del request, timeout
        calls += 1
        raise _http_error(401, {"error": {"type": "invalid_api_key"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(LLMAuthenticationError):
        _client().complete_json(system_prompt="system", user_prompt="user")

    assert calls == 1


def test_quota_denial_on_429_is_definitive(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        raise _http_error(
            429,
            {
                "error": {
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            },
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(LLMQuotaError):
        _client().complete_json(system_prompt="system", user_prompt="user")


def test_malformed_usage_fails_response_contract(monkeypatch) -> None:
    response = FakeHTTPResponse(
        {
            "model": "test-model",
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": "10"},
        }
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: response,
    )

    with pytest.raises(LLMResponseError, match="token counts"):
        _client().complete_json(system_prompt="system", user_prompt="user")
