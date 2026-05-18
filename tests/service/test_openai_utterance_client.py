"""Unit tests for ``lifeform_service.openai_utterance_client``.

We focus on:

* config validation: empty fields raise ValueError loud at construction
  (no silent fallback);
* env wiring: ``build_utterance_client_from_env`` honours
  ``PROTOCOL_LLM_*`` the same way the JSON-mode sibling does;
* prompt wire format: the body sent to the upstream OpenAI-compat
  endpoint contains the expected fields (temperature, seed, max_tokens,
  the two messages, the bearer header) and decodes choices[0].content
  correctly;
* fail-loud paths: HTTP error, non-JSON envelope, missing choices,
  empty content — all raise ``RuntimeError`` with the cause attached.

The test never opens a real socket; we monkeypatch ``urllib.request``
to capture the outgoing payload and return canned JSON.
"""

from __future__ import annotations

import io
import json
from typing import Any
from urllib import error as urllib_error

import pytest

from lifeform_core import OpenAiCompatConfig
from lifeform_service.openai_utterance_client import (
    OpenAiUtteranceClient,
    build_utterance_client_from_env,
)


# ---------------------------------------------------------------------------
# OpenAiUtteranceClient construction + complete()
# ---------------------------------------------------------------------------


def _good_config() -> OpenAiCompatConfig:
    return OpenAiCompatConfig(
        base_url="https://example.com/v1",
        api_key="sk-test",
        model="qwen-plus",
        timeout_seconds=5.0,
    )


def test_construct_rejects_empty_fields():
    with pytest.raises(ValueError):
        OpenAiUtteranceClient(
            OpenAiCompatConfig(
                base_url=" ", api_key="x", model="x"
            )
        )
    with pytest.raises(ValueError):
        OpenAiUtteranceClient(
            OpenAiCompatConfig(
                base_url="https://example.com/v1",
                api_key=" ",
                model="x",
            )
        )
    with pytest.raises(ValueError):
        OpenAiUtteranceClient(
            OpenAiCompatConfig(
                base_url="https://example.com/v1",
                api_key="sk",
                model=" ",
            )
        )


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _stub_urlopen(monkeypatch, captured: dict[str, Any], envelope: dict[str, Any]):
    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.headers.items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps(envelope).encode("utf-8"))

    monkeypatch.setattr(
        "lifeform_service.openai_utterance_client.urllib.request.urlopen",
        fake_urlopen,
    )


def test_complete_round_trip_returns_content_string(monkeypatch):
    client = OpenAiUtteranceClient(_good_config())
    captured: dict[str, Any] = {}
    _stub_urlopen(
        monkeypatch,
        captured,
        {
            "choices": [
                {"message": {"content": "  hi there  "}, "index": 0}
            ]
        },
    )
    out = client.complete(
        system_prompt="sys",
        user_prompt="usr",
        temperature=0.5,
        seed=42,
    )
    assert out == "hi there"
    # Wire format.
    assert captured["method"] == "POST"
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "qwen-plus"
    assert captured["body"]["temperature"] == 0.5
    assert captured["body"]["seed"] == 42
    assert captured["body"]["max_tokens"] == 256
    # No response_format — plain text mode.
    assert "response_format" not in captured["body"]
    msgs = captured["body"]["messages"]
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_complete_raises_on_http_error(monkeypatch):
    client = OpenAiUtteranceClient(_good_config())

    def boom(request, timeout=None):
        raise urllib_error.HTTPError(
            request.full_url,
            500,
            "Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b'{"error":"upstream"}'),
        )

    monkeypatch.setattr(
        "lifeform_service.openai_utterance_client.urllib.request.urlopen",
        boom,
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        client.complete(
            system_prompt="sys",
            user_prompt="usr",
            temperature=0.0,
            seed=0,
        )


def test_complete_raises_on_missing_choices(monkeypatch):
    client = OpenAiUtteranceClient(_good_config())
    captured: dict[str, Any] = {}
    _stub_urlopen(monkeypatch, captured, {})
    with pytest.raises(RuntimeError, match="missing 'choices'"):
        client.complete(
            system_prompt="sys",
            user_prompt="usr",
            temperature=0.0,
            seed=0,
        )


def test_complete_raises_on_empty_content(monkeypatch):
    client = OpenAiUtteranceClient(_good_config())
    captured: dict[str, Any] = {}
    _stub_urlopen(
        monkeypatch,
        captured,
        {"choices": [{"message": {"content": "   "}}]},
    )
    with pytest.raises(RuntimeError, match="missing 'content'"):
        client.complete(
            system_prompt="sys",
            user_prompt="usr",
            temperature=0.0,
            seed=0,
        )


def test_complete_raises_on_non_json_envelope(monkeypatch):
    client = OpenAiUtteranceClient(_good_config())

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(b"not json")

    monkeypatch.setattr(
        "lifeform_service.openai_utterance_client.urllib.request.urlopen",
        fake_urlopen,
    )
    with pytest.raises(RuntimeError, match="non-JSON envelope"):
        client.complete(
            system_prompt="sys",
            user_prompt="usr",
            temperature=0.0,
            seed=0,
        )


# ---------------------------------------------------------------------------
# build_utterance_client_from_env
# ---------------------------------------------------------------------------


def _clean_protocol_env(monkeypatch):
    for k in (
        "PROTOCOL_LLM_API_KEY",
        "PROTOCOL_LLM_PROVIDER",
        "PROTOCOL_LLM_BASE_URL",
        "PROTOCOL_LLM_MODEL",
        "PROTOCOL_LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(k, raising=False)


def test_build_from_env_returns_none_without_api_key(monkeypatch):
    _clean_protocol_env(monkeypatch)
    assert build_utterance_client_from_env() is None


def test_build_from_env_uses_qwen_preset_defaults(monkeypatch):
    _clean_protocol_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("PROTOCOL_LLM_PROVIDER", "qwen")
    client = build_utterance_client_from_env()
    assert isinstance(client, OpenAiUtteranceClient)
    assert client.base_url.endswith("/compatible-mode/v1")
    assert client.model.startswith("qwen-")


def test_build_from_env_raises_on_unknown_provider_without_base_url(monkeypatch):
    _clean_protocol_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("PROTOCOL_LLM_PROVIDER", "no-such-provider")
    with pytest.raises(ValueError):
        build_utterance_client_from_env()


def test_build_from_env_honors_explicit_overrides(monkeypatch):
    _clean_protocol_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("PROTOCOL_LLM_PROVIDER", "no-such-provider")
    monkeypatch.setenv(
        "PROTOCOL_LLM_BASE_URL", "https://custom.example.com/v1"
    )
    monkeypatch.setenv("PROTOCOL_LLM_MODEL", "custom-model")
    client = build_utterance_client_from_env()
    assert isinstance(client, OpenAiUtteranceClient)
    assert client.base_url == "https://custom.example.com/v1"
    assert client.model == "custom-model"
