"""Provider preset / env routing tests for the protocol uptake LLM."""

from __future__ import annotations

import pytest

from lifeform_service.openai_compat_client import (
    PROVIDER_PRESETS,
    build_client_from_env,
    describe_active_provider,
)


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "PROTOCOL_LLM_PROVIDER",
        "PROTOCOL_LLM_BASE_URL",
        "PROTOCOL_LLM_API_KEY",
        "PROTOCOL_LLM_MODEL",
        "PROTOCOL_LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_no_api_key_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    assert build_client_from_env() is None


def test_qwen_preset_resolves_dashscope_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_PROVIDER", "qwen")
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "test-key")
    client = build_client_from_env()
    assert client is not None
    assert "dashscope.aliyuncs.com" in client.base_url
    assert client.model == "qwen-plus"


def test_dashscope_alias_works(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_PROVIDER", "dashscope")
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "test-key")
    client = build_client_from_env()
    assert client is not None
    assert "dashscope.aliyuncs.com" in client.base_url


def test_explicit_base_url_overrides_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_PROVIDER", "qwen")
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "test-key")
    monkeypatch.setenv(
        "PROTOCOL_LLM_BASE_URL", "https://custom.example.com/v1"
    )
    client = build_client_from_env()
    assert client is not None
    assert client.base_url == "https://custom.example.com/v1"


def test_explicit_model_overrides_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_PROVIDER", "qwen")
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "test-key")
    monkeypatch.setenv("PROTOCOL_LLM_MODEL", "qwen-max")
    client = build_client_from_env()
    assert client is not None
    assert client.model == "qwen-max"


def test_unknown_provider_without_base_url_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_PROVIDER", "made-up-provider")
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "test-key")
    with pytest.raises(ValueError, match="not a known preset"):
        build_client_from_env()


def test_default_provider_is_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "test-key")
    client = build_client_from_env()
    assert client is not None
    assert "api.openai.com" in client.base_url


def test_describe_active_provider_does_not_leak_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("PROTOCOL_LLM_PROVIDER", "qwen")
    monkeypatch.setenv("PROTOCOL_LLM_API_KEY", "sk-secret-xxx")
    info = describe_active_provider()
    assert info["provider"] == "qwen"
    assert info["api_key_present"] == "yes"
    # The actual key value must NEVER appear in describe output.
    assert "sk-secret-xxx" not in str(info)


def test_known_presets_are_complete() -> None:
    """Lock the canonical preset set."""
    assert "openai" in PROVIDER_PRESETS
    assert "qwen" in PROVIDER_PRESETS
    assert "dashscope" in PROVIDER_PRESETS
    assert "vllm" in PROVIDER_PRESETS
    assert "lifeform-openai-compat" in PROVIDER_PRESETS
