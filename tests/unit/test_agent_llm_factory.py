# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for agent.llm_factory."""

from __future__ import annotations

from typing import Any, cast

import pytest

from harmoniatextor.agent.llm_factory import (
    create_chat_model,
    is_deepseek_endpoint,
    resolve_setting,
)


class TestLLMFactory:
    """Chat model factory."""

    @pytest.fixture(autouse=True)
    def _no_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Isolate the factory from the user's real config file.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr("harmoniatextor.agent.llm_factory.load_config", dict)

    def test_explicit_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit arguments are forwarded.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        created = cast(
            "object", create_chat_model(model="m", base_url="u", api_key="k")
        )
        assert created == "model"
        assert captured["model"] == "m"

    def test_environment_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables are used as defaults.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_BASE_URL", "env-url")
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured.update(
                {key: kwargs[key] for key in ("model", "base_url", "api_key")}
            )
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        create_chat_model()
        assert captured == {
            "model": "env-model",
            "base_url": "env-url",
            "api_key": "env-key",
        }

    def test_openai_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OPENAI_API_KEY is used when LLM_API_KEY is absent.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured["api_key"] = kwargs["api_key"]
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        create_chat_model()
        assert captured["api_key"] == "openai-key"

    def test_config_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The config file overrides environment and defaults.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(
            "harmoniatextor.agent.llm_factory.load_config",
            lambda: {
                "model": "cfg-model",
                "base_url": "cfg-url",
                "api_key": "cfg-key",
            },
        )
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        create_chat_model()
        assert captured["model"] == "cfg-model"
        assert captured["base_url"] == "cfg-url"
        assert captured["api_key"] == "cfg-key"
        assert "temperature" not in captured

    def test_deepseek_disables_thinking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DeepSeek endpoints run with thinking disabled.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        create_chat_model(
            model="deepseek-flash",
            base_url="https://api.deepseek.com",
            api_key="k",
        )
        assert captured["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_other_endpoints_keep_default_thinking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-DeepSeek endpoints receive no thinking parameter.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        create_chat_model(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="k",
        )
        assert captured["extra_body"] is None


class TestDeepSeekEndpoint:
    """DeepSeek endpoint detection."""

    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("https://api.deepseek.com", True),
            ("https://api.deepseek.com/v1", True),
            ("https://deepseek.com", True),
            ("https://api.openai.com/v1", False),
            ("https://notdeepseek.com", False),
            ("https://api.deepseek.com.evil.test", False),
            (None, False),
            ("", False),
        ],
    )
    def test_detects_host(self, base_url: str | None, expected: bool) -> None:
        """Only real DeepSeek hosts are detected.

        Args:
            base_url: The endpoint URL.
            expected: Whether it is a DeepSeek endpoint.
        """
        assert is_deepseek_endpoint(base_url) is expected


class TestResolveSetting:
    """Setting precedence resolution."""

    def test_explicit_wins(self) -> None:
        """An explicit value wins over everything."""
        assert resolve_setting("x", {"key": "y"}, "key", "ENV", "d") == "x"

    def test_config_wins(self) -> None:
        """The config file wins over environment and defaults."""
        assert resolve_setting(None, {"key": "y"}, "key", "ENV", "d") == "y"

    def test_environment_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The environment wins over the default.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setenv("ENV", "e")
        assert resolve_setting(None, {}, "key", "ENV", "d") == "e"

    def test_default(self) -> None:
        """The default is used when nothing else is set."""
        assert resolve_setting(None, {}, "key", "ENV", "d") == "d"

    def test_without_env_name(self) -> None:
        """A missing environment name falls back to the default."""
        assert resolve_setting(None, {}, "key", None, "d") == "d"
