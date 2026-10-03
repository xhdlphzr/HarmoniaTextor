# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for agent.llm_factory."""

from __future__ import annotations

from typing import Any, cast

import pytest

from harmoniatextor.agent.llm_factory import create_chat_model, resolve_setting


class TestLLMFactory:
    """Chat model factory."""

    @pytest.fixture(autouse=True)
    def _no_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Isolate the factory from the user's real config file."""
        monkeypatch.setattr("harmoniatextor.agent.llm_factory.load_config", dict)

    def test_explicit_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit arguments are forwarded."""
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
        """Environment variables are used as defaults."""
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
        """OPENAI_API_KEY is used when LLM_API_KEY is absent."""
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
        """The config file overrides environment and defaults."""
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


class TestResolveSetting:
    """Setting precedence resolution."""

    def test_explicit_wins(self) -> None:
        """An explicit value wins over everything."""
        assert resolve_setting("x", {"key": "y"}, "key", "ENV", "d") == "x"

    def test_config_wins(self) -> None:
        """The config file wins over environment and defaults."""
        assert resolve_setting(None, {"key": "y"}, "key", "ENV", "d") == "y"

    def test_environment_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The environment wins over the default."""
        monkeypatch.setenv("ENV", "e")
        assert resolve_setting(None, {}, "key", "ENV", "d") == "e"

    def test_default(self) -> None:
        """The default is used when nothing else is set."""
        assert resolve_setting(None, {}, "key", "ENV", "d") == "d"

    def test_without_env_name(self) -> None:
        """A missing environment name falls back to the default."""
        assert resolve_setting(None, {}, "key", None, "d") == "d"
