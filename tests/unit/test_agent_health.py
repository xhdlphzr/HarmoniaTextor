# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for agent.health."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from harmoniatextor.agent import health


class _Model:
    """A chat model stub for connectivity tests."""

    def __init__(self, *, fail: bool) -> None:
        """Store whether the probe should fail.

        Args:
            fail: The fail.
        """
        self._fail = fail

    def invoke(self, _messages: Any) -> str:
        """Return a pong, or raise when configured to fail.

        Args:
            _messages: The messages.

        Returns:
            The resulting text.

        Raises:
            RuntimeError: When the operation cannot proceed.
        """
        if self._fail:
            raise RuntimeError("unreachable")
        return "pong"


def _model(*, fail: bool) -> BaseChatModel:
    """Wrap the stub in the expected chat-model type.

    Args:
        fail: The fail.

    Returns:
        The model result.
    """
    return cast("BaseChatModel", _Model(fail=fail))


class TestCheckConnection:
    """Endpoint connectivity probing."""

    def test_unknown_without_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No configured or environment key yields the unknown state.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert health.check_connection({"api_key": ""}) == health.UNKNOWN

    def test_environment_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An environment key is enough to probe the endpoint.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        assert health.check_connection({"api_key": ""}, _model(fail=False)) == health.OK

    def test_ok(self) -> None:
        """A reachable endpoint reports ok."""
        assert (
            health.check_connection({"api_key": "k"}, _model(fail=False)) == health.OK
        )

    def test_error(self) -> None:
        """A failing request reports error."""
        assert (
            health.check_connection({"api_key": "k"}, _model(fail=True)) == health.ERROR
        )

    def test_defaults_to_current(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a mapping the effective configuration is used.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(health, "current_config", lambda: {"api_key": "k"})
        assert health.check_connection(model=_model(fail=False)) == health.OK

    def test_model_construction_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failure to build the client reports error.

        Args:
            monkeypatch: The pytest monkeypatch fixture.

        Raises:
            RuntimeError: When the operation cannot proceed.
        """

        def boom() -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(health, "create_chat_model", boom)
        assert health.check_connection({"api_key": "k"}) == health.ERROR
