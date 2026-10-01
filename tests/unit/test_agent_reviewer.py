# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for agent.reviewer."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.tool import invalid_tool_call

from harmoniatextor.agent.reviewer import (
    _SYSTEM,
    ReviewerAI,
    _extract,
    _submit_review,
)


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    """Build an assistant message requesting a tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def invalid_call(name: str, call_id: str = "x1", error: str = "bad json") -> AIMessage:
    """Build an assistant message with an unparseable tool call."""
    return AIMessage(
        content="",
        invalid_tool_calls=[invalid_tool_call(name=name, args="{bad", id=call_id, error=error)],
    )


class _ReviewModel:
    """A model double returning scripted review responses."""

    def __init__(self, responses: list[AIMessage]) -> None:
        """Initialise the script.

        Args:
            responses: Responses returned in order.
        """
        self.responses = responses
        self.cursor = 0
        self.bound: list[object] = []

    def bind_tools(self, tools: list[object], **_kwargs: object) -> _ReviewModel:
        """Remember the bound tools.

        Args:
            tools: Bound tools.
            _kwargs: Extra keyword arguments (ignored).

        Returns:
            This model.
        """
        self.bound = tools
        return self

    def invoke(self, messages: list[BaseMessage], **_kwargs: object) -> AIMessage:
        """Validate the request then return the next scripted response."""
        assert_no_dangling_tool_calls(messages)
        response = self.responses[min(self.cursor, len(self.responses) - 1)]
        self.cursor += 1
        return response


def assert_no_dangling_tool_calls(messages: list[BaseMessage]) -> None:
    """Assert every assistant tool call (valid or invalid) has a response.

    ``langchain_openai`` serialises both ``tool_calls`` and
    ``invalid_tool_calls`` into the request's ``tool_calls`` field, so a
    matching tool response is required for either kind.
    """
    index = 0
    while index < len(messages):
        valid = getattr(messages[index], "tool_calls", None) or []
        invalid = getattr(messages[index], "invalid_tool_calls", None) or []
        calls = [*valid, *invalid]
        if calls:
            needed = {str(call.get("id", "")) for call in calls}
            answered: set[str] = set()
            scan = index + 1
            while scan < len(messages) and isinstance(messages[scan], ToolMessage):
                answered.add(str(getattr(messages[scan], "tool_call_id", "")))
                scan += 1
            assert needed <= answered, f"dangling tool calls at {index}: {needed - answered}"
        index += 1


class TestReviewer:
    """The independent reviewer AI."""

    def test_pass(self) -> None:
        """A passing verdict is returned."""
        model = _ReviewModel([tool_call("submit_review", {"passed": True, "suggestions": ""})])
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert result.passed

    def test_reject(self) -> None:
        """A rejection carries its suggestions."""
        model = _ReviewModel([tool_call("submit_review", {"passed": False, "suggestions": "问题"})])
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert not result.passed
        assert result.suggestions == "问题"

    def test_no_verdict(self) -> None:
        """A model that never submits yields a rejection."""
        model = _ReviewModel([AIMessage(content="再看看")])
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert not result.passed
        assert result.suggestions

    def test_extract_ignores_other_tools(self) -> None:
        """Non-submit tool calls are ignored."""
        assert _extract(tool_call("other", {})) is None

    def test_other_tool_call(self) -> None:
        """A non-submit tool call is answered and retried."""
        model = _ReviewModel(
            [
                tool_call("other", {}, call_id="x1"),
                tool_call("submit_review", {"passed": True, "suggestions": ""}),
            ]
        )
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert result.passed

    def test_invalid_submit_call(self) -> None:
        """An unparseable submit call is answered and retried."""
        model = _ReviewModel(
            [
                invalid_call("submit_review", call_id="x1"),
                tool_call("submit_review", {"passed": True, "suggestions": ""}),
            ]
        )
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert result.passed

    def test_submit_review_tool(self) -> None:
        """The submit tool acknowledges the verdict."""
        assert _submit_review(True) == "已收到评审结论。"

    def test_system_checks_movement_division(self) -> None:
        """The reviewer is told to check movement division."""
        assert "乐章划分" in _SYSTEM

    def test_system_excludes_symbolic_rules(self) -> None:
        """The reviewer is told not to re-check symbolic-layer rules."""
        assert "平行五度" in _SYSTEM
        assert "以这些规则为由打回" in _SYSTEM
        assert "符号层" in _SYSTEM
