# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for agent.compression."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import invalid_tool_call

from harmoniatextor.agent.compression import (
    _bounded_transcript,
    compress_messages,
    content_text,
    ensure_tool_responses,
    message_text,
    token_count,
)

_COUNTED_TOKENS = 7


_HEURISTIC_MIN = 10


_COMPRESSED_MESSAGES = 1


_REPAIRED_MESSAGES = 4


_COMPLETE_MESSAGES = 3


_LONG_TEXT_CHARS = 7000


_TOKENIZER_PER_TEXT = 3


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


class _SummarizerModel:
    """A model double that counts tokens and returns a summary."""

    def __init__(self, *, tokens: int = 0, content: str = "摘要") -> None:
        """Initialise the double.

        Args:
            tokens: Token count reported to the caller.
            content: Summary returned by ``invoke``.
        """
        self.tokens = tokens
        self.content = content
        self.invocations = 0

    def get_num_tokens_from_messages(self, _messages: object) -> int:
        """Return the configured token count."""
        return self.tokens

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return the configured summary."""
        self.invocations += 1
        return AIMessage(content=self.content)


class _NoCountModel:
    """A model double without a token counter."""

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return a summary."""
        return AIMessage(content="摘要")


class _BrokenCountModel:
    """A model double whose token counter fails."""

    def get_num_tokens_from_messages(self, _messages: object) -> int:
        """Raise to exercise the fallback path."""
        raise RuntimeError("no counter")

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return a summary."""
        return AIMessage(content="摘要")


class _TokenizerModel:
    """A model double exposing only a per-text tokenizer."""

    def get_num_tokens(self, _text: str) -> int:
        """Return a fixed per-text token count."""
        return _TOKENIZER_PER_TEXT

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return a summary."""
        return AIMessage(content="摘要")


class _BrokenTokenizerModel:
    """A model double whose every tokenizer fails."""

    def get_num_tokens_from_messages(self, _messages: object) -> int:
        """Raise to skip the full-message counter."""
        raise RuntimeError("no counter")

    def get_num_tokens(self, _text: str) -> int:
        """Raise to force the heuristic."""
        raise RuntimeError("no tokenizer")

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return a summary."""
        return AIMessage(content="摘要")


class TestCompression:
    """Conversation compression."""

    def test_content_text_list(self) -> None:
        """List content is flattened to text."""
        assert content_text([{"text": "hi"}, "yo"]) == "hiyo"

    def test_bounded_transcript(self) -> None:
        """A long history is truncated and bounded for the summariser."""
        messages: list[BaseMessage] = [
            HumanMessage(content="x" * _LONG_TEXT_CHARS) for _ in range(5)
        ]
        text = _bounded_transcript(messages, budget_chars=50)
        assert "截断" in text
        assert len(text) < _LONG_TEXT_CHARS

    def test_message_text_content(self) -> None:
        """Visible content is preferred over reasoning."""
        message = AIMessage(content="可见文字", additional_kwargs={"reasoning_content": "思考"})
        assert message_text(message) == "可见文字"

    def test_message_text_reasoning_fallback(self) -> None:
        """Reasoning content is surfaced when the content is empty."""
        message = AIMessage(content="", additional_kwargs={"reasoning_content": "思考"})
        assert message_text(message) == "思考"

    def test_message_text_reasoning_alias(self) -> None:
        """The ``reasoning`` alias is also accepted."""
        message = AIMessage(content="", additional_kwargs={"reasoning": "推理"})
        assert message_text(message) == "推理"

    def test_message_text_empty(self) -> None:
        """A message without any text yields an empty string."""
        assert message_text(AIMessage(content="")) == ""

    def test_token_count_counter(self) -> None:
        """A model counter is preferred."""
        model = _SummarizerModel(tokens=_COUNTED_TOKENS)
        counted = token_count(cast("BaseChatModel", model), [HumanMessage(content="x")])
        assert counted == _COUNTED_TOKENS

    def test_token_count_fallback(self) -> None:
        """The character heuristic is used without a counter."""
        model = _NoCountModel()
        heuristic = token_count(cast("BaseChatModel", model), [HumanMessage(content="a" * 40)])
        assert heuristic >= _HEURISTIC_MIN

    def test_token_count_broken(self) -> None:
        """A failing counter falls back to the heuristic."""
        model = _BrokenCountModel()
        heuristic = token_count(cast("BaseChatModel", model), [HumanMessage(content="a" * 40)])
        assert heuristic >= _HEURISTIC_MIN

    def test_token_count_tokenizer(self) -> None:
        """A per-text tokenizer is used when the full-message counter is absent."""
        model = _TokenizerModel()
        counted = token_count(cast("BaseChatModel", model), [HumanMessage(content="你好")])
        assert counted == _TOKENIZER_PER_TEXT + 1

    def test_token_count_broken_tokenizer(self) -> None:
        """A failing tokenizer falls back to the heuristic."""
        model = _BrokenTokenizerModel()
        heuristic = token_count(cast("BaseChatModel", model), [HumanMessage(content="a" * 40)])
        assert heuristic >= _HEURISTIC_MIN

    def test_token_count_tokenizer_with_calls(self) -> None:
        """Tool-call arguments are counted by the tokenizer too."""
        model = _TokenizerModel()
        counted = token_count(cast("BaseChatModel", model), [tool_call("read", {})])
        assert counted == _TOKENIZER_PER_TEXT * 2 + 1

    def test_below_threshold(self) -> None:
        """Short conversations are left untouched."""
        model = _SummarizerModel(tokens=1)
        messages = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert not compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
        )
        assert model.invocations == 0

    def test_compress_with_system(self) -> None:
        """A long conversation is folded into the system message."""
        model = _SummarizerModel(tokens=10_000)
        messages: list[BaseMessage] = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
        )
        assert model.invocations == 1
        assert len(messages) == _COMPRESSED_MESSAGES
        text = content_text(messages[0].content)
        assert "sys" in text
        assert "摘要" in text
        assert "<xml/>" in text

    def test_compress_pins_plan(self) -> None:
        """Pinned text is kept verbatim in the system message."""
        model = _SummarizerModel(tokens=10_000)
        messages: list[BaseMessage] = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
            pinned_provider=lambda: "Step 1 规划内容",
        )
        assert "Step 1 规划内容" in content_text(messages[0].content)

    def test_compress_without_system(self) -> None:
        """A conversation without a system prompt keeps the summary and artifact."""
        model = _SummarizerModel(tokens=10_000)
        messages: list[BaseMessage] = [HumanMessage(content="hi")]
        assert compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
        )
        assert len(messages) == _COMPRESSED_MESSAGES
        assert "<xml/>" in content_text(messages[0].content)

    def test_compress_renders_tool_calls(self) -> None:
        """Tool calls are included in the summary transcript."""
        model = _SummarizerModel(tokens=10_000)
        messages: list[BaseMessage] = [
            SystemMessage(content="sys"),
            tool_call("technique_imitation", {"theme_id": 1}),
        ]
        assert compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
        )

    def test_ensure_tool_responses(self) -> None:
        """A dangling tool call gets a placeholder tool response."""
        messages: list[BaseMessage] = [
            SystemMessage(content="sys"),
            tool_call("technique_imitation", {"theme_id": 1}, call_id="c1"),
            HumanMessage(content="continue"),
        ]
        ensure_tool_responses(messages)
        assert len(messages) == _REPAIRED_MESSAGES
        assert isinstance(messages[2], ToolMessage)
        assert messages[2].tool_call_id == "c1"

    def test_ensure_tool_responses_complete(self) -> None:
        """A complete history is left untouched."""
        messages: list[BaseMessage] = [
            SystemMessage(content="sys"),
            tool_call("technique_imitation", {"theme_id": 1}, call_id="c1"),
            ToolMessage(content="ok", tool_call_id="c1"),
        ]
        ensure_tool_responses(messages)
        assert len(messages) == _COMPLETE_MESSAGES

    def test_ensure_tool_responses_invalid(self) -> None:
        """An unparseable tool call also gets a placeholder response."""
        messages: list[BaseMessage] = [
            SystemMessage(content="sys"),
            invalid_call("technique_imitation", call_id="x1"),
            HumanMessage(content="continue"),
        ]
        ensure_tool_responses(messages)
        assert len(messages) == _REPAIRED_MESSAGES
        assert isinstance(messages[2], ToolMessage)
        assert messages[2].tool_call_id == "x1"
