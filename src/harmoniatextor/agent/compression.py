# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Conversation compression for long-running agent sessions.

When a session grows past 90% of the model context window it is summarised by
the model itself into a structured digest and restarted, keeping the current
musical artifact so work can continue without losing state.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

__all__ = [
    "COMPRESS_RATIO",
    "DEFAULT_CONTEXT_WINDOW",
    "compress_messages",
    "content_text",
    "ensure_tool_responses",
    "message_text",
    "token_count",
]

COMPRESS_RATIO = 0.9
DEFAULT_CONTEXT_WINDOW = 200_000
_CHARS_PER_TOKEN = 4
_MAX_MESSAGE_CHARS = 6000
_TRANSCRIPT_RATIO = 0.5
_ASCII_MAX = 127

_SUMMARY_SYSTEM = "你是一位对话压缩助手。"
_SUMMARY_PROMPT = (
    "请把下面的对话压缩为结构化摘要,保留继续工作所需的一切关键信息。\n"
    "按以下格式输出:\n"
    "【目标】当前任务的核心目标\n"
    "【重要细节】已确定的关键决策、调式、声部、主题编号等\n"
    "【工作状态】已完成 / 进行中 / 被阻塞\n"
    "【下一步行动】建议的下一步具体操作"
)


def content_text(content: Any) -> str:
    """Normalise a chat message content to plain text.

    Args:
        content: Message content, possibly a list of content blocks.

    Returns:
        The text content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def message_text(message: Any) -> str:
    """Return a chat message's visible text, falling back to reasoning.

    Some OpenAI-compatible providers (for example DeepSeek) put the model's
    chain-of-thought in ``additional_kwargs['reasoning_content']`` while leaving
    ``content`` empty on tool-calling turns.  Surfacing it keeps the neural-layer
    feedback visible in the progress panel.

    Args:
        message: A chat message.

    Returns:
        The message text, or an empty string when there is none.
    """
    text = content_text(getattr(message, "content", ""))
    if text:
        return text
    extra = getattr(message, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning"):
        value = extra.get(key)
        if value:
            return str(value)
    return ""


def _estimate_tokens(text: str) -> int:
    """Estimate the token count of a text.

    ASCII text averages roughly four characters per token, while non-ASCII text
    (Chinese, MusicXML text nodes, ...) is conservatively counted one token per
    character so the estimate errs high rather than low.

    Args:
        text: The text to estimate.

    Returns:
        An estimated token count, at least one.
    """
    non_ascii = sum(1 for char in text if ord(char) > _ASCII_MAX)
    ascii_count = len(text) - non_ascii
    return non_ascii + ascii_count // _CHARS_PER_TOKEN + 1


def _count_with_tokenizer(
    count_tokens: Callable[[str], int], messages: list[BaseMessage]
) -> int:
    """Count a conversation with the model's own tokenizer.

    Every OpenAI-compatible chat model exposes ``get_num_tokens(text)``, which
    uses tiktoken (the real BPE tokenizer) even for endpoints whose
    ``get_num_tokens_from_messages`` is not implemented, so this is accurate
    rather than estimated.

    Args:
        count_tokens: The model's ``get_num_tokens`` callable.
        messages: Conversation messages.

    Returns:
        The token count, including a small per-message overhead.
    """
    total = 0
    for message in messages:
        total += int(count_tokens(content_text(message.content)))
        calls = getattr(message, "tool_calls", None)
        if calls:
            total += int(count_tokens(str(calls)))
    return total + len(messages)


def _count_with_heuristic(messages: list[BaseMessage]) -> int:
    """Count a conversation with a character heuristic.

    Only used when the model exposes no tokenizer at all.

    Args:
        messages: Conversation messages.

    Returns:
        The estimated token count.
    """
    total = 0
    for message in messages:
        total += _estimate_tokens(content_text(message.content))
        calls = getattr(message, "tool_calls", None)
        if calls:
            total += _estimate_tokens(str(calls))
    return total


def token_count(chat_model: BaseChatModel, messages: list[BaseMessage]) -> int:
    """Count the tokens of a conversation as accurately as the model allows.

    The model's own tokenizer is always preferred: first
    ``get_num_tokens_from_messages`` (exact for OpenAI models), then
    ``get_num_tokens`` per message (tiktoken, the real tokenizer, for
    OpenAI-compatible endpoints such as DeepSeek).  A character heuristic is
    only the last resort when the model exposes no tokenizer.

    Args:
        chat_model: Model used to count tokens.
        messages: Conversation messages.

    Returns:
        The token count.
    """
    counter = getattr(chat_model, "get_num_tokens_from_messages", None)
    if callable(counter):
        # A model tokenizer is best-effort: fall through on any failure.
        with contextlib.suppress(Exception):
            return int(counter(messages))
    count_tokens = getattr(chat_model, "get_num_tokens", None)
    if callable(count_tokens):
        with contextlib.suppress(Exception):
            return _count_with_tokenizer(count_tokens, messages)
    return _count_with_heuristic(messages)


def _render(message: BaseMessage) -> str:
    """Render a message as plain text for the summariser.

    Args:
        message: The message to render.

    Returns:
        A role-tagged text line.
    """
    text = content_text(message.content)
    calls = getattr(message, "tool_calls", None)
    if calls:
        text = f"{text} {calls}"
    return f"[{message.type}] {text}"


def _bounded_transcript(messages: list[BaseMessage], budget_chars: int) -> str:
    """Render a transcript that fits a character budget, newest first.

    Long histories (for example several full scores echoed into tool results)
    would make the summariser call itself exceed the context window.  Each
    message is truncated and only the most recent messages within the budget are
    kept, so the summariser always receives a request it can process.

    Args:
        messages: Conversation messages.
        budget_chars: Maximum transcript length in characters.

    Returns:
        The bounded transcript, oldest first.
    """
    lines: list[str] = []
    total = 0
    for message in reversed(messages):
        rendered = _render(message)
        if len(rendered) > _MAX_MESSAGE_CHARS:
            rendered = rendered[:_MAX_MESSAGE_CHARS] + " …(过长已截断)"
        if lines and total + len(rendered) > budget_chars:
            break
        lines.append(rendered)
        total += len(rendered)
    return "\n\n".join(reversed(lines))


def ensure_tool_responses(messages: list[BaseMessage]) -> None:
    """Ensure every assistant tool call is followed by matching tool messages.

    Some OpenAI-compatible providers reject a request when an assistant message
    with tool calls is not immediately followed by one tool message per call
    id.  ``invalid_tool_calls`` (calls whose arguments were not parseable) are
    serialised as tool calls too, so they must be answered as well.  This
    repairs such a history in place.

    Args:
        messages: Conversation messages, modified in place.
    """
    index = 0
    while index < len(messages):
        calls = [
            *(getattr(messages[index], "tool_calls", None) or []),
            *(getattr(messages[index], "invalid_tool_calls", None) or []),
        ]
        if calls:
            scan = index + 1
            answered: set[str] = set()
            while scan < len(messages) and isinstance(messages[scan], ToolMessage):
                answered.add(str(getattr(messages[scan], "tool_call_id", "")))
                scan += 1
            for call in calls:
                call_id = str(call.get("id", ""))
                if call_id not in answered:
                    messages.insert(
                        scan,
                        ToolMessage(
                            content="[missing tool result]", tool_call_id=call_id
                        ),
                    )
                    scan += 1
                    answered.add(call_id)
        index += 1


def compress_messages(
    chat_model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    context_window: int,
    artifact_label: str,
    artifact_provider: Callable[[], str],
    pinned_provider: Callable[[], str] | None = None,
) -> bool:
    """Compress a conversation in place when it exceeds 90% of the window.

    Everything except the original system prompt is discarded; the system
    prompt is then extended with a model-written structured summary, the
    current artifact (for example the full MusicXML) and any pinned text (for
    example the Step 1 plan, kept verbatim), so no assistant tool calls can
    ever be left without matching tool responses.

    Args:
        chat_model: Model used to write the summary.
        messages: Conversation messages, modified in place.
        context_window: Model context window in tokens.
        artifact_label: Label used when re-attaching the artifact.
        artifact_provider: Callable returning the current artifact text.
        pinned_provider: Optional callable returning text to keep verbatim.

    Returns:
        ``True`` when a compression was performed.
    """
    threshold = int(context_window * COMPRESS_RATIO)
    if token_count(chat_model, messages) < threshold:
        return False
    budget = max(2000, int(context_window * _TRANSCRIPT_RATIO))
    transcript = _bounded_transcript(messages, budget)
    response = chat_model.invoke(
        [
            SystemMessage(content=_SUMMARY_SYSTEM),
            HumanMessage(content=f"{_SUMMARY_PROMPT}\n\n【对话】\n{transcript}"),
        ]
    )
    summary = content_text(response.content)
    base = (
        content_text(messages[0].content)
        if messages and isinstance(messages[0], SystemMessage)
        else ""
    )
    extended = (
        f"{base}\n\n"
        f"[上下文压缩] 以下是先前对话的压缩摘要,请据此继续:\n{summary}\n\n"
        f"{artifact_label}:\n{artifact_provider()}"
    )
    pinned = pinned_provider() if pinned_provider is not None else ""
    if pinned:
        extended += f"\n\n[创作规划(必须完整遵守)]\n{pinned}"
    messages[:] = [SystemMessage(content=extended)]
    return True
