# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""LangChain-based composer agent."""

from __future__ import annotations

from harmoniatextor.agent.compression import (
    COMPRESS_RATIO,
    DEFAULT_CONTEXT_WINDOW,
    compress_messages,
    token_count,
)
from harmoniatextor.agent.llm_factory import DEFAULT_MODEL, create_chat_model
from harmoniatextor.agent.loop import AgentLoop, AgentRunResult
from harmoniatextor.agent.prompts import system_prompt
from harmoniatextor.agent.reviewer import ReviewerAI, ReviewResult
from harmoniatextor.agent.tools import build_tools, result_payload

__all__ = [
    "COMPRESS_RATIO",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_MODEL",
    "AgentLoop",
    "AgentRunResult",
    "ReviewResult",
    "ReviewerAI",
    "build_tools",
    "compress_messages",
    "create_chat_model",
    "result_payload",
    "system_prompt",
    "token_count",
]
