# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Connectivity probe for the configured OpenAI-compatible endpoint."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from harmoniatextor.agent.llm_factory import create_chat_model
from harmoniatextor.config import current_config

__all__ = ["ERROR", "OK", "UNKNOWN", "check_connection"]

#: The endpoint answered a trivial request.
OK = "ok"
#: The endpoint could not be reached or rejected the request.
ERROR = "error"
#: No credentials are configured, so nothing was probed.
UNKNOWN = "unknown"


def _has_credentials(config: dict[str, Any]) -> bool:
    """Return whether any API key is configured.

    Args:
        config: Effective configuration mapping.

    Returns:
        ``True`` when the configuration or environment carries a key.
    """
    if config.get("api_key"):
        return True
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def check_connection(
    config: dict[str, Any] | None = None,
    model: BaseChatModel | None = None,
) -> str:
    """Probe the endpoint with a trivial request.

    Args:
        config: Effective configuration; defaults to :func:`current_config`.
        model: A ready chat model, mainly for tests.

    Returns:
        :data:`OK` when reachable, :data:`ERROR` when the probe fails, or
        :data:`UNKNOWN` when no credentials are configured.
    """
    values = config if config is not None else current_config()
    if not _has_credentials(values):
        return UNKNOWN
    try:
        client = model if model is not None else create_chat_model()
        client.invoke([HumanMessage(content="ping")])
    except Exception:  # noqa: BLE001 - any probe failure means "unreachable"
        return ERROR
    return OK
