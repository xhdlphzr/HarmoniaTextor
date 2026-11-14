# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Configuration-driven factory for OpenAI-compatible chat models.

Settings are resolved in this order: explicit arguments, then
``~/.harmonia_textor/config.json``, then environment variables, then built-in
defaults.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from harmoniatextor.config import load_config

__all__ = [
    "DEFAULT_MODEL",
    "create_chat_model",
    "is_deepseek_endpoint",
    "resolve_setting",
]

DEFAULT_MODEL = "gpt-4o-mini"

_DEEPSEEK_HOST = "deepseek.com"

#: DeepSeek enables its thinking mode by default.  With tool calls in the
#: request DeepSeek then requires every previous turn's ``reasoning_content`` to
#: be sent back, and it rejects a forced ``tool_choice``; ``ChatOpenAI`` (the
#: OpenAI-compatible client used for every endpoint) does neither.  The composer
#: is a tool-driven loop that forces a tool call while the score is empty, so the
#: DeepSeek endpoint runs with thinking explicitly disabled.
_DISABLE_THINKING: dict[str, Any] = {"thinking": {"type": "disabled"}}


def is_deepseek_endpoint(base_url: str | None) -> bool:
    """Return whether a base URL points at the DeepSeek API.

    Args:
        base_url: The configured API base URL, or ``None``.

    Returns:
        ``True`` for ``deepseek.com`` and its subdomains.
    """
    host = (urlparse(base_url or "").hostname or "").lower()
    return host == _DEEPSEEK_HOST or host.endswith("." + _DEEPSEEK_HOST)


def resolve_setting(
    explicit: Any,
    config: dict[str, Any],
    key: str,
    env: str | None,
    default: Any,
) -> Any:
    """Resolve a setting from explicit value, config, environment or default.

    Args:
        explicit: An explicitly provided value, or ``None``.
        config: The loaded configuration mapping.
        key: The configuration key to read.
        env: The environment variable name to read, if any.
        default: The fallback default.

    Returns:
        The first non-empty value in the precedence order.
    """
    if explicit is not None:
        return explicit
    configured = config.get(key)
    if configured is not None and configured != "":
        return configured
    if env is not None:
        from_env = os.environ.get(env)
        if from_env:
            return from_env
    return default


def create_chat_model(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    """Create a chat model from configuration, environment or defaults.

    Configuration comes from ``~/.harmonia_textor/config.json``.  The same
    endpoint drives both the composer and the reviewer.  Environment variables
    are ``LLM_MODEL``, ``LLM_BASE_URL`` and ``LLM_API_KEY`` (falling back to
    ``OPENAI_API_KEY``).  Any OpenAI-compatible endpoint can be used, including
    local servers.

    DeepSeek's thinking mode is disabled explicitly: it rejects a forced
    ``tool_choice`` and needs ``reasoning_content`` replayed on every later
    request, which the OpenAI-compatible client does not support, while the
    composer relies on both.

    Args:
        model: Model name; overrides configuration and environment.
        base_url: API base URL; overrides configuration and environment.
        api_key: API key; overrides configuration and environment.

    Returns:
        A configured chat model.
    """
    config = load_config()
    resolved_model = resolve_setting(model, config, "model", "LLM_MODEL", DEFAULT_MODEL)
    resolved_base = resolve_setting(base_url, config, "base_url", "LLM_BASE_URL", None)
    resolved_key = resolve_setting(api_key, config, "api_key", "LLM_API_KEY", None)
    if not resolved_key:
        resolved_key = os.environ.get("OPENAI_API_KEY")
    extra_body = _DISABLE_THINKING if is_deepseek_endpoint(resolved_base) else None
    return ChatOpenAI(
        model=resolved_model,
        base_url=resolved_base,
        api_key=resolved_key,
        extra_body=extra_body,
    )
