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

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from harmoniatextor.config import load_config

__all__ = ["DEFAULT_MODEL", "create_chat_model", "resolve_setting"]

DEFAULT_MODEL = "gpt-4o-mini"


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
    return ChatOpenAI(
        model=resolved_model,
        base_url=resolved_base,
        api_key=resolved_key,
    )
