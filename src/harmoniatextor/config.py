# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""User configuration stored in ``~/.harmonia_textor/config.json``.

The file configures the OpenAI-compatible endpoint shared by the composer
("writer") agent and the independent reviewer ("checker") agent.  It is
optional: when it is missing, environment variables and built-in defaults are
used instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_KEYS",
    "CONFIG_DIR_NAME",
    "config_dir",
    "config_path",
    "context_window_tokens",
    "current_config",
    "default_config",
    "ensure_config",
    "load_config",
    "save_config",
]

CONFIG_DIR_NAME = ".harmonia_textor"
_CONFIG_FILE_NAME = "config.json"

ALLOWED_KEYS = ("base_url", "api_key", "model", "context_window")

_DEFAULT_ENDPOINT: dict[str, Any] = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "",
    "model": "gpt-4o-mini",
    "context_window": 200,
}

_TOKENS_PER_K = 1000


def context_window_tokens(config_values: dict[str, Any] | None = None) -> int:
    """Return the configured context window in tokens.

    The stored value is expressed in thousands of tokens (k token).

    Args:
        config_values: Configuration mapping; defaults to the effective config.

    Returns:
        The context window in tokens, at least one thousand.
    """
    values = config_values if config_values is not None else current_config()
    raw = values.get("context_window", _DEFAULT_ENDPOINT["context_window"])
    try:
        k_tokens = int(float(raw))
    except TypeError, ValueError:
        k_tokens = int(_DEFAULT_ENDPOINT["context_window"])
    return max(1, k_tokens) * _TOKENS_PER_K


def config_dir() -> Path:
    """Return the user configuration directory.

    Returns:
        ``~/.harmonia_textor``.
    """
    return Path.home() / CONFIG_DIR_NAME


def config_path() -> Path:
    """Return the configuration file path.

    Returns:
        ``~/.harmonia_textor/config.json``.
    """
    return config_dir() / _CONFIG_FILE_NAME


def default_config() -> dict[str, Any]:
    """Return the default configuration values.

    Returns:
        A dictionary with the default API settings.
    """
    return dict(_DEFAULT_ENDPOINT)


def load_config() -> dict[str, Any]:
    """Load the configuration file.

    Returns:
        The parsed configuration, or an empty mapping when the file is missing
        or malformed.
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError, OSError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if str(key) in ALLOWED_KEYS}


def current_config() -> dict[str, Any]:
    """Return the effective configuration.

    Returns:
        The defaults merged with the stored configuration.
    """
    return {**default_config(), **load_config()}


def save_config(values: dict[str, Any]) -> Path:
    """Persist configuration values over the existing configuration.

    Args:
        values: Partial configuration to store.

    Returns:
        The configuration file path.
    """
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    allowed = {key: value for key, value in values.items() if key in ALLOWED_KEYS}
    merged = {**current_config(), **allowed}
    path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def ensure_config() -> Path:
    """Create the configuration file with defaults when missing.

    Returns:
        The configuration file path.
    """
    path = config_path()
    if not path.exists():
        return save_config({})
    return path
