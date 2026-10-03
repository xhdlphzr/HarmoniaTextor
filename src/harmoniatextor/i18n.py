# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Interface localisation loaded from the top-level ``i18n`` YAML catalogues.

Every user-facing string is addressed by a language-neutral message id such as
``nav.create``.  The catalogue files (``i18n/en.yaml`` and ``i18n/zh.yaml``)
live at the repository root and are bundled next to the frozen executable, so
this module only resolves and caches them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "CATALOG",
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "catalog_for",
    "i18n_dir",
    "normalize_language",
    "translate",
]

LANGUAGES: tuple[str, ...] = ("en", "zh")
DEFAULT_LANGUAGE = "en"


def i18n_dir() -> Path:
    """Return the directory holding the message catalogues.

    Returns:
        The bundled ``i18n`` directory of a frozen build, or the repository
        ``i18n`` directory during development.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base is not None:
        return Path(base) / "i18n"
    return Path(__file__).resolve().parents[2] / "i18n"


def _load(language: str) -> dict[str, str]:
    """Load one language catalogue from its YAML file.

    Args:
        language: A supported language code.

    Returns:
        A mapping of message id to localised text.
    """
    raw: Any = (
        yaml.safe_load((i18n_dir() / f"{language}.yaml").read_text(encoding="utf-8"))
        or {}
    )
    return {str(key): str(value) for key, value in raw.items()}


#: Language code -> (message id -> localised text).
CATALOG: dict[str, dict[str, str]] = {
    language: _load(language) for language in LANGUAGES
}


def normalize_language(value: Any) -> str:
    """Clamp a value to a supported language code.

    Args:
        value: Raw language value.

    Returns:
        A supported language code, falling back to :data:`DEFAULT_LANGUAGE`.
    """
    text = str(value or "").strip().lower()
    return text if text in LANGUAGES else DEFAULT_LANGUAGE


def catalog_for(language: Any) -> dict[str, str]:
    """Return the catalogue for a language.

    Args:
        language: Raw language value.

    Returns:
        The message-id to text mapping for the language.
    """
    return CATALOG[normalize_language(language)]


def translate(message_id: str, language: Any, **kwargs: Any) -> str:
    """Translate a message id and substitute placeholders.

    Args:
        message_id: Language-neutral message id.
        language: Target language.
        **kwargs: Values for ``{name}`` placeholders.

    Returns:
        The localised text, falling back to English and then to the id itself.
    """
    table = catalog_for(language)
    result = table.get(message_id)
    if result is None:
        result = CATALOG[DEFAULT_LANGUAGE].get(message_id, message_id)
    return result.format(**kwargs) if kwargs else result
