# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Locate bundled static assets such as the web favicon."""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["assets_dir", "favicon_icon"]


def assets_dir() -> Path:
    """Return the directory holding bundled assets.

    The directory is ``assets/`` next to the repository root during
    development, or ``<bundle>/assets`` when frozen by PyInstaller.

    Returns:
        The assets directory.
    """
    frozen = getattr(sys, "_MEIPASS", None)
    base = Path(frozen) if isinstance(frozen, str) else Path(__file__).resolve().parent.parent
    return base / "assets"


def favicon_icon() -> Path | None:
    """Return the icon served as the web favicon.

    Returns:
        ``Franx.png`` when present, otherwise ``None``.
    """
    candidate = assets_dir() / "Franx.png"
    return candidate if candidate.exists() else None
