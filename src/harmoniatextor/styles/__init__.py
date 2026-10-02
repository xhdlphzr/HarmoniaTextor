# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Style kits: named selections of symbolic rules and composition techniques."""

from __future__ import annotations

from harmoniatextor.styles.base import StyleKit
from harmoniatextor.styles.builtin import (
    ALL_RULE_IDS,
    ALL_TECHNIQUE_IDS,
    BUILTIN_KITS,
    DEFAULT_STYLE_ID,
    FULL_STYLE_ID,
)
from harmoniatextor.styles.registry import StyleRegistry
from harmoniatextor.styles.store import StyleKitStore

__all__ = [
    "ALL_RULE_IDS",
    "ALL_TECHNIQUE_IDS",
    "BUILTIN_KITS",
    "DEFAULT_STYLE_ID",
    "FULL_STYLE_ID",
    "StyleKit",
    "StyleKitStore",
    "StyleRegistry",
]
