# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Domain layer: entities, value objects and validated parameter models."""

from __future__ import annotations

from harmoniatextor.domain.enums import (
    VOICE_SLOTS,
    Severity,
    ToolKind,
    VoiceSlot,
    WorkStatus,
)
from harmoniatextor.domain.interval import DiatonicInterval, parse_interval
from harmoniatextor.domain.key import KeySpec, parse_key
from harmoniatextor.domain.models import (
    CheckReport,
    CheckViolation,
    Movement,
    Revision,
    Theme,
    ThemeNote,
    Work,
    theme_fingerprint,
)

__all__ = [
    "VOICE_SLOTS",
    "CheckReport",
    "CheckViolation",
    "DiatonicInterval",
    "KeySpec",
    "Movement",
    "Revision",
    "Severity",
    "Theme",
    "ThemeNote",
    "ToolKind",
    "VoiceSlot",
    "Work",
    "WorkStatus",
    "parse_interval",
    "parse_key",
    "theme_fingerprint",
]
