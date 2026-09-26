# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Checker context and structural expectations provided by genres."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CheckerContext", "StructuralExpectation"]


@dataclass(frozen=True, slots=True)
class StructuralExpectation:
    """A structural position with an expected tonal/cadential outcome.

    Attributes:
        measure: One-based measure where the expectation is evaluated.
        key: Expected key specification at that position, if any.
        cadence: Whether an authentic cadence is required there.
        label: Human readable name of the structural position.
    """

    measure: int
    key: str | None = None
    cadence: bool = False
    label: str = ""


@dataclass(slots=True)
class CheckerContext:
    """Context passed to every check rule.

    Attributes:
        genre: Active genre identifier.
        tonic: Home key specification.
        expectations: Structural expectations used by tonal rules.
        enforce_voice_count: Whether the fixed voice count rule applies.
        spacing_limit: Maximum semitone gap between adjacent voices.
        outer_voices: Optional explicit outer voice pair.
    """

    genre: str = "plain"
    tonic: str = "C"
    expectations: list[StructuralExpectation] = field(default_factory=list)
    enforce_voice_count: bool = True
    spacing_limit: int = 16
    outer_voices: tuple[str, str] | None = None
