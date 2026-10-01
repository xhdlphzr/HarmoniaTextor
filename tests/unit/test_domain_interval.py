# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for domain.interval."""

from __future__ import annotations

import pytest

from harmoniatextor.domain.interval import parse_interval

_FIFTH_SEMITONES = 7


_FIFTH_STEPS = 4


_FOURTH_SEMITONES = -5


_OCTAVE_SEMITONES = 12


class TestInterval:
    """Interval parsing."""

    def test_perfect_fifth(self) -> None:
        """A perfect fifth is seven semitones and four steps."""
        parsed = parse_interval(5)
        assert parsed.semitones == _FIFTH_SEMITONES
        assert parsed.steps == _FIFTH_STEPS
        assert parsed.name == "P5"

    def test_negative_fourth(self) -> None:
        """A descending fourth is negative."""
        parsed = parse_interval(-4)
        assert parsed.semitones == _FOURTH_SEMITONES
        assert parsed.name == "-P4"

    def test_major_third(self) -> None:
        """A third uses major quality."""
        assert parse_interval(3).name == "M3"

    def test_octave_semitones(self) -> None:
        """Values above seven are treated as semitones."""
        parsed = parse_interval(12)
        assert parsed.semitones == _OCTAVE_SEMITONES
        assert parsed.name == "P8"

    def test_zero_rejected(self) -> None:
        """A zero interval is invalid."""
        with pytest.raises(ValueError, match="non-zero"):
            parse_interval(0)
