# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for techniques.helpers."""

from __future__ import annotations

import pytest
from music21 import stream

from harmoniatextor.techniques import TechniqueContext, TechniqueError
from harmoniatextor.techniques.helpers import (
    average_axis_midi,
    chord_pitches,
    functional_figure,
    get_theme,
    transpose_to_key,
)


class TestHelpers:
    """Shared helper behaviour."""

    def test_get_theme_missing(self, score4: stream.Score) -> None:
        """A missing theme raises."""
        with pytest.raises(TechniqueError, match="does not exist"):
            get_theme(TechniqueContext(score=score4, themes={}), 9)

    def test_average_axis_empty(self) -> None:
        """An empty theme has no axis."""
        with pytest.raises(TechniqueError, match="empty theme"):
            average_axis_midi([])

    def test_transpose_to_key_empty(self) -> None:
        """Transposing an empty theme raises."""
        with pytest.raises(TechniqueError, match="empty theme"):
            transpose_to_key([], "G")

    def test_chord_pitches_invalid(self) -> None:
        """An invalid figure raises."""
        with pytest.raises(TechniqueError, match="cannot realise"):
            chord_pitches("C", "not-a-chord")

    def test_functional_figure(self) -> None:
        """Functions map to roman numerals."""
        assert functional_figure("C", "T") == "I"
        assert functional_figure("a", "T") == "i"
        assert functional_figure("C", "D") == "V"
        with pytest.raises(TechniqueError, match="unknown function"):
            functional_figure("C", "X")
