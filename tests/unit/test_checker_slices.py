# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for checker.slices."""

from __future__ import annotations

from music21 import stream

from harmoniatextor.checker.slices import build_slices, pitch_classes, tonic_pc
from harmoniatextor.score.streamops import ScoreEditor


class TestSlices:
    """Vertical slicing helpers."""

    def test_build_slices(self, score4: stream.Score) -> None:
        """Notes at the same offset form one slice."""
        editor = ScoreEditor(score4)
        editor.place_note("soprano", 1, 0.0, "C5", 1.0)
        editor.place_note("alto", 1, 0.0, "E4", 1.0)
        slices = build_slices(score4)
        assert slices
        assert slices[0].measure == 1
        assert pitch_classes(slices[0].events) == {0, 4}

    def test_tonic_pc(self) -> None:
        """The tonic pitch class is resolved."""
        assert tonic_pc("C") == 0
