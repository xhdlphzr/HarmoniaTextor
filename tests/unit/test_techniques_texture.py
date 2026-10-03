# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for techniques.texture."""

from __future__ import annotations

from typing import Any

import pytest
from music21 import stream

from harmoniatextor.techniques import (
    TechniqueContext,
    TechniqueError,
    build_default_registry,
)

REGISTRY = build_default_registry()


def apply(technique_id: str, score: stream.Score, **params: Any) -> Any:
    """Apply a technique by identifier to a score.

    Args:
        technique_id: The technique id.
        score: The score to inspect.
        params: Validated parameters.

    Returns:
        The result.
    """
    technique = REGISTRY.get(technique_id)
    context = TechniqueContext(score=score, themes={})
    return technique.apply(context, technique.params_model(**params))


class TestTexture:
    """Texture techniques."""

    def test_alberti_bass(self, score4: stream.Score) -> None:
        """An Alberti figure fills the voice.

        Args:
            score4: An empty four-voice score.
        """
        apply(
            "alberti_bass",
            score4,
            voice="piano",
            key="C",
            measure_range={"start": 1, "end": 2},
        )
        assert list(score4.recurse().notes)

    def test_broken_chord_up(self, score4: stream.Score) -> None:
        """An ascending arpeggio is written.

        Args:
            score4: An empty four-voice score.
        """
        apply(
            "broken_chord",
            score4,
            voice="piano",
            key="C",
            measure_range={"start": 1, "end": 1},
            direction="up",
        )

    def test_broken_chord_down(self, score4: stream.Score) -> None:
        """A descending arpeggio is written.

        Args:
            score4: An empty four-voice score.
        """
        apply(
            "broken_chord",
            score4,
            voice="piano",
            key="C",
            measure_range={"start": 1, "end": 1},
            direction="down",
        )

    def test_broken_chord_updown(self, score4: stream.Score) -> None:
        """An up-down arpeggio is written.

        Args:
            score4: An empty four-voice score.
        """
        apply(
            "broken_chord",
            score4,
            voice="piano",
            key="C",
            measure_range={"start": 1, "end": 1},
            direction="updown",
        )

    def test_broken_chord_bad_value(self, score4: stream.Score) -> None:
        """An unknown note value raises a technique error.

        Args:
            score4: An empty four-voice score.
        """
        with pytest.raises(TechniqueError):
            apply(
                "broken_chord",
                score4,
                voice="piano",
                key="C",
                measure_range={"start": 1, "end": 1},
                note_value="hemidemisemiquaver",
            )

    def test_parallel_chords(self, score4: stream.Score) -> None:
        """Parallel chords step by an interval.

        Args:
            score4: An empty four-voice score.
        """
        apply(
            "parallel_chords",
            score4,
            voice="piano",
            key="C",
            measure_range={"start": 1, "end": 4},
            repetitions=4,
        )

    def test_parallel_chords_overflow(self, score4: stream.Score) -> None:
        """Too many repetitions yields a warning.

        Args:
            score4: An empty four-voice score.
        """
        result = apply(
            "parallel_chords",
            score4,
            voice="piano",
            key="C",
            measure_range={"start": 1, "end": 2},
            repetitions=4,
        )
        assert result.warnings

    def test_planing_triad(self, score4: stream.Score) -> None:
        """A planing triad slides up the scale.

        Args:
            score4: An empty four-voice score.
        """
        apply(
            "planing",
            score4,
            voice="piano",
            key="C",
            measure_range={"start": 1, "end": 4},
        )

    def test_planing_seventh(self, score4: stream.Score) -> None:
        """A planing seventh chord slides up the scale.

        Args:
            score4: An empty four-voice score.
        """
        apply(
            "planing",
            score4,
            voice="piano",
            key="C",
            measure_range={"start": 1, "end": 3},
            chord_size="seventh",
        )
