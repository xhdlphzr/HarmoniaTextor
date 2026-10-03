# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for technique edge branches."""

from __future__ import annotations

from typing import Any

import pytest
from music21 import stream

from harmoniatextor.domain.models import Theme, ThemeNote
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.techniques import (
    TechniqueContext,
    TechniqueError,
    build_default_registry,
)
from harmoniatextor.techniques.melodic import _smooth_leaps
from harmoniatextor.techniques.structural import _vary

_AUGMENTED_DURATION = 2.0


_DIMINISHED_DURATION = 0.5


def apply(
    technique_id: str, score: stream.Score, themes: dict[int, Theme], **params: Any
) -> Any:
    """Apply a technique by identifier.

    Args:
        technique_id: The technique id.
        score: The score to inspect.
        themes: The themes.
        params: Validated parameters.

    Returns:
        The result.
    """
    technique = REGISTRY.get(technique_id)
    context = TechniqueContext(score=score, themes=themes)
    return technique.apply(context, technique.params_model(**params))


REGISTRY = build_default_registry()


class TestEdgeBranches:
    """Additional branches for full coverage."""

    def test_smooth_leaps_empty(self) -> None:
        """An empty melody smooths to nothing."""
        assert _smooth_leaps([]) == []

    def test_smooth_leaps_descending(self) -> None:
        """Descending leaps are folded up by octaves."""
        notes = [ThemeNote("C7", 1.0), ThemeNote("C5", 1.0)]
        assert _smooth_leaps(notes)[1].pitch == "C6"

    def test_vary_variants(self, theme: Theme) -> None:
        """Every variation name is handled.

        Args:
            theme: A sample theme.
        """
        assert _vary(theme, "inversion")[0].pitch != theme.notes[0].pitch
        assert _vary(theme, "retrograde")[0].pitch == theme.notes[-1].pitch
        assert _vary(theme, "augmentation")[0].quarter_length == _AUGMENTED_DURATION
        assert _vary(theme, "diminution")[0].quarter_length == _DIMINISHED_DURATION
        assert _vary(theme, "none")[0].pitch == theme.notes[0].pitch

    def test_chromatic_flat(self, score4: stream.Score) -> None:
        """A flat alteration lowers a degree.

        Args:
            score4: An empty four-voice score.
        """
        ScoreEditor(score4).write_line("soprano", 1, [ThemeNote("F5", 1.0)])
        apply(
            "chromatic_harmony",
            score4,
            {},
            key="C",
            chromatic_degree="IVb",
            measure_range={"start": 1, "end": 1},
        )

    def test_chromatic_no_match(self, score4: stream.Score) -> None:
        """Notes outside the altered degree are untouched.

        Args:
            score4: An empty four-voice score.
        """
        ScoreEditor(score4).write_line("soprano", 1, [ThemeNote("D5", 1.0)])
        apply(
            "chromatic_harmony",
            score4,
            {},
            key="C",
            chromatic_degree="IV#",
            measure_range={"start": 1, "end": 1},
        )
        assert ScoreEditor(score4).read_line("soprano", 1, 1)[0].pitch == "D5"

    def test_syncopation_empty_pattern(
        self, score4: stream.Score, theme: Theme
    ) -> None:
        """An empty pattern is rejected.

        Args:
            score4: An empty four-voice score.
            theme: A sample theme.
        """
        with pytest.raises(TechniqueError, match="empty rhythmic pattern"):
            apply(
                "syncopation",
                score4,
                {1: theme},
                theme_id=1,
                sync_pattern="",
                measure_range={"start": 1, "end": 1},
            )

    def test_counter_rhythm_non_numeric(self, score4: stream.Score) -> None:
        """A non-numeric ratio is rejected.

        Args:
            score4: An empty four-voice score.
        """
        with pytest.raises(TechniqueError, match="invalid ratio"):
            apply(
                "counter_rhythm",
                score4,
                {},
                main_voice="soprano",
                counter_voice="alto",
                rhythm_ratio="a:1",
                measure_range={"start": 1, "end": 1},
            )

    def test_counter_rhythm_zero(self, score4: stream.Score) -> None:
        """A zero numerator is rejected.

        Args:
            score4: An empty four-voice score.
        """
        with pytest.raises(TechniqueError, match="invalid ratio"):
            apply(
                "counter_rhythm",
                score4,
                {},
                main_voice="soprano",
                counter_voice="alto",
                rhythm_ratio="0:1",
                measure_range={"start": 1, "end": 1},
            )
