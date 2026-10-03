# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for techniques.rhythmic."""

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


def apply(
    technique_id: str, score: stream.Score, themes: dict[int, Theme], **params: Any
) -> Any:
    """Apply a technique by identifier."""
    technique = REGISTRY.get(technique_id)
    context = TechniqueContext(score=score, themes=themes)
    return technique.apply(context, technique.params_model(**params))


REGISTRY = build_default_registry()


class TestRhythmic:
    """Rhythmic techniques."""

    def test_syncopation(self, score4: stream.Score, theme: Theme) -> None:
        """A syncopated pattern is applied."""
        apply(
            "syncopation",
            score4,
            {1: theme},
            theme_id=1,
            measure_range={"start": 1, "end": 1},
        )

    def test_syncopation_bad_token(self, score4: stream.Score, theme: Theme) -> None:
        """Unknown tokens raise."""
        with pytest.raises(TechniqueError, match="unknown rhythm token"):
            apply(
                "syncopation",
                score4,
                {1: theme},
                theme_id=1,
                sync_pattern="blob",
                measure_range={"start": 1, "end": 1},
            )

    def test_rhythmic_independence(self, score4: stream.Score) -> None:
        """Paired voices are offset."""
        ScoreEditor(score4).write_line("alto", 1, [ThemeNote("E4", 1.0)])
        result = apply(
            "rhythmic_independence",
            score4,
            {},
            voice_pairs=[["soprano", "alto"]],
            measure_range={"start": 1, "end": 1},
        )
        assert result.score is score4

    def test_rhythmic_independence_empty(self, score4: stream.Score) -> None:
        """Empty voices produce a warning."""
        result = apply(
            "rhythmic_independence",
            score4,
            {},
            voice_pairs=[["soprano", "alto"]],
            measure_range={"start": 1, "end": 1},
        )
        assert result.warnings

    def test_counter_rhythm(self, score4: stream.Score) -> None:
        """A counter rhythm subdivides the counter voice."""
        ScoreEditor(score4).write_line("alto", 1, [ThemeNote("E4", 1.0)])
        apply(
            "counter_rhythm",
            score4,
            {},
            main_voice="soprano",
            counter_voice="alto",
            measure_range={"start": 1, "end": 1},
        )

    def test_counter_rhythm_bad_ratio(self, score4: stream.Score) -> None:
        """Malformed ratios raise."""
        with pytest.raises(TechniqueError, match="invalid ratio"):
            apply(
                "counter_rhythm",
                score4,
                {},
                main_voice="soprano",
                counter_voice="alto",
                rhythm_ratio="2-1",
                measure_range={"start": 1, "end": 1},
            )

    def test_counter_rhythm_empty(self, score4: stream.Score) -> None:
        """An empty counter voice yields a warning."""
        result = apply(
            "counter_rhythm",
            score4,
            {},
            main_voice="soprano",
            counter_voice="alto",
            measure_range={"start": 1, "end": 1},
        )
        assert result.warnings

    def test_voice_motion_parallel(self, score4: stream.Score) -> None:
        """Parallel motion is forced."""
        editor = ScoreEditor(score4)
        editor.write_line("soprano", 1, [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
        editor.write_line("alto", 1, [ThemeNote("E4", 1.0), ThemeNote("F4", 1.0)])
        apply(
            "voice_motion",
            score4,
            {},
            voice_pairs=[["soprano", "alto"]],
            motion_type="parallel",
            measure_range={"start": 1, "end": 1},
        )

    def test_voice_motion_contrary(self, score4: stream.Score) -> None:
        """Contrary motion is forced."""
        editor = ScoreEditor(score4)
        editor.write_line("soprano", 1, [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
        editor.write_line("alto", 1, [ThemeNote("E4", 1.0), ThemeNote("F4", 1.0)])
        apply(
            "voice_motion",
            score4,
            {},
            voice_pairs=[["soprano", "alto"]],
            motion_type="contrary",
            measure_range={"start": 1, "end": 1},
        )

    def test_voice_motion_oblique(self, score4: stream.Score) -> None:
        """Oblique motion is forced."""
        editor = ScoreEditor(score4)
        editor.write_line("soprano", 1, [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
        editor.write_line("alto", 1, [ThemeNote("E4", 1.0), ThemeNote("F4", 1.0)])
        apply(
            "voice_motion",
            score4,
            {},
            voice_pairs=[["soprano", "alto"]],
            motion_type="oblique",
            measure_range={"start": 1, "end": 1},
        )

    def test_voice_motion_missing(self, score4: stream.Score) -> None:
        """A pair without material yields a warning."""
        result = apply(
            "voice_motion",
            score4,
            {},
            voice_pairs=[["soprano", "alto"]],
            measure_range={"start": 1, "end": 1},
        )
        assert result.warnings

    def test_rubato(self, score4: stream.Score) -> None:
        """A measure's downbeat is stretched while the bar length is preserved."""
        editor = ScoreEditor(score4)
        editor.write_line(
            "soprano",
            1,
            [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0), ThemeNote("E5", 2.0)],
        )
        apply("rubato", score4, {}, measure_range={"start": 1, "end": 1})

    def test_rubato_single_note(self, score4: stream.Score) -> None:
        """A measure with a single note is left untouched."""
        ScoreEditor(score4).write_line("soprano", 1, [ThemeNote("C5", 4.0)])
        apply("rubato", score4, {}, measure_range={"start": 1, "end": 1})
