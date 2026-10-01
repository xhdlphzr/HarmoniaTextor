# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for techniques.melodic."""

from __future__ import annotations

from typing import Any

from music21 import stream

from harmoniatextor.domain.models import Theme, ThemeNote
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.techniques import TechniqueContext, build_default_registry


def apply(technique_id: str, score: stream.Score, themes: dict[int, Theme], **params: Any) -> Any:
    """Apply a technique by identifier."""
    technique = REGISTRY.get(technique_id)
    context = TechniqueContext(score=score, themes=themes)
    return technique.apply(context, technique.params_model(**params))


def big_leap_theme() -> Theme:
    """Return a theme with a leap larger than an octave."""
    return Theme(
        id=2,
        movement_id="m01",
        voice="soprano",
        start_measure=1,
        notes=[ThemeNote("C5", 1.0), ThemeNote("C7", 1.0)],
        created_revision="r-0",
    )


REGISTRY = build_default_registry()


class TestMelodic:
    """Melodic techniques."""

    def test_imitation_strict(self, score4: stream.Score, theme: Theme) -> None:
        """Strict imitation transposes literally."""
        result = apply(
            "imitation", score4, {1: theme}, theme_id=1, target_voice="bass", delay_measures=1
        )
        assert result.score is score4

    def test_imitation_non_strict(self, score4: stream.Score) -> None:
        """Non-strict imitation smooths large leaps."""
        result = apply(
            "imitation",
            score4,
            {2: big_leap_theme()},
            theme_id=2,
            target_voice="bass",
            strict=False,
        )
        assert result.warnings

    def test_inversion_auto(self, score4: stream.Score, theme: Theme) -> None:
        """Automatic inversion uses the average axis."""
        assert (
            apply("inversion", score4, {1: theme}, theme_id=1, target_voice="alto").score is score4
        )

    def test_inversion_explicit_axis(self, score4: stream.Score, theme: Theme) -> None:
        """An explicit axis is honoured."""
        assert (
            apply("inversion", score4, {1: theme}, theme_id=1, axis="C4", target_voice="alto").score
            is score4
        )

    def test_retrograde_preserve(self, score4: stream.Score, theme: Theme) -> None:
        """Retrograde preserves rhythm by default."""
        assert (
            apply("retrograde", score4, {1: theme}, theme_id=1, target_voice="alto").score is score4
        )

    def test_retrograde_free(self, score4: stream.Score, theme: Theme) -> None:
        """Retrograde can reverse rhythm too."""
        assert apply(
            "retrograde", score4, {1: theme}, theme_id=1, target_voice="alto", preserve_rhythm=False
        )

    def test_augmentation(self, score4: stream.Score, theme: Theme) -> None:
        """Augmentation lengthens durations."""
        apply("augmentation", score4, {1: theme}, theme_id=1, target_voice="alto")

    def test_diminution(self, score4: stream.Score, theme: Theme) -> None:
        """Diminution shortens durations."""
        apply("diminution", score4, {1: theme}, theme_id=1, target_voice="alto")

    def test_transposition_interval(self, score4: stream.Score, theme: Theme) -> None:
        """Transposition by interval."""
        apply("transposition", score4, {1: theme}, theme_id=1, interval=5, target_voice="alto")

    def test_transposition_key(self, score4: stream.Score, theme: Theme) -> None:
        """Transposition into an absolute key."""
        apply("transposition", score4, {1: theme}, theme_id=1, target_key="G", target_voice="alto")

    def test_sequence(self, score4: stream.Score, theme: Theme) -> None:
        """A melodic sequence repeats and steps."""
        apply("sequence", score4, {1: theme}, theme_id=1, target_voice="alto", repetitions=3)

    def test_voice_exchange_themes(self, score4: stream.Score, theme: Theme) -> None:
        """Voice exchange by themes."""
        second = Theme(
            id=2,
            movement_id="m01",
            voice="bass",
            start_measure=1,
            notes=[ThemeNote("C3", 1.0), ThemeNote("G3", 1.0)],
            created_revision="r-0",
        )

        editor = ScoreEditor(score4)
        editor.write_line("soprano", 1, theme.notes)
        editor.write_line("bass", 1, second.notes)
        result = apply(
            "voice_exchange",
            score4,
            {1: theme, 2: second},
            theme_id_1=1,
            theme_id_2=2,
            measure_range={"start": 1, "end": 1},
        )
        assert result.score is score4

    def test_voice_exchange_voices(self, score4: stream.Score) -> None:
        """Voice exchange by explicit voices."""
        apply(
            "voice_exchange",
            score4,
            {},
            voice_1="soprano",
            voice_2="bass",
            measure_range={"start": 1, "end": 1},
        )
