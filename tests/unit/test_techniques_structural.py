# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for techniques.structural."""

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
from harmoniatextor.techniques.structural import _require_voices


def apply(
    technique_id: str, score: stream.Score, themes: dict[int, Theme], **params: Any
) -> Any:
    """Apply a technique by identifier."""
    technique = REGISTRY.get(technique_id)
    context = TechniqueContext(score=score, themes=themes)
    return technique.apply(context, technique.params_model(**params))


REGISTRY = build_default_registry()


class TestStructural:
    """Structural techniques."""

    def test_exposition_secondary(self, score4: stream.Score, theme: Theme) -> None:
        """Exposition with a secondary theme."""
        second = Theme(2, "m01", "alto", 1, [ThemeNote("E4", 1.0)], "r-0")
        apply(
            "exposition",
            score4,
            {1: theme, 2: second},
            main_theme_id=1,
            secondary_theme_id=2,
            tonic="C",
            dominant="G",
            measure_start=1,
        )

    def test_exposition_generated_secondary(
        self, score4: stream.Score, theme: Theme
    ) -> None:
        """Exposition can generate a secondary theme from the main one."""
        apply(
            "exposition",
            score4,
            {1: theme},
            main_theme_id=1,
            tonic="C",
            dominant="G",
            measure_start=1,
        )

    def test_development_overflow(self, score4: stream.Score, theme: Theme) -> None:
        """Development warns when it runs out of measures."""
        result = apply(
            "development",
            score4,
            {1: theme},
            theme_ids=[1],
            target_keys=["a", "e", "g", "d"],
            techniques=["inversion", "retrograde", "augmentation", "diminution"],
            measure_range={"start": 1, "end": 1},
        )
        assert result.warnings

    def test_recapitulation(self, score4: stream.Score, theme: Theme) -> None:
        """Recapitulation returns themes to the tonic."""
        apply(
            "recapitulation",
            score4,
            {1: theme},
            theme_ids=[1],
            return_to_tonic="C",
            measure_start=1,
        )

    def test_rondo(self, score4: stream.Score, theme: Theme) -> None:
        """Rondo alternates the refrain with episodes."""
        second = Theme(2, "m01", "alto", 1, [ThemeNote("G4", 1.0)], "r-0")
        apply(
            "rondo",
            score4,
            {1: theme, 2: second},
            refrain_theme_id=1,
            episode_theme_ids=[2],
            tonic="C",
            measure_start=1,
        )

    def test_rondo_episode_keys(self, score4: stream.Score, theme: Theme) -> None:
        """Rondo can place episodes in explicit keys."""
        second = Theme(2, "m01", "alto", 1, [ThemeNote("G4", 1.0)], "r-0")
        apply(
            "rondo",
            score4,
            {1: theme, 2: second},
            refrain_theme_id=1,
            episode_theme_ids=[2],
            episode_keys=["a"],
            tonic="C",
            measure_start=1,
        )

    def test_stretto(self, score4: stream.Score, theme: Theme) -> None:
        """Stretto stacks entries."""
        apply(
            "stretto",
            score4,
            {1: theme},
            theme_id=1,
            entry_delay=1.0,
            voices=["soprano", "alto", "tenor"],
        )

    def test_pedal_point(self, score4: stream.Score) -> None:
        """Pedal points sustain a pitch."""
        apply(
            "pedal_point",
            score4,
            {},
            pitch="C3",
            voice="bass",
            measure_range={"start": 1, "end": 2},
        )

    def test_pedal_tone(self, score4: stream.Score) -> None:
        """Pedal tones pulse a pitch."""
        apply(
            "pedal_tone",
            score4,
            {},
            pitch="C3",
            voice="bass",
            measure_range={"start": 1, "end": 2},
        )

    def test_pedal_tone_snaps_above(self, score4: stream.Score) -> None:
        """A constrained pedal snaps upper voices to its triad."""
        editor = ScoreEditor(score4)
        editor.write_line("soprano", 1, [ThemeNote("C#5", 1.0)])
        editor.write_line("alto", 1, [ThemeNote("E4", 1.0)])
        result = apply(
            "pedal_tone",
            score4,
            {},
            pitch="C3",
            voice="bass",
            measure_range={"start": 1, "end": 1},
            above_movement=False,
        )
        assert result.warnings


def test_require_voices_refuses_default_texture() -> None:
    """A structural technique never invents a default four-part texture."""
    with pytest.raises(TechniqueError, match="add_part"):
        _require_voices(ScoreEditor(stream.Score()))
