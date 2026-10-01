# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for techniques.harmonic."""

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


REGISTRY = build_default_registry()


class TestHarmonic:
    """Harmonic techniques."""

    def test_functional_cycle(self, score4: stream.Score) -> None:
        """A functional cycle is realised."""
        apply("functional_cycle", score4, {}, key="C", measure_start=1)

    def test_dominant_seventh(self, score4: stream.Score) -> None:
        """A dominant seventh is placed."""
        apply("dominant_seventh", score4, {}, key="C", measure_position="2+1", voice="bass")

    def test_diminished_seventh(self, score4: stream.Score) -> None:
        """A diminished seventh resolves to a target key."""
        apply("diminished_seventh", score4, {}, key="C", target_key="G", measure_position="3+1")

    def test_harmonic_sequence(self, score4: stream.Score) -> None:
        """A harmonic sequence steps through keys."""
        apply("harmonic_sequence", score4, {}, chord_sequence=["I", "V"], repetitions=2, key="C")

    def test_chromatic_harmony(self, score4: stream.Score) -> None:
        """A degree can be raised chromatically."""
        ScoreEditor(score4).write_line("soprano", 1, [ThemeNote("F5", 1.0)])
        apply(
            "chromatic_harmony",
            score4,
            {},
            key="C",
            chromatic_degree="IV#",
            measure_range={"start": 1, "end": 1},
        )

    def test_chromatic_harmony_no_accidental(self, score4: stream.Score) -> None:
        """A degree without an accidental yields a warning."""
        result = apply(
            "chromatic_harmony",
            score4,
            {},
            key="C",
            chromatic_degree="IV",
            measure_range={"start": 1, "end": 1},
        )
        assert result.warnings

    def test_chromatic_harmony_unresolvable(self, score4: stream.Score) -> None:
        """An unresolvable degree yields a warning."""
        result = apply(
            "chromatic_harmony",
            score4,
            {},
            key="C",
            chromatic_degree="Q#",
            measure_range={"start": 1, "end": 1},
        )
        assert result.warnings

    def test_modulation_bridge_auto(self, score4: stream.Score) -> None:
        """An automatic bridge is realised."""
        apply(
            "modulation_bridge",
            score4,
            {},
            start_key="C",
            target_key="G",
            measure_range={"start": 1, "end": 4},
        )

    def test_modulation_bridge_explicit(self, score4: stream.Score) -> None:
        """Explicit bridge chords are used."""
        apply(
            "modulation_bridge",
            score4,
            {},
            start_key="C",
            target_key="G",
            bridge_chords=["V", "I"],
            measure_range={"start": 1, "end": 2},
        )

    def test_modulation_bridge_overflow(self, score4: stream.Score) -> None:
        """A short range drops extra chords with a warning."""
        result = apply(
            "modulation_bridge",
            score4,
            {},
            start_key="C",
            target_key="G",
            bridge_chords=["I", "IV", "V", "I"],
            measure_range={"start": 1, "end": 2},
        )
        assert result.warnings
