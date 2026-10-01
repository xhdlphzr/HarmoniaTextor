# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for all 25 technique packs."""

from __future__ import annotations

from typing import Any

import pytest
from music21 import stream

from harmoniatextor.domain.models import Theme, ThemeNote
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.techniques import TechniqueContext, TechniqueError, build_default_registry
from harmoniatextor.techniques.helpers import (
    average_axis_midi,
    chord_pitches,
    functional_figure,
    get_theme,
    transpose_to_key,
)
from harmoniatextor.techniques.melodic import _smooth_leaps
from harmoniatextor.techniques.structural import _require_voices, _vary

REGISTRY = build_default_registry()

_TECHNIQUE_COUNT = 25
_AUGMENTED_DURATION = 2.0
_DIMINISHED_DURATION = 0.5


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

    def test_exposition_generated_secondary(self, score4: stream.Score, theme: Theme) -> None:
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
            "pedal_tone", score4, {}, pitch="C3", voice="bass", measure_range={"start": 1, "end": 2}
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
        """Every variation name is handled."""
        assert _vary(theme, "inversion")[0].pitch != theme.notes[0].pitch
        assert _vary(theme, "retrograde")[0].pitch == theme.notes[-1].pitch
        assert _vary(theme, "augmentation")[0].quarter_length == _AUGMENTED_DURATION
        assert _vary(theme, "diminution")[0].quarter_length == _DIMINISHED_DURATION
        assert _vary(theme, "none")[0].pitch == theme.notes[0].pitch

    def test_chromatic_flat(self, score4: stream.Score) -> None:
        """A flat alteration lowers a degree."""
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
        """Notes outside the altered degree are untouched."""
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

    def test_syncopation_empty_pattern(self, score4: stream.Score, theme: Theme) -> None:
        """An empty pattern is rejected."""
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
        """A non-numeric ratio is rejected."""
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
        """A zero numerator is rejected."""
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


class TestRegistry:
    """Registry behaviour."""

    def test_len_and_ids(self) -> None:
        """All 25 techniques are registered."""
        assert len(REGISTRY) == _TECHNIQUE_COUNT
        assert "imitation" in REGISTRY

    def test_unknown(self) -> None:
        """Unknown techniques raise KeyError."""
        with pytest.raises(KeyError):
            REGISTRY.get("nope")

    def test_duplicate(self) -> None:
        """Duplicate registration is rejected."""
        with pytest.raises(ValueError, match="duplicate"):
            REGISTRY.register(REGISTRY.get("imitation"))

    def test_schema(self) -> None:
        """Techniques expose a JSON schema."""
        assert "properties" in REGISTRY.get("imitation").schema()


def test_require_voices_refuses_default_texture() -> None:
    """A structural technique never invents a default four-part texture."""
    with pytest.raises(TechniqueError, match="add_part"):
        _require_voices(ScoreEditor(stream.Score()))
