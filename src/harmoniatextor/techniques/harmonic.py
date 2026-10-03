# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Harmonic laying techniques (functional cycles, cadences, modulation, ...)."""

from __future__ import annotations

from harmoniatextor.domain.interval import DiatonicInterval, parse_interval
from harmoniatextor.domain.key import parse_key
from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.domain.params import (
    ChromaticHarmonyParams,
    ChromaticModulationParams,
    ColorChordParams,
    DiminishedSeventhParams,
    DominantSeventhParams,
    ExtendedHarmonyParams,
    FunctionalCycleParams,
    HarmonicSequenceParams,
    ModalHarmonyParams,
    ModulationBridgeParams,
    WholeToneParams,
    parse_measure_position,
)
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.techniques.base import (
    Technique,
    TechniqueCategory,
    TechniqueContext,
    TechniqueResult,
)
from harmoniatextor.techniques.helpers import (
    Placement,
    chord_pitches,
    functional_figure,
    midi_to_name,
    name_to_midi,
    note_value,
    realize_chord,
    transpose_note,
)

__all__ = [
    "ChromaticHarmonyTechnique",
    "ChromaticModulationTechnique",
    "ColorChordTechnique",
    "DiminishedSeventhTechnique",
    "DominantSeventhTechnique",
    "ExtendedHarmonyTechnique",
    "FunctionalCycleTechnique",
    "HarmonicSequenceTechnique",
    "ModalHarmonyTechnique",
    "ModulationBridgeTechnique",
    "WholeToneTechnique",
]

_EXTENSION_SEMITONES: dict[str, int] = {"7": 10, "9": 14, "11": 17, "13": 21}
_COLOR_SEMITONES: dict[str, int] = {
    "6": 9,
    "b9": 13,
    "9": 14,
    "#9": 15,
    "11": 17,
    "#11": 18,
    "b13": 20,
    "13": 21,
}
_MODE_PROGRESSIONS: dict[str, list[str]] = {
    "ionian": ["I", "IV", "V", "I"],
    "dorian": ["i", "IV", "VII", "i"],
    "phrygian": ["i", "bII", "bVII", "i"],
    "lydian": ["I", "II", "V", "I"],
    "mixolydian": ["I", "bVII", "IV", "I"],
    "aeolian": ["i", "iv", "v", "i"],
}
_WHOLE_TONE_SEMITONES = 2
_WHOLE_TONE_SIZE = 6


def _transpose_key(key: str, interval: DiatonicInterval) -> str:
    """Transpose a key specification by a diatonic interval.

    Args:
        key: Source key.
        interval: Interval to apply.

    Returns:
        The transposed key specification.
    """
    spec = parse_key(key)
    tonic = transpose_note(f"{spec.tonic}4", interval)
    letter = tonic[0]
    accidental = tonic[1] if len(tonic) > 1 and tonic[1] in "#-" else ""
    accidental = "#" if accidental == "#" else ("b" if accidental == "-" else "")
    return f"{letter}{accidental}" if spec.is_major else f"{letter.lower()}{accidental}"


class FunctionalCycleTechnique(Technique[FunctionalCycleParams]):
    """Realise a tonic-subdominant-dominant functional cycle."""

    id = "functional_cycle"
    name = "功能圈"
    category = TechniqueCategory.HARMONIC
    summary = "Realise a T-S-D functional progression as four-part harmony."
    params_model = FunctionalCycleParams

    def apply(
        self, ctx: TechniqueContext, params: FunctionalCycleParams
    ) -> TechniqueResult:
        """Realise the cycle.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        figures = [
            params.start_chord,
            *(functional_figure(params.key, f) for f in params.sequence),
        ]
        for index, figure in enumerate(figures):
            realize_chord(
                editor,
                params.key,
                figure,
                Placement(params.measure_start + index, 0.0, bar),
            )
        return TechniqueResult(ctx.score)


class DominantSeventhTechnique(Technique[DominantSeventhParams]):
    """Place a dominant seventh chord and prepare its resolution."""

    id = "dominant_seventh"
    name = "属七和弦"
    category = TechniqueCategory.HARMONIC
    summary = "Realise a V7 (or applied V7) at a specific position."
    params_model = DominantSeventhParams

    def apply(
        self, ctx: TechniqueContext, params: DominantSeventhParams
    ) -> TechniqueResult:
        """Realise the dominant seventh.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        measure, beat = parse_measure_position(params.measure_position)
        editor = ScoreEditor(ctx.score)
        pitches = chord_pitches(params.key, params.chord)
        realize_chord(
            editor, params.key, params.chord, Placement(measure, beat - 1.0, 1.0)
        )
        editor.place_note(params.voice, measure, beat - 1.0, pitches[0], 1.0)
        return TechniqueResult(ctx.score)


class DiminishedSeventhTechnique(Technique[DiminishedSeventhParams]):
    """Place a diminished seventh and resolve it towards a target key."""

    id = "diminished_seventh"
    name = "减七和弦"
    category = TechniqueCategory.HARMONIC
    summary = "Realise a vii°7 and pivot it towards a target key."
    params_model = DiminishedSeventhParams

    def apply(
        self, ctx: TechniqueContext, params: DiminishedSeventhParams
    ) -> TechniqueResult:
        """Realise the diminished seventh and its resolution.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        measure, beat = parse_measure_position(params.measure_position)
        editor = ScoreEditor(ctx.score)
        realize_chord(
            editor, params.key, params.chord, Placement(measure, beat - 1.0, 1.0)
        )
        target = parse_key(params.target_key)
        tonic_figure = "I" if target.is_major else "i"
        realize_chord(
            editor,
            params.target_key,
            tonic_figure,
            Placement(measure + 1, 0.0, editor.bar_length()),
        )
        return TechniqueResult(ctx.score)


class HarmonicSequenceTechnique(Technique[HarmonicSequenceParams]):
    """Repeat a chord template while transposing it stepwise."""

    id = "harmonic_sequence"
    name = "和声模进"
    category = TechniqueCategory.HARMONIC
    summary = "Sequence a chord template, moving the whole pattern by an interval."
    params_model = HarmonicSequenceParams

    def apply(
        self, ctx: TechniqueContext, params: HarmonicSequenceParams
    ) -> TechniqueResult:
        """Realise a harmonic sequence.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        step = parse_interval(params.step_interval)
        key = params.key
        measure = 1
        for _ in range(params.repetitions):
            for figure in params.chord_sequence:
                realize_chord(editor, key, figure, Placement(measure, 0.0, bar))
                measure += 1
            key = _transpose_key(key, step)
        return TechniqueResult(ctx.score)


class ChromaticHarmonyTechnique(Technique[ChromaticHarmonyParams]):
    """Alter one scale degree chromatically within a measure range."""

    id = "chromatic_harmony"
    name = "半音化和声"
    category = TechniqueCategory.HARMONIC
    summary = "Raise or lower a scale degree by a semitone across a passage."
    params_model = ChromaticHarmonyParams

    def apply(
        self, ctx: TechniqueContext, params: ChromaticHarmonyParams
    ) -> TechniqueResult:
        """Apply chromatic alteration.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        editor = ScoreEditor(ctx.score)
        warnings: list[str] = []
        degree = params.chromatic_degree
        if degree.endswith("#"):
            delta, base = 1, degree[:-1]
        elif degree.endswith("b"):
            delta, base = -1, degree[:-1]
        else:
            warnings.append(f"degree {degree!r} has no accidental; nothing to alter")
            return TechniqueResult(ctx.score, warnings)
        try:
            target_pc = name_to_midi(chord_pitches(params.key, base)[0]) % 12
        except Exception:  # noqa: BLE001 - music21 raises many types for bad figures
            warnings.append(f"cannot resolve degree {degree!r} in key {params.key!r}")
            return TechniqueResult(ctx.score, warnings)

        def alter(_offset: float, pitch: str, _duration: float) -> str:
            midi = name_to_midi(pitch)
            if midi % 12 == target_pc:
                return midi_to_name(midi + delta)
            return pitch

        for voice in editor.voice_names():
            editor.map_notes(
                voice, params.measure_range.start, params.measure_range.end, alter
            )
        return TechniqueResult(ctx.score, warnings)


class ModulationBridgeTechnique(Technique[ModulationBridgeParams]):
    """Modulate from one key to another using pivot chords."""

    id = "modulation_bridge"
    name = "转调桥段"
    category = TechniqueCategory.HARMONIC
    summary = "Bridge two keys with an explicit pivot progression."
    params_model = ModulationBridgeParams

    def apply(
        self, ctx: TechniqueContext, params: ModulationBridgeParams
    ) -> TechniqueResult:
        """Realise a modulation bridge.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        warnings: list[str] = []
        start = params.measure_range.start
        end = params.measure_range.end
        if params.bridge_chords:
            figures = list(params.bridge_chords)
        else:
            target = parse_key(params.target_key)
            tonic = "I" if target.is_major else "i"
            figures = ["V", "IV", "V", tonic]
        span = end - start + 1
        if len(figures) > span:
            warnings.append("bridge ran out of measures; extra chords were dropped")
            figures = figures[:span]
        for index, figure in enumerate(figures):
            current_key = params.start_key if index == 0 else params.target_key
            realize_chord(
                editor, current_key, figure, Placement(start + index, 0.0, bar)
            )
        return TechniqueResult(ctx.score, warnings)


class ChromaticModulationTechnique(Technique[ChromaticModulationParams]):
    """Bridge two keys with a chromatic (Neapolitan) pivot and a cadence."""

    id = "chromatic_modulation"
    name = "半音化转调"
    category = TechniqueCategory.HARMONIC
    summary = "Bridge two keys with a chromatic Neapolitan pivot and a target cadence."
    params_model = ChromaticModulationParams

    def apply(
        self, ctx: TechniqueContext, params: ChromaticModulationParams
    ) -> TechniqueResult:
        """Realise the chromatic modulation.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        start = params.measure_range.start
        span = params.measure_range.end - start + 1
        warnings: list[str] = []
        plan = [
            (params.start_key, "I"),
            (params.target_key, "bII"),
            (params.target_key, "V"),
            (params.target_key, "I"),
        ]
        if len(plan) > span:
            warnings.append(
                "chromatic modulation ran out of measures; extra chords were dropped"
            )
            plan = plan[:span]
        for index, (key, figure) in enumerate(plan):
            realize_chord(editor, key, figure, Placement(start + index, 0.0, bar))
        return TechniqueResult(ctx.score, warnings)


def _sounding_root(editor: ScoreEditor, measure: int, key: str) -> int:
    """Return the MIDI root of a measure, preferring the lowest sounding pitch.

    Args:
        editor: Score editor.
        measure: One-based measure number.
        key: Fallback key whose tonic is used when the measure is silent.

    Returns:
        A MIDI pitch number.
    """
    midis = [
        name_to_midi(note.pitch)
        for voice in editor.voice_names()
        for note in editor.read_line(voice, measure, measure)
    ]
    if midis:
        return min(midis)
    return name_to_midi(chord_pitches(key, "I")[0])


class ExtendedHarmonyTechnique(Technique[ExtendedHarmonyParams]):
    """Stack 7/9/11/13 extensions above each measure's sounding root."""

    id = "extended_harmony"
    name = "扩展和声"
    category = TechniqueCategory.HARMONIC
    summary = "Stack 7/9/11/13 extensions above the sounding root of each measure."
    params_model = ExtendedHarmonyParams

    def apply(
        self, ctx: TechniqueContext, params: ExtendedHarmonyParams
    ) -> TechniqueResult:
        """Write the extension chords.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        for measure in range(params.measure_range.start, params.measure_range.end + 1):
            root = _sounding_root(editor, measure, params.key)
            tones = [
                midi_to_name(root + _EXTENSION_SEMITONES[extension])
                for extension in params.extensions
            ]
            editor.place_chord(params.voice, measure, 0.0, tones, bar)
        return TechniqueResult(ctx.score)


class ColorChordTechnique(Technique[ColorChordParams]):
    """Add a single colour tone above each measure's sounding root."""

    id = "color_chord"
    name = "色彩和弦"
    category = TechniqueCategory.HARMONIC
    summary = "Add a single colour tone (6/9/#11/b13) above the sounding root."
    params_model = ColorChordParams

    def apply(self, ctx: TechniqueContext, params: ColorChordParams) -> TechniqueResult:
        """Write the colour tones.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        offset = _COLOR_SEMITONES.get(params.color, _COLOR_SEMITONES["9"])
        for measure in range(params.measure_range.start, params.measure_range.end + 1):
            root = _sounding_root(editor, measure, params.key)
            editor.place_note(
                params.voice, measure, 0.0, midi_to_name(root + offset), bar
            )
        return TechniqueResult(ctx.score)


class ModalHarmonyTechnique(Technique[ModalHarmonyParams]):
    """Realise a modal chord progression derived from a church mode."""

    id = "modal_harmony"
    name = "调式和声"
    category = TechniqueCategory.HARMONIC
    summary = "Realise a modal chord progression derived from the chosen church mode."
    params_model = ModalHarmonyParams

    def apply(
        self, ctx: TechniqueContext, params: ModalHarmonyParams
    ) -> TechniqueResult:
        """Realise the modal progression.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        figures = list(params.chord_sequence) or list(_MODE_PROGRESSIONS[params.mode])
        start = params.measure_range.start
        span = params.measure_range.end - start + 1
        warnings: list[str] = []
        if len(figures) > span:
            warnings.append(
                "modal progression ran out of measures; extra chords were dropped"
            )
            figures = figures[:span]
        for index, figure in enumerate(figures):
            realize_chord(
                editor, params.key, figure, Placement(start + index, 0.0, bar)
            )
        return TechniqueResult(ctx.score, warnings)


class WholeToneTechnique(Technique[WholeToneParams]):
    """Fill a voice with a whole-tone scale run."""

    id = "whole_tone"
    name = "全音阶"
    category = TechniqueCategory.HARMONIC
    summary = "Fill a voice with a whole-tone scale run."
    params_model = WholeToneParams

    def apply(self, ctx: TechniqueContext, params: WholeToneParams) -> TechniqueResult:
        """Write the whole-tone run.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and warnings.
        """
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        length = note_value(params.note_value)
        root = name_to_midi(params.root)
        sign = -1 if params.direction == "down" else 1
        scale = [
            midi_to_name(root + sign * _WHOLE_TONE_SEMITONES * index)
            for index in range(_WHOLE_TONE_SIZE)
        ]
        span = params.measure_range.end - params.measure_range.start + 1
        count = max(1, round(span * bar / length))
        notes = [
            ThemeNote(pitch=scale[index % len(scale)], quarter_length=length)
            for index in range(count)
        ]
        editor.write_line(params.voice, params.measure_range.start, notes)
        return TechniqueResult(ctx.score)
