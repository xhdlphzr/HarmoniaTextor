# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Validated parameter models shared by every technique and tool.

Each technique exposes exactly one parameter model.  The models are the single
source of truth for the JSON schema used by the LangChain tools and the HTTP
API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AddMovementParams",
    "AddPartParams",
    "AlbertiBassParams",
    "AnnotateParams",
    "AugmentationParams",
    "BrokenChordParams",
    "ChromaticHarmonyParams",
    "ChromaticModulationParams",
    "ColorChordParams",
    "CounterRhythmParams",
    "DeleteMeasureParams",
    "DevelopmentParams",
    "DiminishedSeventhParams",
    "DiminutionParams",
    "DominantSeventhParams",
    "EditParams",
    "ExpositionParams",
    "ExtendedHarmonyParams",
    "FreeVoiceLeadingParams",
    "FunctionalCycleParams",
    "HarmonicSequenceParams",
    "ImitationParams",
    "InsertMeasureParams",
    "InversionParams",
    "MeasurePosition",
    "MeasureRange",
    "ModalHarmonyParams",
    "ModulationBridgeParams",
    "ParallelChordsParams",
    "PedalPointParams",
    "PedalToneParams",
    "PlaningParams",
    "RecapitulationParams",
    "RemovePartParams",
    "RetrogradeParams",
    "RhythmicIndependenceParams",
    "RondoParams",
    "RubatoParams",
    "SequenceParams",
    "SetMovementPromptParams",
    "SetTempoParams",
    "SetTimeSignatureParams",
    "SetTitleParams",
    "StrettoParams",
    "SubmitThemeParams",
    "SyncopationParams",
    "TranspositionParams",
    "VoiceExchangeParams",
    "VoiceMotionParams",
    "WholeToneParams",
    "parse_measure_position",
]

_MODEL_CONFIG = ConfigDict(extra="forbid")


class _Params(BaseModel):
    """Base class for all parameter models."""

    model_config = _MODEL_CONFIG


class MeasureRange(_Params):
    """An inclusive, one-based measure range.

    Attributes:
        start: First measure.
        end: Last measure.
    """

    start: int = Field(ge=1, description="First measure (one-based).")
    end: int = Field(ge=1, description="Last measure (one-based, inclusive).")

    @model_validator(mode="after")
    def _check_order(self) -> MeasureRange:
        """Ensure ``start`` does not exceed ``end``.

        Returns:
            The check order result.

        Raises:
            ValueError: When the operation cannot proceed.
        """
        if self.start > self.end:
            raise ValueError("measure range start must not exceed end")
        return self


def parse_measure_position(value: str) -> tuple[int, float]:
    """Parse a ``"measure+beat"`` position string.

    Args:
        value: Position such as ``"12+1.5"``; a bare measure defaults to beat 1.

    Returns:
        A ``(measure, beat)`` tuple.

    Raises:
        ValueError: If the string is malformed or the measure is below one.
    """
    head, _, tail = value.partition("+")
    try:
        measure = int(head)
        beat = float(tail) if tail else 1.0
    except ValueError as exc:
        raise ValueError(f"invalid measure position: {value!r}") from exc
    if measure < 1 or beat <= 0:
        raise ValueError(f"invalid measure position: {value!r}")
    return measure, beat


class MeasurePosition(_Params):
    """A position expressed as measure plus beat.

    Attributes:
        measure_position: String of the form ``"12+1.5"``.
    """

    measure_position: str = Field(description='Position such as "12+1.5".')

    @field_validator("measure_position")
    @classmethod
    def _validate_position(cls, value: str) -> str:
        """Validate the position string eagerly.

        Args:
            value: Raw value.

        Returns:
            The resulting text.
        """
        parse_measure_position(value)
        return value


class _MelodicParams(_Params):
    """Common parameters for techniques that transform one theme."""

    theme_id: int = Field(ge=1, description="Theme number to transform.")
    target_voice: str = Field(min_length=1, description="Voice slot to write into.")


class ImitationParams(_MelodicParams):
    """Parameters for the imitation technique."""

    delay_measures: int = Field(
        default=2, ge=0, description="Delay before entry, in measures."
    )
    interval: int = Field(default=5, description="Signed interval; +5 is the dominant.")
    strict: bool = Field(default=True, description="Whether the imitation is literal.")


class InversionParams(_MelodicParams):
    """Parameters for the inversion technique."""

    axis: str = Field(
        default="auto", description='Mirror axis pitch such as "C4", or "auto".'
    )


class RetrogradeParams(_MelodicParams):
    """Parameters for the retrograde technique."""

    preserve_rhythm: bool = Field(default=True, description="Keep original durations.")


class AugmentationParams(_MelodicParams):
    """Parameters for the augmentation technique."""

    factor: int = Field(default=2, ge=2, description="Duration multiplier.")


class DiminutionParams(_MelodicParams):
    """Parameters for the diminution technique."""

    factor: int = Field(default=2, ge=2, description="Duration divisor.")


class TranspositionParams(_MelodicParams):
    """Parameters for the transposition technique."""

    interval: int = Field(
        default=5, description="Signed interval used when no target key."
    )
    target_key: str | None = Field(
        default=None, description='Absolute target key, e.g. "G".'
    )


class SequenceParams(_MelodicParams):
    """Parameters for the sequence technique."""

    step_interval: int = Field(default=2, description="Interval moved per repetition.")
    repetitions: int = Field(
        default=3, ge=1, le=16, description="Number of repetitions."
    )
    start_measure: int = Field(
        default=1, ge=1, description="First measure of the sequence."
    )


class VoiceExchangeParams(_Params):
    """Parameters for the voice exchange technique."""

    theme_id_1: int | None = Field(default=None, ge=1, description="First theme.")
    theme_id_2: int | None = Field(default=None, ge=1, description="Second theme.")
    voice_1: str | None = Field(
        default=None, description="First voice when no themes given."
    )
    voice_2: str | None = Field(
        default=None, description="Second voice when no themes given."
    )
    measure_range: MeasureRange = Field(
        description="Measures where voices are exchanged."
    )

    @model_validator(mode="after")
    def _check_targets(self) -> VoiceExchangeParams:
        """Require either two themes or two voices.

        Returns:
            The check targets result.

        Raises:
            ValueError: When the operation cannot proceed.
        """
        has_themes = self.theme_id_1 is not None and self.theme_id_2 is not None
        has_voices = self.voice_1 is not None and self.voice_2 is not None
        if not has_themes and not has_voices:
            raise ValueError("voice exchange needs two themes or two voices")
        return self


class ExpositionParams(_Params):
    """Parameters for the exposition technique."""

    main_theme_id: int = Field(ge=1, description="Primary theme.")
    secondary_theme_id: int | None = Field(
        default=None, ge=1, description="Optional secondary theme."
    )
    tonic: str = Field(min_length=1, description="Home key.")
    dominant: str = Field(min_length=1, description="Contrasting key.")
    measure_start: int = Field(ge=1, description="First measure of the exposition.")


class DevelopmentParams(_Params):
    """Parameters for the development technique."""

    theme_ids: list[int] = Field(min_length=1, description="Themes to develop.")
    target_keys: list[str] = Field(
        min_length=1, description="Key sequence to pass through."
    )
    techniques: list[str] = Field(
        default_factory=lambda: ["inversion"], description="Techniques used."
    )
    measure_range: MeasureRange = Field(
        description="Measures occupied by the development."
    )


class RecapitulationParams(_Params):
    """Parameters for the recapitulation technique."""

    theme_ids: list[int] = Field(min_length=1, description="Themes to recapitulate.")
    return_to_tonic: str = Field(min_length=1, description="Key to return to.")
    measure_start: int = Field(ge=1, description="First measure of the recapitulation.")


class RondoParams(_Params):
    """Parameters for the rondo form technique.

    Attributes:
        refrain_theme_id: The recurring refrain theme (A).
        episode_theme_ids: Contrasting episode themes (B, C, ...) in order.
        episode_keys: Optional key per episode; empty uses the dominant.
        tonic: Home key the refrain returns to.
        measure_start: First measure of the rondo.
    """

    refrain_theme_id: int = Field(ge=1, description="Recurring refrain theme (A).")
    episode_theme_ids: list[int] = Field(
        min_length=1, description="Contrasting episode themes (B, C, ...) in order."
    )
    episode_keys: list[str] = Field(
        default_factory=list,
        description="Optional key per episode; empty uses the dominant.",
    )
    tonic: str = Field(min_length=1, description="Home key the refrain returns to.")
    measure_start: int = Field(ge=1, description="First measure of the rondo.")


class StrettoParams(_Params):
    """Parameters for the stretto technique."""

    theme_id: int = Field(ge=1, description="Theme to stack.")
    entry_delay: float = Field(
        default=2.0, gt=0, description="Delay between entries, in beats."
    )
    voices: list[str] = Field(
        min_length=2, description="Voices participating, in entry order."
    )


class PedalPointParams(_Params):
    """Parameters for the sustained pedal point technique."""

    pitch: str = Field(min_length=1, description='Sustained pitch such as "G2".')
    voice: str = Field(min_length=1, description="Voice slot holding the pedal.")
    measure_range: MeasureRange = Field(description="Measures of the pedal.")
    pedal_type: Literal["tonic", "dominant"] = Field(
        default="tonic", description="Pedal function."
    )


class PedalToneParams(_Params):
    """Parameters for the pulsed pedal tone technique."""

    pitch: str = Field(min_length=1, description="Fixed pitch.")
    voice: str = Field(min_length=1, description="Voice slot holding the pedal.")
    measure_range: MeasureRange = Field(description="Measures of the pedal.")
    above_movement: bool = Field(
        default=True, description="Let upper voices move freely."
    )


class FunctionalCycleParams(_Params):
    """Parameters for the functional cycle technique."""

    start_chord: str = Field(default="I", description="Starting chord.")
    sequence: list[str] = Field(
        default_factory=lambda: ["T", "S", "D", "T"], description="Functions."
    )
    key: str = Field(min_length=1, description="Key of the cycle.")
    measure_start: int = Field(
        default=1, ge=1, description="First measure of the cycle."
    )


class DominantSeventhParams(MeasurePosition):
    """Parameters for the dominant seventh technique."""

    chord: str = Field(default="V7", description="Chord symbol such as V7 or V7/IV.")
    key: str = Field(min_length=1, description="Current key.")
    voice: str = Field(min_length=1, description="Voice receiving the root.")


class DiminishedSeventhParams(MeasurePosition):
    """Parameters for the diminished seventh technique."""

    chord: str = Field(default="vii°7", description="Chord symbol.")
    key: str = Field(min_length=1, description="Current key.")
    target_key: str = Field(min_length=1, description="Key to modulate towards.")


class HarmonicSequenceParams(_Params):
    """Parameters for the harmonic sequence technique."""

    chord_sequence: list[str] = Field(min_length=1, description="Chord template.")
    step_interval: int = Field(default=2, description="Interval moved per repetition.")
    repetitions: int = Field(
        default=3, ge=1, le=8, description="Number of repetitions."
    )
    key: str = Field(min_length=1, description="Starting key.")


class ChromaticHarmonyParams(_Params):
    """Parameters for the chromatic harmony technique."""

    measure_range: MeasureRange = Field(description="Measures to chromatically alter.")
    chromatic_degree: str = Field(min_length=1, description='Degree such as "IV#".')
    key: str = Field(min_length=1, description="Current key.")


class ModulationBridgeParams(_Params):
    """Parameters for the modulation bridge technique."""

    start_key: str = Field(min_length=1, description="Key to leave.")
    target_key: str = Field(min_length=1, description="Key to reach.")
    bridge_chords: list[str] | None = Field(
        default=None, description="Optional pivot chords."
    )
    measure_range: MeasureRange = Field(description="Measures of the bridge.")


class SyncopationParams(_Params):
    """Parameters for the syncopation technique."""

    theme_id: int = Field(ge=1, description="Theme to syncopate.")
    measure_range: MeasureRange = Field(description="Measures to affect.")
    sync_pattern: str = Field(
        default="quarter-half-quarter", description="Rhythmic pattern."
    )


class RhythmicIndependenceParams(_Params):
    """Parameters for the rhythmic independence technique."""

    voice_pairs: list[list[str]] = Field(
        min_length=1, description="Voice pairs to differentiate."
    )
    measure_range: MeasureRange = Field(description="Measures to affect.")


class CounterRhythmParams(_Params):
    """Parameters for the counter-rhythm technique."""

    main_voice: str = Field(min_length=1, description="Reference voice.")
    counter_voice: str = Field(min_length=1, description="Voice to re-rhythm.")
    rhythm_ratio: str = Field(
        default="2:1", description='Ratio such as "2:1" or "3:2".'
    )
    measure_range: MeasureRange = Field(description="Measures to affect.")


class VoiceMotionParams(_Params):
    """Parameters for the voice motion technique."""

    voice_pairs: list[list[str]] = Field(
        min_length=1, description="Voice pairs to move."
    )
    motion_type: Literal["parallel", "contrary", "oblique"] = Field(
        default="contrary", description="Required motion type."
    )
    measure_range: MeasureRange = Field(description="Measures to affect.")


class AlbertiBassParams(_Params):
    """Parameters for the Alberti-bass accompaniment technique.

    Attributes:
        voice: Voice slot that receives the Alberti figure.
        measure_range: Measures to fill.
        key: Key used to realise the chords.
        chord_sequence: Roman-numeral chord per measure, cycled when shorter
            than the range.
    """

    voice: str = Field(
        min_length=1, description="Voice slot receiving the Alberti figure."
    )
    measure_range: MeasureRange = Field(description="Measures to fill.")
    key: str = Field(min_length=1, description="Key used to realise the chords.")
    chord_sequence: list[str] = Field(
        default_factory=lambda: ["I", "V", "vi", "IV"],
        description="Roman-numeral chord per measure, cycled when shorter than the range.",
    )


class BrokenChordParams(_Params):
    """Parameters for the broken-chord (arpeggio) technique.

    Attributes:
        voice: Voice slot that receives the arpeggio.
        measure_range: Measures to fill.
        key: Key used to realise the chord.
        chord: Roman-numeral chord to arpeggiate.
        note_value: Note value name such as ``"eighth"``.
        direction: Arpeggio direction.
    """

    voice: str = Field(min_length=1, description="Voice slot receiving the arpeggio.")
    measure_range: MeasureRange = Field(description="Measures to fill.")
    key: str = Field(min_length=1, description="Key used to realise the chord.")
    chord: str = Field(default="I", description="Roman-numeral chord to arpeggiate.")
    note_value: str = Field(
        default="eighth", description="Note value name, e.g. 'eighth'."
    )
    direction: Literal["up", "down", "updown"] = Field(
        default="up", description="Arpeggio direction."
    )


class ParallelChordsParams(_Params):
    """Parameters for the parallel-chords technique.

    Attributes:
        voice: Voice slot that receives the chords.
        measure_range: Measures the chords occupy.
        key: Key used to realise the source chord.
        chord: Source Roman-numeral chord.
        step_interval: Diatonic interval moved per repetition.
        repetitions: Number of chord statements.
    """

    voice: str = Field(min_length=1, description="Voice slot receiving the chords.")
    measure_range: MeasureRange = Field(description="Measures the chords occupy.")
    key: str = Field(min_length=1, description="Key used to realise the source chord.")
    chord: str = Field(default="I", description="Source Roman-numeral chord.")
    step_interval: int = Field(
        default=2, description="Diatonic interval moved per repetition."
    )
    repetitions: int = Field(
        default=4, ge=1, le=16, description="Number of chord statements."
    )


class PlaningParams(_Params):
    """Parameters for the parallel planing technique.

    Attributes:
        voice: Voice slot that receives the stacked chords.
        measure_range: Measures the chords occupy.
        key: Key used to realise the source chord.
        chord_size: Whether to stack a triad or a seventh chord.
        step: Diatonic step moved per measure.
    """

    voice: str = Field(
        min_length=1, description="Voice slot receiving the stacked chords."
    )
    measure_range: MeasureRange = Field(description="Measures the chords occupy.")
    key: str = Field(min_length=1, description="Key used to realise the source chord.")
    chord_size: Literal["triad", "seventh"] = Field(
        default="triad", description="Chord size."
    )
    step: int = Field(default=1, description="Diatonic step moved per measure.")


class ChromaticModulationParams(_Params):
    """Parameters for the chromatic modulation technique.

    Attributes:
        start_key: Key to leave.
        target_key: Key to reach.
        measure_range: Measures the modulation occupies.
    """

    start_key: str = Field(min_length=1, description="Key to leave.")
    target_key: str = Field(min_length=1, description="Key to reach.")
    measure_range: MeasureRange = Field(description="Measures the modulation occupies.")


def _default_extensions() -> list[Literal["7", "9", "11", "13"]]:
    """Return the default chord extensions.

    Returns:
        The seventh and ninth degrees.
    """
    return ["7", "9"]


class ExtendedHarmonyParams(_Params):
    """Parameters for the extended-harmony technique.

    Attributes:
        voice: Voice slot that receives the extension chord.
        measure_range: Measures to extend.
        key: Key used to resolve the chord root.
        extensions: Extension degrees to stack (``"7"``, ``"9"``, ``"11"``, ``"13"``).
    """

    voice: str = Field(
        min_length=1, description="Voice slot receiving the extension chord."
    )
    measure_range: MeasureRange = Field(description="Measures to extend.")
    key: str = Field(min_length=1, description="Key used to resolve the chord root.")
    extensions: list[Literal["7", "9", "11", "13"]] = Field(
        default_factory=_default_extensions,
        min_length=1,
        description="Extension degrees stacked above the root.",
    )


class ColorChordParams(_Params):
    """Parameters for the colour-chord technique.

    Attributes:
        voice: Voice slot that receives the colour tone.
        measure_range: Measures to colour.
        key: Key used to resolve the chord root.
        color: Colour degree such as ``"6"``, ``"9"``, ``"#11"`` or ``"b13"``.
    """

    voice: str = Field(
        min_length=1, description="Voice slot receiving the colour tone."
    )
    measure_range: MeasureRange = Field(description="Measures to colour.")
    key: str = Field(min_length=1, description="Key used to resolve the chord root.")
    color: str = Field(
        default="9", description="Colour degree such as '6', '9', '#11', 'b13'."
    )


class ModalHarmonyParams(_Params):
    """Parameters for the modal-harmony technique.

    Attributes:
        key: Home key.
        mode: Church mode name.
        measure_range: Measures to harmonise.
        chord_sequence: Optional chord figures; a mode default is used when empty.
    """

    key: str = Field(min_length=1, description="Home key.")
    mode: Literal["ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian"] = (
        Field(default="ionian", description="Church mode.")
    )
    measure_range: MeasureRange = Field(description="Measures to harmonise.")
    chord_sequence: list[str] = Field(
        default_factory=list,
        description="Optional chord figures; a mode default is used when empty.",
    )


class WholeToneParams(_Params):
    """Parameters for the whole-tone technique.

    Attributes:
        voice: Voice slot that receives the scale.
        measure_range: Measures to fill.
        root: Root pitch of the whole-tone scale.
        direction: Scale direction.
        note_value: Note value name.
    """

    voice: str = Field(min_length=1, description="Voice slot receiving the scale.")
    measure_range: MeasureRange = Field(description="Measures to fill.")
    root: str = Field(default="C4", description='Root pitch such as "C4".')
    direction: Literal["up", "down"] = Field(
        default="up", description="Scale direction."
    )
    note_value: str = Field(
        default="eighth", description="Note value name, e.g. 'eighth'."
    )


class RubatoParams(_Params):
    """Parameters for the rubato technique.

    Attributes:
        measure_range: Measures to breathe.
        amount: Agogic amount (0-0.9); the downbeat is stretched and the rest compressed.
    """

    measure_range: MeasureRange = Field(description="Measures to breathe.")
    amount: float = Field(
        default=0.2,
        gt=0,
        lt=0.9,
        description="Agogic amount; the downbeat is stretched.",
    )


class FreeVoiceLeadingParams(_Params):
    """Parameters for the free-voice-leading rule exemption.

    Attributes:
        voice: Voice whose measures are exempted.
        measure_range: Measures where the exemption applies.
        reason: Musical justification; required to discourage lazy use.
    """

    voice: str = Field(min_length=1, description="Voice whose measures are exempted.")
    measure_range: MeasureRange = Field(
        description="Measures where the exemption applies."
    )
    reason: str = Field(
        min_length=4,
        description=(
            "Musical justification for waiving the voice-leading rules here. Only request "
            "this when free voice leading genuinely improves the music."
        ),
    )


class SubmitThemeParams(_Params):
    """Parameters for the theme submission tool.

    Attributes:
        musicxml: A MusicXML fragment containing only the theme melody.
        key: The key and mode chosen by the composer; empty keeps the current key.
        voice: Target voice slot; empty means the fragment's first part.
        instrument: Instrument to assign to the target voice slot.
    """

    musicxml: str = Field(min_length=1, description="MusicXML of the theme melody.")
    key: str = Field(
        default="",
        description=(
            "Key and mode chosen by the composer, e.g. 'C' (major) or 'a' (minor); "
            "empty keeps the current key."
        ),
    )
    voice: str = Field(
        default="",
        description="Target voice slot such as 'violin1'; empty uses the fragment's first part.",
    )
    instrument: str = Field(
        min_length=1,
        description=(
            "Required. The instrument for the target part, e.g. 'Violin', 'Flute', "
            "'Cello', 'Piano'. It is never inferred from the voice name."
        ),
    )


class AddPartParams(_Params):
    """Parameters for the add-part tool.

    Attributes:
        voice: Voice slot name for the new part.
        instrument: Instrument to assign to the new part (required).
    """

    voice: str = Field(min_length=1, description="Voice slot name for the new part.")
    instrument: str = Field(
        min_length=1,
        description=(
            "Required. The instrument for the new part, e.g. 'Violin', 'Flute', "
            "'Cello', 'Piano'. It is never inferred from the voice name."
        ),
    )


class RemovePartParams(_Params):
    """Parameters for the remove-part tool.

    Attributes:
        voice: Voice slot to remove.
    """

    voice: str = Field(min_length=1, description="Voice slot name to remove.")


class SetTempoParams(_Params):
    """Parameters for the set-tempo tool.

    Attributes:
        bpm: New tempo in quarter notes per minute.
    """

    bpm: int = Field(
        ge=20, le=300, description="New tempo in quarter notes per minute (20-300)."
    )


class SetTimeSignatureParams(_Params):
    """Parameters for the set-time-signature tool.

    Attributes:
        time_signature: New time signature such as ``"3/4"`` or ``"6/8"``.
    """

    time_signature: str = Field(
        min_length=3,
        description="New time signature for the whole movement, e.g. '3/4', '6/8'.",
    )

    @field_validator("time_signature")
    @classmethod
    def _validate_meter(cls, value: str) -> str:
        """Validate the ``numerator/denominator`` shape.

        Args:
            value: Raw time-signature string.

        Returns:
            The stripped time signature.

        Raises:
            ValueError: When the value is not of the form ``"n/d"``.
        """
        text = value.strip()
        numerator, _, denominator = text.partition("/")
        if not numerator.isdigit() or not denominator.isdigit():
            raise ValueError("time signature must look like '3/4'")
        if int(numerator) < 1 or int(denominator) < 1:
            raise ValueError("time signature parts must be positive")
        return text


class AnnotateParams(_Params):
    """Parameters for the annotate tool.

    Attributes:
        measure: One-based measure number to annotate.
        voice: Voice slot to annotate.
        mark: The kind of expressive mark.
        value: Mark-specific value.
    """

    measure: int = Field(ge=1, description="One-based measure number to annotate.")
    voice: str = Field(min_length=1, description="Voice slot to annotate.")
    mark: Literal[
        "dynamic",
        "text",
        "crescendo",
        "diminuendo",
        "accent",
        "tenuto",
        "staccato",
        "slur",
        "pedal",
        "tempo",
    ] = Field(description="The kind of expressive mark.")
    value: str = Field(
        default="",
        description=(
            "For 'dynamic' a name like 'pp'/'f'; for 'text' free text such as "
            "'dolce'; for 'tempo' the BPM as a number; ignored for the rest."
        ),
    )


class SetTitleParams(_Params):
    """Parameters for the set-title tool.

    Attributes:
        title: The work title chosen by the composer.
    """

    title: str = Field(
        min_length=1, description="A concise, fitting title for the work."
    )


class EditParams(_Params):
    """Parameters for the measure-edit tool.

    Attributes:
        measure: One-based measure number to replace.
        voice: Voice slot whose measure is replaced.
        musicxml: MusicXML fragment whose first melody replaces the measure;
            empty clears the measure.
    """

    measure: int = Field(ge=1, description="One-based measure number to replace.")
    voice: str = Field(min_length=1, description="Voice slot to replace.")
    musicxml: str = Field(
        default="",
        description="MusicXML fragment for the measure; empty clears the measure.",
    )


class InsertMeasureParams(_Params):
    """Parameters for the insert-measure tool.

    Attributes:
        measure: One-based position of the new measure.
        voice: Voice that receives the optional fragment.
        musicxml: MusicXML fragment filling the new measure.
    """

    measure: int = Field(ge=1, description="One-based position of the new measure.")
    voice: str = Field(default="", description="Voice that receives the fragment.")
    musicxml: str = Field(
        default="", description="MusicXML fragment for the new measure."
    )


class DeleteMeasureParams(_Params):
    """Parameters for the delete-measure tool.

    Attributes:
        measure: One-based measure number to remove.
    """

    measure: int = Field(ge=1, description="One-based measure number to remove.")


class AddMovementParams(_Params):
    """Parameters for the add-movement planning tool.

    Attributes:
        name: Optional display name.
    """

    name: str = Field(default="", description="Optional display name.")


class SetMovementPromptParams(_Params):
    """Parameters for the set-movement-prompt planning tool.

    Attributes:
        movement: Movement number.
        prompt: The creation requirement for this movement.
    """

    movement: int = Field(ge=1, description="Movement number (1-based).")
    prompt: str = Field(min_length=1, description="The concrete creation requirement.")
