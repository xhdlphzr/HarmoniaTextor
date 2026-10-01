# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""High-level editing operations over a :class:`music21.stream.Score`.

The editor hides music21 measure/offset bookkeeping from the technique packs so
that every technique can focus on musical transformation.
"""

from __future__ import annotations

from collections.abc import Callable

from music21 import articulations, chord, dynamics, expressions, meter, spanner, stream, tempo
from music21 import key as m21key
from music21 import note as m21note
from music21 import pitch as m21pitch

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.score.io import make_instrument, new_part

__all__ = ["ScoreEditor"]

_ARTICULATIONS: dict[str, type[articulations.Articulation]] = {
    "accent": articulations.Accent,
    "tenuto": articulations.Tenuto,
    "staccato": articulations.Staccato,
}

_EPSILON = 1e-6
_MIN_SLUR_NOTES = 2


class ScoreEditor:
    """Mutable view over a score for deterministic symbolic editing.

    Attributes:
        score: The wrapped score.
    """

    def __init__(
        self,
        score: stream.Score,
        *,
        key: str | None = None,
        time_signature: str | None = None,
        tempo_bpm: int | None = None,
    ) -> None:
        """Initialise the editor.

        Args:
            score: Score to wrap and mutate in place.
            key: Key name used when a new voice is created; derived from the
                score when omitted.
            time_signature: Time signature for new voices; derived when omitted.
            tempo_bpm: Tempo for new voices; derived when omitted.
        """
        self.score = score
        self._key = key
        self._time_signature = time_signature
        self._tempo_bpm = tempo_bpm

    def default_key(self) -> str:
        """Return the key used for newly created voices.

        Returns:
            An explicit key when configured, else the score's first key, else
            ``"C"``.
        """
        if self._key:
            return self._key
        found = list(self.score.recurse().getElementsByClass(m21key.Key))
        return str(found[0].tonic.name) if found else "C"

    def default_time_signature(self) -> str:
        """Return the time signature used for newly created voices.

        Returns:
            An explicit signature when configured, else the score's first, else
            ``"4/4"``.
        """
        if self._time_signature:
            return self._time_signature
        found = list(self.score.recurse().getElementsByClass(meter.TimeSignature))
        return str(found[0].ratioString) if found else "4/4"

    def default_tempo(self) -> int:
        """Return the tempo used for newly created voices.

        Returns:
            An explicit tempo when configured, else the score's first, else
            ``96``.
        """
        if self._tempo_bpm:
            return self._tempo_bpm
        found = list(self.score.recurse().getElementsByClass(tempo.MetronomeMark))
        if found and found[0].number is not None:
            return int(found[0].number)
        return 96

    def _ensure_attributes(self, part: stream.Part) -> None:
        """Give a part a first measure carrying key, meter and tempo.

        Args:
            part: Part to complete in place.
        """
        measure = part.measure(1)
        if measure is None:
            measure = stream.Measure(number=1)
            part.insert(0.0, measure)
        if not list(measure.getElementsByClass(meter.TimeSignature)):
            measure.insert(0.0, meter.TimeSignature(self.default_time_signature()))
        if not list(measure.getElementsByClass(m21key.Key)):
            measure.insert(0.0, m21key.Key(self.default_key()))
        if not list(measure.getElementsByClass(tempo.MetronomeMark)):
            measure.insert(0.0, tempo.MetronomeMark(number=self.default_tempo()))

    def bar_length(self) -> float:
        """Return the length of one measure in quarter notes.

        Returns:
            The bar length, defaulting to four quarter notes when no time
            signature is present.
        """
        signatures = list(self.score.recurse().getElementsByClass("TimeSignature"))
        if not signatures:
            return 4.0
        return float(signatures[0].barDuration.quarterLength)

    def voice_names(self) -> list[str]:
        """Return the voice names of all parts, top to bottom.

        Returns:
            A list of voice slot names.
        """
        return [str(part.id or part.partName) for part in self.score.parts]

    def get_part(self, voice: str, *, create: bool = False) -> stream.Part | None:
        """Look up a part by voice slot.

        Args:
            voice: Voice slot name.
            create: Whether to create the part when it is missing.

        Returns:
            The part, or ``None`` when absent and ``create`` is false.
        """
        for part in self.score.parts:
            if str(part.id or part.partName) == voice:
                return part
        if not create:
            return None
        part = new_part(voice)
        self._ensure_attributes(part)
        self.score.insert(0.0, part)
        return part

    def set_instrument(self, voice: str, instrument_name: str) -> None:
        """Assign an instrument to a voice slot.

        Args:
            voice: Voice slot to retarget; created when missing.
            instrument_name: Instrument name such as ``"Violin"``.
        """
        part = self.get_part(voice, create=True)
        assert part is not None
        for existing in list(part.getElementsByClass("Instrument")):
            part.remove(existing)
        part.insert(0.0, make_instrument(instrument_name))

    def set_key(self, key_name: str) -> None:
        """Retune every key signature in the score.

        Each existing key is replaced with a fresh one: mutating a ``Key`` in
        place leaves its cached ``sharps`` stale, so the change would never reach
        the exported MusicXML.

        Args:
            key_name: music21-compatible key name, e.g. ``"C"`` or ``"a"``.
        """
        for element in list(self.score.recurse().getElementsByClass(m21key.Key)):
            container = element.activeSite
            assert container is not None
            offset = element.offset
            container.remove(element)
            container.insert(offset, m21key.Key(key_name))

    def set_tempo(self, bpm: int) -> None:
        """Retune every tempo marking in the score.

        Each existing marking is replaced with a fresh one so the new tempo is
        what actually reaches the exported MusicXML and audio.

        Args:
            bpm: New tempo in quarter notes per minute.
        """
        self._tempo_bpm = bpm
        marks = list(self.score.recurse().getElementsByClass(tempo.MetronomeMark))
        for element in marks:
            container = element.activeSite
            assert container is not None
            offset = element.offset
            container.remove(element)
            container.insert(offset, tempo.MetronomeMark(number=bpm))
        if not marks and self.score.parts:
            self._ensure_attributes(self.score.parts[0])

    def remove_part(self, voice: str) -> bool:
        """Remove a voice from the score.

        Args:
            voice: Voice slot name to remove.

        Returns:
            ``True`` when a part was removed.
        """
        for part in list(self.score.parts):
            if str(part.id or part.partName) == voice:
                self.score.remove(part)
                return True
        return False

    def annotate(  # noqa: PLR0912
        self, voice: str, measure: int, mark: str, value: str = ""
    ) -> bool:
        """Add an expressive mark to one measure of a voice.

        Supported marks are ``dynamic`` (value like ``"f"``), ``text`` (an
        expression such as ``"dolce"``), ``crescendo``, ``diminuendo``,
        ``accent``, ``tenuto``, ``staccato``, ``slur``, ``pedal`` and ``tempo``
        (value is the BPM, for a mid-piece tempo change).

        Args:
            voice: Voice slot to annotate.
            measure: One-based measure number.
            mark: The mark kind.
            value: Mark-specific value.

        Returns:
            ``True`` when the mark was applied.
        """
        part = self.get_part(voice)
        if part is None:
            return False
        target = part.measure(measure)
        if target is None:
            return False
        notes = [item for item in target.notes if isinstance(item, m21note.Note)]
        if mark == "dynamic":
            target.insert(0.0, dynamics.Dynamic(value or "mf"))  # type: ignore[no-untyped-call]
        elif mark == "text":
            target.insert(
                0.0,
                expressions.TextExpression(value or "dolce"),  # type: ignore[no-untyped-call]
            )
        elif mark in {"crescendo", "diminuendo"}:
            wedge = (
                dynamics.Crescendo()  # type: ignore[no-untyped-call]
                if mark == "crescendo"
                else dynamics.Diminuendo()  # type: ignore[no-untyped-call]
            )
            target.insert(0.0, wedge)
            if notes:
                wedge.addSpannedElements(notes[0], notes[-1])
        elif mark in _ARTICULATIONS:
            for item in notes:
                item.articulations.append(_ARTICULATIONS[mark]())
        elif mark == "slur":
            if len(notes) < _MIN_SLUR_NOTES:
                return False
            slur = spanner.Slur()  # type: ignore[no-untyped-call]
            target.insert(0.0, slur)
            slur.addSpannedElements(notes[0], notes[-1])
        elif mark == "pedal":
            pedal = expressions.PedalMark()
            pedal.pedalForm = expressions.PedalForm.Symbol
            pedal.pedalType = expressions.PedalType.Sustain
            target.insert(0.0, pedal)
            if notes:
                pedal.addSpannedElements(notes[0], notes[-1])
        elif mark == "tempo":
            try:
                bpm = int(value)
            except ValueError:
                return False
            target.insert(0.0, tempo.MetronomeMark(number=bpm))
        else:
            return False
        return True

    def _ensure_measure(self, part: stream.Part, number: int) -> stream.Measure:
        """Return the measure with ``number``, creating intervening measures.

        Args:
            part: Part to modify.
            number: One-based measure number.

        Returns:
            The requested measure.
        """
        bar = self.bar_length()
        measures = list(part.getElementsByClass(stream.Measure))
        last = measures[-1].number if measures else 0
        while last < number:
            last += 1
            measure = stream.Measure(number=last)
            part.insert((last - 1) * bar, measure)
        found = part.measure(number)
        assert found is not None
        return found

    def clear_measure_range(self, voice: str, start: int, end: int) -> None:
        """Remove all sounding and resting material from a measure range.

        Args:
            voice: Voice slot to clear.
            start: First measure (one-based).
            end: Last measure (one-based, inclusive).
        """
        part = self.get_part(voice, create=True)
        assert part is not None
        for measure in list(part.getElementsByClass(stream.Measure)):
            if start <= measure.number <= end:
                for element in list(measure.notesAndRests):
                    measure.remove(element)

    def _reflow(self, part: stream.Part, measures: list[stream.Measure]) -> None:
        """Rebuild a part's measures sequentially from one.

        Args:
            part: Part to rebuild in place.
            measures: Measures in their new order.
        """
        bar = self.bar_length()
        for measure in list(part.getElementsByClass(stream.Measure)):
            part.remove(measure)
        for index, measure in enumerate(measures, start=1):
            measure.number = index
            part.insert((index - 1) * bar, measure)
        self._ensure_attributes(part)

    def insert_measure(self, number: int) -> None:
        """Insert an empty measure at a position in every voice.

        Later measures shift one position later.

        Args:
            number: One-based position of the new measure.
        """
        bar = self.bar_length()
        for part in self.score.parts:
            measures = list(part.getElementsByClass(stream.Measure))
            index = min(max(number - 1, 0), len(measures))
            measure = stream.Measure(number=index + 1)
            measure.insert(0.0, m21note.Rest(quarterLength=bar))
            measures.insert(index, measure)
            self._reflow(part, measures)

    def delete_measure(self, number: int) -> bool:
        """Delete a measure from every voice.

        Later measures shift one position earlier.

        Args:
            number: One-based measure number to remove.

        Returns:
            ``True`` when a measure was removed.
        """
        removed = False
        for part in self.score.parts:
            measures = list(part.getElementsByClass(stream.Measure))
            if 1 <= number <= len(measures):
                del measures[number - 1]
                self._reflow(part, measures)
                removed = True
        return removed

    def place_note(
        self,
        voice: str,
        measure: int,
        offset: float,
        pitch: str,
        quarter_length: float,
    ) -> None:
        """Insert a single note, replacing any note at the same offset.

        Args:
            voice: Voice slot.
            measure: One-based measure number.
            offset: Offset within the measure in quarter notes.
            pitch: Scientific pitch name.
            quarter_length: Duration in quarter notes.
        """
        part = self.get_part(voice, create=True)
        assert part is not None
        target = self._ensure_measure(part, measure)
        for existing in list(target.notesAndRests):
            if abs(existing.offset - offset) < _EPSILON:
                target.remove(existing)
        target.insert(offset, m21note.Note(pitch, quarterLength=quarter_length))

    def place_chord(
        self,
        voice: str,
        measure: int,
        offset: float,
        pitches: list[str],
        quarter_length: float,
    ) -> None:
        """Insert a chord, replacing any note at the same offset.

        Args:
            voice: Voice slot.
            measure: One-based measure number.
            offset: Offset within the measure in quarter notes.
            pitches: Chord pitches.
            quarter_length: Duration in quarter notes.
        """
        part = self.get_part(voice, create=True)
        assert part is not None
        target = self._ensure_measure(part, measure)
        for existing in list(target.notesAndRests):
            if abs(existing.offset - offset) < _EPSILON:
                target.remove(existing)
        target.insert(offset, chord.Chord(pitches, quarterLength=quarter_length))

    def write_line(
        self,
        voice: str,
        start_measure: int,
        notes: list[ThemeNote],
        *,
        start_offset: float = 0.0,
    ) -> None:
        """Write a melody sequentially starting at a measure.

        Args:
            voice: Target voice slot.
            start_measure: First measure of the melody (one-based).
            notes: Notes in performance order.
            start_offset: Offset within the first measure.
        """
        bar = self.bar_length()
        position = (start_measure - 1) * bar + start_offset
        for item in notes:
            measure = int(position // bar) + 1
            offset = position - (measure - 1) * bar
            self.place_note(voice, measure, offset, item.pitch, item.quarter_length)
            position += item.quarter_length

    def map_notes(
        self,
        voice: str,
        start: int,
        end: int,
        transform: Callable[[float, str, float], str],
    ) -> None:
        """Rewrite the pitches of notes in a measure range.

        Args:
            voice: Voice slot to modify.
            start: First measure (one-based).
            end: Last measure (one-based, inclusive).
            transform: Callback mapping ``(offset, pitch, duration)`` to a new
                pitch name.
        """
        part = self.get_part(voice)
        if part is None:
            return
        for measure in part.getElementsByClass(stream.Measure):
            if not start <= measure.number <= end:
                continue
            for element in measure.notes:
                if isinstance(element, m21note.Note):
                    new_pitch = transform(
                        float(element.offset),
                        element.nameWithOctave,
                        float(element.quarterLength),
                    )
                    element.pitch = m21pitch.Pitch(new_pitch)

    def read_line(self, voice: str, start: int, end: int) -> list[ThemeNote]:
        """Read a monophonic line from a measure range.

        Args:
            voice: Voice slot to read.
            start: First measure (one-based).
            end: Last measure (one-based, inclusive).

        Returns:
            The notes found in the range, in performance order.
        """
        part = self.get_part(voice)
        if part is None:
            return []
        result: list[ThemeNote] = []
        for measure in part.getElementsByClass(stream.Measure):
            if not start <= measure.number <= end:
                continue
            for element in measure.notes:
                if isinstance(element, m21note.Note):
                    result.append(
                        ThemeNote(
                            pitch=element.nameWithOctave,
                            quarter_length=float(element.quarterLength),
                        )
                    )
        return result
