# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for the built-in IPA vocal singer."""

from __future__ import annotations

import numpy as np
from music21 import chord, stream, tempo

from harmoniatextor.domain.models import Syllable, ThemeNote
from harmoniatextor.render.vocal.extract import (
    quarters_to_seconds,
    tempo_map,
    vocal_lines,
)
from harmoniatextor.render.vocal.ipa import (
    VOWELS,
    base_symbol,
    consonant_kind,
    tokenize,
)
from harmoniatextor.render.vocal.singer import (
    SAMPLE_RATE,
    FOFSinger,
    SungNote,
    _affricate,
    _allocate,
    _breath,
    _fricative,
    _noise,
    _phoneme,
    _pulse_positions,
    _voiced,
    _vowel,
)
from harmoniatextor.score.io import from_musicxml, new_score, to_musicxml
from harmoniatextor.score.lyrics import attach_lyrics, clear_lyrics
from harmoniatextor.score.streamops import ScoreEditor


def _rms(samples: np.ndarray) -> float:
    """Return the RMS of a sample buffer.

    Args:
        samples: Mono samples.

    Returns:
        The root mean square, or zero for an empty buffer.
    """
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


class TestIpa:
    """Phoneme tokenisation and classification."""

    def test_tokenize(self) -> None:
        """Combining marks stay with their base and stress marks are dropped."""
        assert tokenize("lʌv") == ["l", "ʌ", "v"]
        assert tokenize("juː") == ["j", "uː"]
        assert tokenize("t͡ʃa") == ["t͡", "ʃ", "a"]
        assert tokenize("ˈlʌv") == ["l", "ʌ", "v"]
        assert tokenize("") == []
        assert tokenize("a  b") == ["a", "b"]

    def test_tokenize_tones_and_diacritics(self) -> None:
        """Tone marks are dropped and tone-bearing letters decompose."""
        assert tokenize("ma˥˩") == ["m", "a"]
        assert tokenize("ma2") == ["m", "a"]
        assert tokenize("nǐ hǎo") == ["n", "i\u030c", "h", "a\u030c", "o"]
        assert tokenize("\u0301a") == ["a"]

    def test_base_symbol(self) -> None:
        """Base symbols strip diacritics, ties and length marks."""
        assert base_symbol("i\u030c") == "i"
        assert base_symbol("uː") == "u"
        assert base_symbol("t͡ʃ") == "tʃ"
        assert base_symbol("") == ""

    def test_consonant_kind(self) -> None:
        """Every phoneme class is recognised, across the extended IPA table."""
        assert consonant_kind("a") == "vowel"
        assert consonant_kind("ɑ") == "vowel"
        assert consonant_kind("i\u030c") == "vowel"
        assert consonant_kind("p") == "stop"
        assert consonant_kind("ɡ") == "stop"
        assert consonant_kind("ʈ") == "stop"
        assert consonant_kind("t͡") == "stop"
        assert consonant_kind("s") == "fricative"
        assert consonant_kind("ʃ") == "fricative"
        assert consonant_kind("ɕ") == "fricative"
        assert consonant_kind("m") == "nasal"
        assert consonant_kind("ɱ") == "nasal"
        assert consonant_kind("l") == "liquid"
        assert consonant_kind("ɻ") == "liquid"
        assert consonant_kind("tʃ") == "affricate"
        assert consonant_kind("ʈʂ") == "affricate"
        assert consonant_kind("☃") == "unknown"
        assert consonant_kind("") == "unknown"

    def test_vowel_table(self) -> None:
        """The vowel table carries three formants per vowel."""
        assert all(len(formants) == 3 for formants in VOWELS.values())


class TestSinger:
    """Formant synthesis."""

    def test_empty(self) -> None:
        """No notes produce no samples."""
        assert FOFSinger().synthesize([]).size == 0

    def test_vowel(self) -> None:
        """A vowel note produces audible, bounded, finite samples."""
        samples = FOFSinger().synthesize([SungNote(60, 0.0, 0.2, "a")])
        assert samples.size > 8000
        assert np.all(np.isfinite(samples))
        assert float(np.max(np.abs(samples))) <= 1.0
        assert float(np.max(np.abs(samples))) > 0.0
        assert _rms(samples) > 0.03

    def test_vowels_have_distinct_spectra(self) -> None:
        """Different vowels expose different upper formants, not just pitch.

        The singer is a formant synthesiser, so open/close vowels must be
        spectrally different; otherwise the voice sounds like a pure pitch
        generator with no vowel articulation.
        """
        centroids: dict[str, float] = {}
        for ipa in ("a", "i", "u", "ɛ"):
            samples = FOFSinger().synthesize([SungNote(60, 0.0, 0.5, ipa)])
            spectrum = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)
            centroids[ipa] = float((spectrum * freqs).sum() / spectrum.sum())
        assert max(centroids.values()) - min(centroids.values()) > 100.0

    def test_duration(self) -> None:
        """The buffer covers the last note end at the documented sample rate."""
        samples = FOFSinger().synthesize([SungNote(60, 0.25, 0.5, "a")])
        assert SAMPLE_RATE == 44100
        assert samples.size == int(np.ceil(0.75 * SAMPLE_RATE)) + 1

    def test_offset_and_silence(self) -> None:
        """A note starting later leaves a silent lead-in."""
        samples = FOFSinger().synthesize([SungNote(60, 0.1, 0.1, "a")])
        assert float(np.max(np.abs(samples[: int(0.05 * 44100)]))) == 0.0
        assert float(np.max(np.abs(samples))) > 0.0

    def test_phoneme_kinds(self) -> None:
        """Stops, fricatives, nasals, liquids, affricates and unknown symbols."""
        notes = [
            SungNote(60, 0.0, 0.2, "pam"),
            SungNote(60, 0.3, 0.2, "sa"),
            SungNote(60, 0.6, 0.2, "la"),
            SungNote(60, 0.9, 0.2, "tʃa"),
            SungNote(60, 1.2, 0.2, "☃a"),
            SungNote(60, 1.5, 0.2, "aɪ"),
            SungNote(60, 1.8, 0.2, "st"),
            SungNote(60, 2.1, 0.2, ""),
            SungNote(60, 2.4, 0.2, "p"),
        ]
        samples = FOFSinger().synthesize(notes)
        assert samples.size > 0
        assert np.all(np.isfinite(samples))
        assert float(np.max(np.abs(samples))) <= 1.0

    def test_unmapped_phonemes_are_voiced(self) -> None:
        """Symbols outside the tables sing a fallback vowel, never silence."""
        for ipa in ("爱", "ɻ", "☃", "χ", "?"):
            samples = FOFSinger().synthesize([SungNote(60, 0.0, 0.3, ipa)])
            assert samples.size > 0
            assert _rms(samples) > 0.01, ipa
            assert np.all(np.isfinite(samples))

    def test_loudness_is_pitch_independent(self) -> None:
        """Notes across the vocal range share a similar RMS."""
        low = _rms(FOFSinger().synthesize([SungNote(36, 0.0, 0.5, "a")]))
        high = _rms(FOFSinger().synthesize([SungNote(84, 0.0, 0.5, "a")]))
        assert low > 0.05
        assert 0.6 <= low / high <= 1.7

    def test_deterministic(self) -> None:
        """The same notes render bit-identically on every call."""
        notes = [SungNote(60, 0.0, 0.25, "pa"), SungNote(64, 0.25, 0.25, "ʂɤ")]
        first = FOFSinger().synthesize(notes)
        second = FOFSinger().synthesize(notes)
        assert np.array_equal(first, second)

    def test_zero_duration(self) -> None:
        """A zero-length note is ignored."""
        samples = FOFSinger().synthesize([SungNote(60, 0.0, 0.0, "a")])
        assert float(np.max(np.abs(samples))) == 0.0

    def test_normalises_loud_overlap(self) -> None:
        """Overlapping notes are normalised to stay within ``[-1, 1]``."""
        notes = [SungNote(60, 0.0, 0.1, "a") for _ in range(100)]
        samples = FOFSinger().synthesize(notes)
        assert float(np.max(np.abs(samples))) <= 1.0
        assert np.all(np.isfinite(samples))

    def test_tiny_note_skips_empty_segment(self) -> None:
        """A one-sample note skips the vowel segment it cannot fit."""
        samples = FOFSinger().synthesize([SungNote(60, 0.0, 1.0 / 44100, "pa")])
        assert samples.size > 0

    def test_internal_guards(self) -> None:
        """Allocation, phoneme and signal helpers handle empty inputs."""
        assert _allocate(0, ["a"]) == []
        assert _allocate(10, []) == []
        assert _phoneme("tʃ", "affricate", 60, 10).size == 10
        assert _phoneme("tʃ", "affricate", 60, 1000).size == 1000
        assert _phoneme("☃", "unknown", 60, 10).size == 10
        assert float(np.max(np.abs(_phoneme("☃", "unknown", 60, 10)))) > 0.0
        assert _vowel("a", 60, 0).size == 0
        assert _voiced((500.0, 1500.0, 2500.0), 60, 0).size == 0
        assert _pulse_positions(0, 440.0, 0).size == 0
        assert _pulse_positions(1, 440.0, 0).tolist() == [0]
        assert _pulse_positions(1000, 440.0, 0).size > 1
        assert _fricative("s", 60, 0).size == 0
        assert _fricative("v", 60, 100).size == 100
        assert float(np.max(np.abs(_fricative("v", 60, 100)))) > 0.0
        assert _affricate("tʃ", 0).size == 0
        assert _noise(0, 0.1).size == 0
        assert _breath(0, 0.1).size == 0


def _sung_score() -> stream.Score:
    """Build a two-measure soprano score with lyrics and a tempo change.

    Returns:
        The score.
    """
    score = new_score(key="C", time_signature="4/4", tempo_bpm=120, voices=["soprano"])
    editor = ScoreEditor(score)
    editor.write_line("soprano", 1, [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
    editor.write_line("soprano", 2, [ThemeNote("E5", 1.0)])
    part = score.parts[0]
    attach_lyrics(
        part,
        1,
        [Syllable("love", "lʌv", "single", "en"), Syllable("you", "juː")],
    )
    attach_lyrics(part, 2, [Syllable("hold", "", "single", "en")])
    measure = part.measure(2)
    assert measure is not None
    measure.insert(0.0, tempo.MetronomeMark(number=60))
    return score


class TestExtract:
    """Tempo mapping and sung-note extraction."""

    def test_tempo_map_default(self) -> None:
        """A score without tempo marks uses the default tempo."""
        assert tempo_map(stream.Score()) == [(0.0, 120.0)]

    def test_tempo_map_starts_late(self) -> None:
        """A late first tempo mark is preceded by a zero-offset segment."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=120, voices=["soprano"]
        )
        part = score.parts[0]
        for mark in list(part.recurse().getElementsByClass(tempo.MetronomeMark)):
            site = mark.activeSite
            assert site is not None
            site.remove(mark)
        measure = part.measure(1)
        assert measure is not None
        measure.insert(2.0, tempo.MetronomeMark(number=90))
        assert tempo_map(score)[0] == (0.0, 90.0)

    def test_quarters_to_seconds(self) -> None:
        """Tempo changes switch the conversion rate."""
        marks = [(0.0, 120.0), (4.0, 60.0)]
        assert quarters_to_seconds(0.0, marks) == 0.0
        assert quarters_to_seconds(4.0, marks) == 2.0
        assert quarters_to_seconds(8.0, marks) == 6.0

    def test_vocal_lines(self) -> None:
        """Notes carry timing, pitch and IPA; a melisma reuses the vowel."""
        lines = vocal_lines(_sung_score(), {"soprano": "en"})
        notes = lines["soprano"]
        assert [note.midi for note in notes] == [72, 74, 76]
        assert notes[0].onset == 0.0
        assert notes[0].duration == 0.5
        assert notes[1].onset == 0.5
        assert notes[2].onset == 2.0
        assert notes[2].duration == 1.0
        assert notes[2].ipa == "juː"

    def test_vocal_lines_after_roundtrip(self) -> None:
        """A real score survives MusicXML serialisation and still sings."""
        score = from_musicxml(to_musicxml(_sung_score()))
        notes = vocal_lines(score, {"soprano": "en"})["soprano"]
        assert [note.midi for note in notes] == [72, 74, 76]
        assert [round(note.onset, 3) for note in notes] == [0.0, 0.5, 2.0]
        assert [round(note.duration, 3) for note in notes] == [0.5, 0.5, 1.0]
        assert notes[0].ipa == "lʌv"

    def test_vocal_lines_fill_cleared_lyrics(self) -> None:
        """A note whose lyric was cleared still sings the previous syllable."""
        score = _sung_score()
        clear_lyrics(score.parts[0], 2)
        notes = vocal_lines(score, {"soprano": "en"})["soprano"]
        assert [note.midi for note in notes] == [72, 74, 76]
        assert notes[2].ipa == "juː"

    def test_vocal_lines_without_lyrics(self) -> None:
        """A marked vocal part without lyrics still sings a neutral schwa.

        Otherwise removing it from the instrumental render would leave that
        voice completely silent in the exported audio.
        """
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=120, voices=["soprano"]
        )
        ScoreEditor(score).write_line("soprano", 1, [ThemeNote("C5", 1.0)])
        notes = vocal_lines(score, {"soprano": "en"})["soprano"]
        assert len(notes) == 1
        assert notes[0].midi == 72
        assert notes[0].duration == 0.5
        assert notes[0].ipa == "ə"

    def test_vocal_lines_ignore_chords(self) -> None:
        """Chords do not count as singable notes in a monophonic line."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=120, voices=["soprano"]
        )
        measure = score.parts[0].measure(1)
        assert measure is not None
        measure.insert(0.0, chord.Chord(["C4", "E4"]))
        assert vocal_lines(score, {"soprano": "en"}) == {}

    def test_vocal_lines_missing_voice(self) -> None:
        """A marked voice without a part is skipped."""
        assert vocal_lines(_sung_score(), {"alto": "en"}) == {}
