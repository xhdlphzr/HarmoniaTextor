# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""A lightweight IPA FOF (formant wave function) singer.

The singer uses FOF synthesis: every glottal pulse excites a damped sinusoidal
wave packet at each vowel formant, and the sum of those overlapping packets
gives the voice its formant structure.  Consonants use noise bursts and
formant-based sonorant models.  The backend is deliberately small, offline and
model-free; a neural backend can implement the same :class:`VocalBackend`
interface later.

Real vowels are never perfectly periodic, so a vibrato that fades in after the
note starts, a deterministic jitter on the pulse train and low-passed breath
noise are mixed into every voiced segment.  Every note is normalised to the same
loudness instead of the same peak, which keeps high and low notes equally
audible, and notes without a recognisable phoneme fall back to a neutral schwa
instead of silence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from harmoniatextor.render.vocal.ipa import (
    VOWELS,
    base_symbol,
    consonant_kind,
    tokenize,
)

__all__ = [
    "SAMPLE_RATE",
    "FOFSinger",
    "FormantSinger",
    "SungNote",
    "VocalBackend",
]

#: Sample rate of the synthesised vocal stems.
SAMPLE_RATE = 44100
_ATTACK = 0.02
_RELEASE = 0.05
_VIBRATO_HZ = 5.0
_VIBRATO_DEPTH = 0.006
#: Seconds before the vibrato reaches full depth (real singers fade it in).
_VIBRATO_ONSET = 0.25
_NOISE_SEED = 0
_BANDWIDTHS = (80.0, 100.0, 140.0)
#: Relative amplitude of the F1/F2/F3 wave packets.  F2 distinguishes /a/ from
#: /i/ and /u/, so it is kept almost as strong as F1 instead of being buried by
#: the glottal roll-off of a plain harmonic series.
_FORMANT_WEIGHTS = (1.0, 0.85, 0.6)
#: FOF attack time: the wave packet rises over this many seconds, then decays.
_FOF_ATTACK = 0.002
#: Maximum tail of one FOF packet; after this the exponential envelope is
#: already inaudible.
_FOF_TAIL = 0.05
#: Deterministic pulse-train jitter, as a fraction of one period.
_PULSE_JITTER = 0.003
#: Target RMS of a voiced segment; every note is normalised to it so the
#: singer does not get louder as the pitch rises.
_VOICED_RMS = 0.2
#: Hard ceiling applied after the per-note gain; leaves headroom for mixing.
_PEAK_CEILING = 0.9
#: Formants of the nasal murmur (m/n/ng), roughly centred but darker than a
#: vowel; using a vowel formant table here made nasals sound like vowels.
_NASAL_FORMANTS = (250.0, 1300.0, 2200.0)
#: Formants of the two common liquid consonants.
_LIQUID_FORMANTS = {
    "l": (360.0, 1300.0, 2700.0),
    "ɭ": (360.0, 1300.0, 2700.0),
    "ɬ": (360.0, 1300.0, 2700.0),
    "r": (400.0, 1200.0, 1600.0),
    "ɹ": (400.0, 1200.0, 1600.0),
    "ɾ": (400.0, 1200.0, 1600.0),
    "ɻ": (400.0, 1200.0, 1600.0),
}
#: Sibilants carry strong high-frequency noise, other fricatives are softer.
_SIBILANTS = frozenset("szʃʒʂʐɕʑ")
#: Voiced fricatives get a quiet voiced murmur behind their noise.
_VOICED_FRICATIVES = frozenset("vzʒðɣʐʑ")
#: Breath noise mixed into every voiced segment (fraction of its RMS).
_BREATH_LEVEL = 0.015
#: Aspiration noise is low-passed around this corner frequency; a real voice
#: has airy energy below the upper formants, not full-band hiss.
_BREATH_CUTOFF_HZ = 1200.0
#: Level of the schwa fallback used for symbols outside the IPA tables.
_FALLBACK_LEVEL = 0.6
_FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SungNote:
    """One sung note handed to a vocal backend.

    Attributes:
        midi: MIDI pitch number.
        onset: Start time in seconds from the start of the score.
        duration: Duration in seconds.
        ipa: IPA pronunciation of the syllable.
        text: Original-language text of the syllable, when known.
        language: Language tag of the syllable, when known.
        syllabic: Syllabic role (``single``/``begin``/``middle``/``end``).
    """

    midi: int
    onset: float
    duration: float
    ipa: str
    text: str = ""
    language: str = ""
    syllabic: str = "single"


class VocalBackend(Protocol):
    """A singing synthesis backend."""

    def synthesize(self, notes: Sequence[SungNote]) -> _FloatArray:
        """Render sung notes to mono samples at :data:`SAMPLE_RATE`."""
        raise NotImplementedError


class FOFSinger:
    """The built-in offline IPA FOF (formant wave function) singer."""

    def synthesize(self, notes: Sequence[SungNote]) -> _FloatArray:
        """Render sung notes to mono samples.

        Args:
            notes: Notes to sing, in performance order.

        Returns:
            Mono float samples at :data:`SAMPLE_RATE`.
        """
        if not notes:
            return np.zeros(0, dtype=np.float64)
        total = (
            int(
                np.ceil(max(note.onset + note.duration for note in notes) * SAMPLE_RATE)
            )
            + 1
        )
        buffer = np.zeros(total, dtype=np.float64)
        for note in notes:
            start = max(0, int(note.onset * SAMPLE_RATE))
            samples = _sing_note(note)
            stop = min(total, start + samples.size)
            if stop > start:
                buffer[start:stop] += samples[: stop - start]
        peak = float(np.max(np.abs(buffer))) if buffer.size else 0.0
        if peak > 1.0:
            buffer = buffer / peak
        return buffer


#: Backwards-compatible alias for the class name used before the FOF rewrite.
FormantSinger = FOFSinger


def _sing_note(note: SungNote) -> _FloatArray:
    """Render one sung note.

    Args:
        note: Note to sing.

    Returns:
        Mono float samples.
    """
    length = int(note.duration * SAMPLE_RATE)
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    phonemes = tokenize(note.ipa) or ["ə"]
    kinds = [consonant_kind(phoneme) for phoneme in phonemes]
    lengths = _allocate(length, kinds)
    pieces: list[_FloatArray] = []
    for phoneme, kind, count in zip(phonemes, kinds, lengths):
        if count <= 0:
            continue
        pieces.append(_phoneme(phoneme, kind, note.midi, count))
    assert pieces
    signal = np.concatenate(pieces)[:length]
    return _envelope(signal)


def _allocate(length: int, kinds: Sequence[str]) -> list[int]:
    """Split a note's samples among its phonemes.

    Consonants take a short fixed share; vowels share the remainder.

    Args:
        length: Total samples.
        kinds: Phoneme kinds.

    Returns:
        Sample counts per phoneme.
    """
    if length <= 0 or not kinds:
        return []
    if len(kinds) == 1:
        return [length]
    lengths = [0] * len(kinds)
    consonants = [index for index, kind in enumerate(kinds) if kind != "vowel"]
    vowels = [index for index, kind in enumerate(kinds) if kind == "vowel"]
    if consonants:
        # Give each consonant a clear share (15%) but never let consonants eat
        # more than 45% of the note; the vowel needs the rest to be identifiable.
        each = max(1, int(length * 0.15))
        total = min(int(length * 0.45), each * len(consonants))
        base = total // len(consonants)
        for index in consonants:
            lengths[index] = base
        lengths[consonants[-1]] += total - base * len(consonants)
    remaining = length - sum(lengths)
    if vowels:
        share = remaining // len(vowels)
        for index in vowels:
            lengths[index] = share
        lengths[vowels[-1]] += remaining - share * len(vowels)
    elif consonants:
        lengths[consonants[-1]] += remaining
    return lengths


def _phoneme(phoneme: str, kind: str, midi: int, count: int) -> _FloatArray:
    """Render one phoneme segment.

    Vowels and sonorants (nasals/liquids) use formant synthesis so their vowel
    colour is audible; obstruents use noise or bursts.  Unknown symbols are
    voiced as a quiet schwa rather than rendered as silence, matching the
    fallback promised by the IPA module.

    Args:
        phoneme: Phoneme token.
        kind: Its kind from :func:`consonant_kind`.
        midi: MIDI pitch of the note.
        count: Number of samples.

    Returns:
        Mono float samples.
    """
    if kind == "vowel":
        return _vowel(phoneme, midi, count)
    if kind == "nasal":
        return _voiced(_NASAL_FORMANTS, midi, count, 0.45)
    if kind == "liquid":
        formants = _LIQUID_FORMANTS.get(base_symbol(phoneme), (390.0, 1400.0, 2400.0))
        return _voiced(formants, midi, count, 0.5)
    if kind == "fricative":
        return _fricative(phoneme, midi, count)
    if kind == "stop":
        return _stop(count)
    if kind == "affricate":
        return _affricate(phoneme, count)
    return _vowel("ə", midi, count) * _FALLBACK_LEVEL


def _pulse_positions(count: int, f0: float, seed: int) -> NDArray[np.int64]:
    """Return deterministic glottal-pulse sample positions for one segment.

    The pulse train follows a vibrato-modulated f0 and a small deterministic
    jitter, which keeps the FOF train from sounding perfectly mechanical.

    Args:
        count: Number of samples.
        f0: Fundamental frequency in Hz.
        seed: Deterministic seed for the jitter.

    Returns:
        Sorted pulse positions in samples.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    vibrato = 1.0 + _VIBRATO_DEPTH * np.sin(
        2.0 * np.pi * _VIBRATO_HZ * time
    ) * np.minimum(1.0, time / _VIBRATO_ONSET)
    phase = np.cumsum(f0 * vibrato) / SAMPLE_RATE
    starts = np.nonzero(np.diff(np.floor(phase)) > 0)[0] + 1
    positions = np.concatenate((np.zeros(1, dtype=np.int64), starts))
    positions = positions[positions < count]
    if positions.size > 1:
        rng = np.random.default_rng(seed)
        period = SAMPLE_RATE / max(f0, 1e-9)
        offsets = (
            rng.uniform(-1.0, 1.0, size=positions.size - 1) * _PULSE_JITTER * period
        )
        positions[1:] = np.clip(positions[1:] + offsets.astype(np.int64), 1, count - 1)
        positions = np.unique(positions)
    return positions


def _voiced(
    formants: tuple[float, float, float],
    midi: int,
    count: int,
    level: float = 1.0,
) -> _FloatArray:
    """Render one voiced segment with FOF (formant wave function) synthesis.

    Each glottal pulse contributes a short damped sinusoid at every formant.
    Summing the overlapping packets produces a natural formant spectrum without
    an explicit harmonic series.

    Args:
        formants: ``(F1, F2, F3)`` centres in Hz.
        midi: MIDI pitch of the note.
        count: Number of samples.
        level: Relative loudness of the segment.

    Returns:
        Mono float samples.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    seed = _NOISE_SEED + midi * 7919 + count + int(level * 1000)
    positions = _pulse_positions(count, f0, seed)
    signal = np.zeros(count, dtype=np.float64)
    kernel_count = min(count, max(1, int(_FOF_TAIL * SAMPLE_RATE)))
    kernel_time = np.arange(kernel_count, dtype=np.float64) / SAMPLE_RATE
    attack = 1.0 - np.exp(-kernel_time / _FOF_ATTACK)
    for formant, bandwidth, weight in zip(formants, _BANDWIDTHS, _FORMANT_WEIGHTS):
        envelope = attack * np.exp(-np.pi * bandwidth * kernel_time) * weight
        kernel = envelope * np.sin(2.0 * np.pi * formant * kernel_time)
        for start in positions:
            available = min(kernel_count, count - int(start))
            if available > 0:
                signal[int(start) : int(start) + available] += kernel[:available]
    rms = float(np.sqrt(np.mean(np.square(signal))))
    signal *= _VOICED_RMS * level / max(rms, 1e-9)
    signal += _breath(
        count,
        _BREATH_LEVEL,
        seed=_NOISE_SEED + midi * 7919 + count + int(level * 1000),
    )
    peak = float(np.max(np.abs(signal)))
    signal *= min(1.0, _PEAK_CEILING / max(peak, 1e-9))
    return signal


def _vowel(phoneme: str, midi: int, count: int) -> _FloatArray:
    """Render a vowel with FOF synthesis.

    The vowel's three formants are excited by a glottal pulse train, with a
    little breath noise on top.  The result is normalised to a fixed RMS so
    every pitch sounds equally loud.

    Args:
        phoneme: Vowel token.
        midi: MIDI pitch of the note.
        count: Number of samples.

    Returns:
        Mono float samples.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    base = base_symbol(phoneme)
    formants = VOWELS.get(base, VOWELS["ə"])
    return _voiced(formants, midi, count)


def _fricative(phoneme: str, midi: int, count: int) -> _FloatArray:
    """Render a fricative as shaped noise, with a murmur when voiced.

    Args:
        phoneme: Fricative token.
        midi: MIDI pitch of the note (used by the voiced murmur).
        count: Number of samples.

    Returns:
        Mono float samples.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    base = base_symbol(phoneme)
    level = 0.3 if base in _SIBILANTS else 0.18
    signal = _noise(count, level, seed=_NOISE_SEED + sum(ord(char) for char in base))
    if base in _VOICED_FRICATIVES:
        signal += _voiced((300.0, 1400.0, 2400.0), midi, count, 0.12)
    return signal


def _affricate(phoneme: str, count: int) -> _FloatArray:
    """Render an affricate as a stop burst followed by fricative noise.

    Args:
        phoneme: Affricate token.
        count: Number of samples.

    Returns:
        Mono float samples.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    signal = np.zeros(count, dtype=np.float64)
    burst = min(count, max(1, int(0.012 * SAMPLE_RATE)))
    seed = _NOISE_SEED + sum(ord(char) for char in base_symbol(phoneme))
    signal[:burst] = _noise(burst, 0.5, seed=seed)
    if count > burst:
        signal[burst:] = _noise(count - burst, 0.25, seed=seed + 1)
    return signal


def _breath(count: int, level: float, *, seed: int = _NOISE_SEED) -> _FloatArray:
    """Render aspiration noise for a voiced segment.

    The noise is low-passed around :data:`_BREATH_CUTOFF_HZ` so it sounds like
    air in a voice rather than full-band hiss.  The seed makes the result
    deterministic while still differing between notes.

    Args:
        count: Number of samples.
        level: Peak level.
        seed: Random seed.

    Returns:
        Mono float samples.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(count))
    freqs = np.fft.rfftfreq(count, 1.0 / SAMPLE_RATE)
    spectrum /= 1.0 + (freqs / _BREATH_CUTOFF_HZ) ** 2
    noise = np.fft.irfft(spectrum, count)
    peak = float(np.max(np.abs(noise)))
    if peak > 0:
        noise = noise / peak * level
    return noise


def _noise(count: int, level: float, *, seed: int = _NOISE_SEED) -> _FloatArray:
    """Render a high-passed noise burst.

    Args:
        count: Number of samples.
        level: Peak level.
        seed: Random seed; differing seeds keep repeated phonemes from being
            bit-identical while the output stays deterministic.

    Returns:
        Mono float samples.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(count)
    noise = np.diff(noise, prepend=0.0)
    peak = float(np.max(np.abs(noise)))
    if peak > 0:
        noise = noise / peak * level
    return noise


def _stop(count: int) -> _FloatArray:
    """Render a stop consonant as a short burst.

    Args:
        count: Number of samples.

    Returns:
        Mono float samples.
    """
    signal = np.zeros(count, dtype=np.float64)
    burst = min(count, max(1, int(0.01 * SAMPLE_RATE)))
    signal[:burst] = _noise(burst, 0.5)
    return signal


def _envelope(signal: _FloatArray) -> _FloatArray:
    """Apply an attack/release envelope to avoid clicks.

    Args:
        signal: Samples to shape.

    Returns:
        The shaped samples.
    """
    attack = min(signal.size, int(_ATTACK * SAMPLE_RATE))
    release = min(signal.size, int(_RELEASE * SAMPLE_RATE))
    envelope = np.ones(signal.size, dtype=np.float64)
    if attack:
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
    if release:
        envelope[-release:] *= np.linspace(1.0, 0.0, release)
    return signal * envelope
