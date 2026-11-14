# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Audio synthesis pipeline built on FluidSynth and ffmpeg.

The pipeline writes a temporary MIDI file, renders it to WAV with FluidSynth,
then transcodes the WAV to the requested compressed format with ffmpeg.
"""

from __future__ import annotations

import copy
import math
import shutil
import subprocess
import tempfile
import threading
import wave
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import numpy as np
from music21 import stream
from numpy.typing import NDArray

from harmoniatextor.render.expression import realize_expressions
from harmoniatextor.render.features import Features
from harmoniatextor.render.vocal.diffsinger import AutoVocalBackend
from harmoniatextor.render.vocal.extract import vocal_lines
from harmoniatextor.render.vocal.singer import (
    SAMPLE_RATE,
    SungNote,
    VocalBackend,
)

__all__ = ["FeatureUnavailableError", "synthesize_audio"]

_CODECS: dict[str, str] = {"m4a": "aac", "mp3": "libmp3lame"}

#: FluidSynth master gain.  The default (0.2) renders far too quietly, so the
#: synth is driven hotter before the loudness pass below.  A slight attenuation
#: leaves a little room for the sung voice, which often shares the same register.
_FLUIDSYNTH_GAIN = "0.85"
#: ffmpeg loudness normalisation.  Raises quiet scores to a loud, consistent
#: level and true-peak limits the result so the boosted gain cannot clip.
_LOUDNESS_FILTER = "loudnorm=I=-14:LRA=11:TP=-1.5"
#: Target RMS of the sounding part of a vocal stem.  DiffSinger stems are often
#: much quieter than the FOF singer's; normalising only downwards left them
#: buried under the strings.
_VOCAL_RMS = 0.2
#: Never amplify a very quiet stem by more than this factor.
_MAX_VOCAL_GAIN = 8.0
#: Samples below this peak are treated as digital silence and dropped.
_DIGITAL_SILENCE = 1e-6
#: Hard peak ceiling for a vocal stem after its level fit.
_VOCAL_PEAK = 0.98

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class FeatureUnavailableError(RuntimeError):
    """Raised when a required audio backend binary is missing."""


def _lock_for(path: Path) -> threading.Lock:
    """Return a process-wide lock serialising exports of one output file.

    The browser may request the same audio file more than once at a time (for
    example the auto-playing ``<audio>`` element and a manual export).  Without
    serialisation both runs would write the same intermediates and output,
    which on Windows raises ``PermissionError`` when one run deletes a file the
    other still holds.

    Args:
        path: Output audio path.

    Returns:
        A lock dedicated to that path.
    """
    key = str(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _apply_section_sizes(
    score: stream.Score, counts: dict[str, int] | None
) -> stream.Score:
    """Expand each voice into as many players as its section size.

    A score with one part per voice sounds like a single player.  For a section
    of several players the part is copied once per player and each copy is
    quietened so the section is louder and thicker without clipping.  The work
    is done on the throw-away score used for audio rendering and never reaches
    the stored notation.

    Args:
        score: The score to render.
        counts: Voice-to-player-count mapping; omitted voices play as one.

    Returns:
        The original score when nothing needs expanding, else an expanded copy.
    """
    if not counts:
        return score
    parts = list(score.parts)
    if not any(counts.get(str(part.id or part.partName), 1) > 1 for part in parts):
        return score
    rendered = stream.Score()
    if score.metadata is not None:
        rendered.metadata = copy.deepcopy(score.metadata)
    for part in parts:
        name = str(part.id or part.partName)
        count = max(1, counts.get(name, 1))
        scale = 1.0 / math.sqrt(count)
        for index in range(count):
            clone = copy.deepcopy(part)
            if count > 1:
                clone.id = f"{name}~{index + 1}"
                clone.partName = clone.id
                for item in clone.recurse().notes:
                    velocity = item.volume.velocity
                    if velocity is not None:
                        item.volume.velocity = max(1, round(velocity * scale))
            rendered.insert(0.0, clone)
    return rendered


def _instrumental_score(
    score: stream.Score, sung_voices: Collection[str]
) -> stream.Score:
    """Return a copy of the score without the parts that are actually sung.

    Only voices for which a vocal stem really exists are removed; a part that
    is merely marked vocal but produced no singable note stays in the
    instrumental render instead of dropping out of the audio entirely.

    Args:
        score: The score to copy.
        sung_voices: Voices that are sung.

    Returns:
        The original score when nothing needs removing, else a copy.
    """
    names = set(sung_voices)
    if not any(str(part.id or part.partName) in names for part in score.parts):
        return score
    clone = copy.deepcopy(score)
    for part in list(clone.parts):
        if str(part.id or part.partName) in names:
            clone.remove(part)
    return clone


def _write_wav(path: Path, samples: NDArray[np.float64]) -> None:
    """Write mono float samples as a 16-bit PCM WAV file.

    Non-finite samples are replaced with silence and the rest is clipped to
    ``[-1, 1]``, so a misbehaving backend can never produce a corrupt file.

    Args:
        path: Destination path.
        samples: Mono samples in ``[-1, 1]``.
    """
    finite = np.nan_to_num(
        np.asarray(samples, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0
    )
    pcm = (np.clip(finite, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _is_silent(samples: NDArray[np.float64]) -> bool:
    """Return whether a stem contains nothing audible.

    Non-finite samples do not count as audible, so a broken backend cannot
    silence a part by returning ``NaN``/``inf`` instead of zeros.

    Args:
        samples: Mono samples.

    Returns:
        ``True`` for an empty or all-but-flat buffer.
    """
    audible = np.isfinite(samples) & (np.abs(samples) > _DIGITAL_SILENCE)
    return not bool(np.any(audible))


def _fit_vocal_level(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalise a vocal stem to the target mix level.

    The RMS is measured on the sounding part only, so a stem with long rests
    (or a long instrumental lead-in) is not over-amplified.  Both quiet and
    loud stems are brought towards the target, with a gain cap and a peak
    ceiling.

    Args:
        samples: Mono samples.

    Returns:
        The level-fitted samples.
    """
    if samples.size == 0:
        return samples
    active = samples[np.abs(samples) > _DIGITAL_SILENCE]
    if active.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(np.square(active))))
    gain = min(_VOCAL_RMS / max(rms, _DIGITAL_SILENCE), _MAX_VOCAL_GAIN)
    scaled = samples * gain
    peak = float(np.max(np.abs(scaled)))
    if peak > _VOCAL_PEAK:
        scaled *= _VOCAL_PEAK / peak
    return scaled


def _render_instrumental(
    score: stream.Score,
    sung_voices: Collection[str],
    player_counts: dict[str, int] | None,
    features: Features,
    temp_dir: Path,
) -> Path | None:
    """Render the instrumental part of a score with FluidSynth.

    Args:
        score: The full score to render.
        sung_voices: Voices that are actually sung and must be excluded.
        player_counts: Voice-to-player-count mapping.
        features: Detected audio backend features.
        temp_dir: Scratch directory for the MIDI and WAV files.

    Returns:
        The instrumental WAV path, or ``None`` when no instrument remains.
    """
    instrumental = _instrumental_score(score, sung_voices)
    if not instrumental.parts:
        return None
    prepared = _apply_section_sizes(instrumental, player_counts)
    realize_expressions(prepared)
    midi_path = temp_dir / "score.mid"
    wav_path = temp_dir / "score.wav"
    prepared.write("midi", fp=str(midi_path))  # type: ignore[no-untyped-call]  # music21
    assert features.fluidsynth is not None
    assert features.soundfont is not None
    subprocess.run(
        [
            str(features.fluidsynth),
            "-ni",
            "-g",
            _FLUIDSYNTH_GAIN,
            "-F",
            str(wav_path),
            str(features.soundfont),
            str(midi_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


def _mix_and_encode(
    inputs: Sequence[Path],
    out_path: Path,
    features: Features,
    codec: str,
) -> None:
    """Mix WAV inputs and encode the result.

    Args:
        inputs: WAV files to mix, in order.
        out_path: Compressed destination.
        features: Detected audio backend features.
        codec: ffmpeg audio codec.
    """
    assert features.ffmpeg is not None
    args = [str(features.ffmpeg), "-y"]
    for path in inputs:
        args.extend(["-i", str(path)])
    if len(inputs) == 1:
        args.extend(["-af", _LOUDNESS_FILTER])
    else:
        streams = "".join(f"[{index}:a]" for index in range(len(inputs)))
        args.extend(
            [
                "-filter_complex",
                (
                    f"{streams}amix=inputs={len(inputs)}:duration=longest:"
                    f"normalize=0,{_LOUDNESS_FILTER}"
                ),
            ]
        )
    args.extend(["-c:a", codec, "-b:a", "192k", str(out_path)])
    subprocess.run(args, check=True, capture_output=True)


def _vocal_stems(
    lines: Mapping[str, Sequence[SungNote]],
    backend: VocalBackend,
    temp_dir: Path,
) -> dict[str, Path]:
    """Synthesise one WAV stem per audible vocal voice.

    A voice whose backend output is empty or completely silent is skipped, so
    its part can still be played by the instrumental render rather than
    disappearing from the mix.

    Args:
        lines: Timed sung notes per voice.
        backend: Vocal synthesis backend.
        temp_dir: Scratch directory.

    Returns:
        A voice-to-stem-path mapping for the voices that produced audio.
    """
    stems: dict[str, Path] = {}
    for voice, notes in lines.items():
        samples = backend.synthesize(notes)
        if _is_silent(samples):
            continue
        path = temp_dir / f"voice-{len(stems)}.wav"
        _write_wav(path, _fit_vocal_level(samples))
        stems[voice] = path
    return stems


def synthesize_audio(
    score: stream.Score,
    out_path: Path,
    features: Features,
    fmt: str = "m4a",
    *,
    player_counts: dict[str, int] | None = None,
    vocal_voices: Mapping[str, str] | None = None,
    vocal_backend: VocalBackend | None = None,
) -> Path:
    """Render a score to a compressed audio file.

    Args:
        score: The score to render.
        out_path: Destination path; the suffix follows ``fmt``.
        features: Detected audio backend features.
        fmt: Target format, either ``"m4a"`` or ``"mp3"``.
        player_counts: Voice-to-player-count mapping used for section playback.
        vocal_voices: Voices that are sung, mapped to language tags.
        vocal_backend: Vocal synthesis backend; defaults to the built-in singer.

    Returns:
        The written audio path.

    Raises:
        FeatureUnavailableError: When ffmpeg, FluidSynth or a soundfont is
            missing.
    """
    if not features.audio_available:
        raise FeatureUnavailableError(
            "ffmpeg, FluidSynth and a soundfont are required for audio export. "
            "They download automatically on first launch; if still unavailable, "
            "check the network or install them on the system PATH."
        )
    codec = _CODECS[fmt]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vocals = dict(vocal_voices or {})
    lines = vocal_lines(score, vocals) if vocals else {}
    backend = vocal_backend or AutoVocalBackend()
    with _lock_for(out_path):
        # A unique temp directory keeps concurrent exports from sharing (and
        # fighting over) the intermediate MIDI/WAV files, and the whole
        # directory is removed best-effort so a lock can never crash the export.
        temp_dir = Path(tempfile.mkdtemp(prefix="ht-audio-"))
        try:
            # Synthesise the voices first: only a voice that really produced
            # audio may be removed from the instrumental render, otherwise a
            # marked-but-unsingable part would vanish from the mix entirely.
            stems = _vocal_stems(lines, backend, temp_dir)
            instrumental = _render_instrumental(
                score, tuple(stems), player_counts, features, temp_dir
            )
            inputs: list[Path] = []
            if instrumental is not None:
                inputs.append(instrumental)
            inputs.extend(stems.values())
            if not inputs:
                raise FeatureUnavailableError("The score has no notes to render.")
            _mix_and_encode(inputs, out_path, features, codec)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return out_path
