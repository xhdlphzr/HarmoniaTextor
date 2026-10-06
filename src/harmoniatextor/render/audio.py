# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Audio synthesis pipeline built on FluidSynth and ffmpeg.

The pipeline writes a temporary MIDI file, renders it to WAV with FluidSynth,
then transcodes the WAV to the requested compressed format with ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from music21 import stream

from harmoniatextor.render.expression import realize_expressions
from harmoniatextor.render.features import Features

__all__ = ["FeatureUnavailableError", "synthesize_audio"]

_CODECS: dict[str, str] = {"m4a": "aac", "mp3": "libmp3lame"}

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


def synthesize_audio(
    score: stream.Score,
    out_path: Path,
    features: Features,
    fmt: str = "m4a",
) -> Path:
    """Render a score to a compressed audio file.

    Args:
        score: The score to render.
        out_path: Destination path; the suffix follows ``fmt``.
        features: Detected audio backend features.
        fmt: Target format, either ``"m4a"`` or ``"mp3"``.

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
    with _lock_for(out_path):
        # A unique temp directory keeps concurrent exports from sharing (and
        # fighting over) the intermediate MIDI/WAV files, and the whole
        # directory is removed best-effort so a lock can never crash the export.
        temp_dir = Path(tempfile.mkdtemp(prefix="ht-audio-"))
        midi_path = temp_dir / "score.mid"
        wav_path = temp_dir / "score.wav"
        try:
            realize_expressions(score)
            score.write("midi", fp=str(midi_path))  # type: ignore[no-untyped-call]  # music21
            assert features.fluidsynth is not None
            assert features.soundfont is not None
            assert features.ffmpeg is not None
            subprocess.run(
                [
                    str(features.fluidsynth),
                    "-ni",
                    "-F",
                    str(wav_path),
                    str(features.soundfont),
                    str(midi_path),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    str(features.ffmpeg),
                    "-y",
                    "-i",
                    str(wav_path),
                    "-c:a",
                    codec,
                    "-b:a",
                    "192k",
                    str(out_path),
                ],
                check=True,
                capture_output=True,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return out_path
