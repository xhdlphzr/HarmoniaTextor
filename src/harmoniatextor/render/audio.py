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
from pathlib import Path

from music21 import stream

from harmoniatextor.render.expression import realize_expressions
from harmoniatextor.render.features import Features

__all__ = ["FeatureUnavailableError", "synthesize_audio"]

_CODECS: dict[str, str] = {"m4a": "aac", "mp3": "libmp3lame"}

#: FluidSynth master gain.  Its default (0.2) renders far too quietly, so the
#: synth is driven hotter before the loudness pass below.
_FLUIDSYNTH_GAIN = "1.0"
#: ffmpeg loudness normalisation.  Raises quiet scores to a loud, consistent
#: level and true-peak limits the result so the boosted gain cannot clip.
_LOUDNESS_FILTER = "loudnorm=I=-14:LRA=11:TP=-1.5"

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


def synthesize_audio(
    score: stream.Score,
    out_path: Path,
    features: Features,
    fmt: str = "m4a",
    *,
    player_counts: dict[str, int] | None = None,
) -> Path:
    """Render a score to a compressed audio file.

    Args:
        score: The score to render.
        out_path: Destination path; the suffix follows ``fmt``.
        features: Detected audio backend features.
        fmt: Target format, either ``"m4a"`` or ``"mp3"``.
        player_counts: Voice-to-player-count mapping used for section playback.

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
            prepared = _apply_section_sizes(score, player_counts)
            realize_expressions(prepared)
            prepared.write("midi", fp=str(midi_path))  # type: ignore[no-untyped-call]  # music21
            assert features.fluidsynth is not None
            assert features.soundfont is not None
            assert features.ffmpeg is not None
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
            subprocess.run(
                [
                    str(features.ffmpeg),
                    "-y",
                    "-i",
                    str(wav_path),
                    "-af",
                    _LOUDNESS_FILTER,
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
