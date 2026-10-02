# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Audio backend feature detection.

Audio rendering relies on FluidSynth, a soundfont and ffmpeg.  They are looked
up first inside ``vendor/`` (downloaded on first launch by
:mod:`harmoniatextor.render.vendor`) and then on the system ``PATH``, so the
application can degrade gracefully instead of failing silently.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = ["FeatureDetector", "Features"]

_SOUNDFONT_PATTERNS = ("*.sf2", "*.sf3")


@dataclass(frozen=True, slots=True)
class Features:
    """Availability of the audio backend binaries.

    Attributes:
        ffmpeg: Path to the ffmpeg executable, if present.
        fluidsynth: Path to the FluidSynth executable, if present.
        soundfont: Path to a ``.sf2``/``.sf3`` soundfont, if present.
    """

    ffmpeg: Path | None = None
    fluidsynth: Path | None = None
    soundfont: Path | None = None

    @property
    def audio_available(self) -> bool:
        """Whether compressed audio export (M4A/MP3) is possible."""
        return (
            self.ffmpeg is not None and self.fluidsynth is not None and self.soundfont is not None
        )

    @property
    def playback_available(self) -> bool:
        """Whether high-fidelity server-side playback is possible."""
        return self.fluidsynth is not None and self.soundfont is not None


class FeatureDetector:
    """Detect the audio backend binaries.

    Attributes:
        vendor_root: Root of the vendored binaries.
    """

    def __init__(self, vendor_root: Path) -> None:
        """Initialise the detector.

        Args:
            vendor_root: Root of the vendored binaries.
        """
        self.vendor_root = vendor_root

    def detect(self) -> Features:
        """Detect the audio backend binaries.

        Returns:
            The detected features.
        """
        return Features(
            ffmpeg=self._find_executable("ffmpeg"),
            fluidsynth=self._find_executable("fluidsynth"),
            soundfont=self._find_soundfont(),
        )

    def _find_executable(self, name: str) -> Path | None:
        """Locate an executable in ``vendor/bin`` or on the ``PATH``.

        Args:
            name: Executable name without suffix, e.g. ``"ffmpeg"``.

        Returns:
            The executable path, or ``None`` when it is unavailable.
        """
        binary_dir = self.vendor_root / "bin"
        for candidate in (binary_dir / f"{name}.exe", binary_dir / name):
            if candidate.is_file():
                return candidate
        found = shutil.which(name)
        return Path(found) if found is not None else None

    def _find_soundfont(self) -> Path | None:
        """Locate a soundfont from the environment or ``vendor/soundfonts``.

        Returns:
            The soundfont path, or ``None`` when none is available.
        """
        candidates: list[Path] = []
        override = os.environ.get("HARMONIA_SOUNDFONT")
        if override:
            candidates.append(Path(override))
        soundfont_dir = self.vendor_root / "soundfonts"
        for pattern in _SOUNDFONT_PATTERNS:
            candidates.extend(sorted(soundfont_dir.glob(pattern)))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None
