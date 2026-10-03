# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for render.features."""

from __future__ import annotations

from pathlib import Path

import pytest

from harmoniatextor.render.features import FeatureDetector


class TestFeatureDetector:
    """Audio backend detection."""

    def test_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing binaries are reported as None.

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.delenv("HARMONIA_SOUNDFONT", raising=False)
        monkeypatch.setattr(
            "harmoniatextor.render.features.shutil.which", lambda _name: None
        )
        features = FeatureDetector(tmp_path / "vendor").detect()
        assert features.ffmpeg is None
        assert features.fluidsynth is None
        assert features.soundfont is None
        assert not features.audio_available
        assert not features.playback_available

    def test_vendor_executables(self, tmp_path: Path) -> None:
        """Binaries with a Windows suffix are detected in vendor/bin.

        Args:
            tmp_path: The pytest temporary path fixture.
        """
        vendor = tmp_path / "vendor"
        (vendor / "bin").mkdir(parents=True)
        (vendor / "soundfonts").mkdir(parents=True)
        (vendor / "bin" / "ffmpeg.exe").write_text("x")
        (vendor / "bin" / "fluidsynth.exe").write_text("x")
        (vendor / "soundfonts" / "default.sf2").write_text("x")
        features = FeatureDetector(vendor).detect()
        assert features.audio_available
        assert features.playback_available

    def test_vendor_suffixless(self, tmp_path: Path) -> None:
        """Binaries without a suffix are detected in vendor/bin.

        Args:
            tmp_path: The pytest temporary path fixture.
        """
        vendor = tmp_path / "vendor"
        (vendor / "bin").mkdir(parents=True)
        (vendor / "bin" / "ffmpeg").write_text("x")
        (vendor / "bin" / "fluidsynth").write_text("x")
        features = FeatureDetector(vendor).detect()
        assert features.ffmpeg == vendor / "bin" / "ffmpeg"
        assert features.fluidsynth == vendor / "bin" / "fluidsynth"

    def test_path_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Binaries on the PATH are used when vendor/ is empty.

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(
            "harmoniatextor.render.features.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        features = FeatureDetector(tmp_path / "vendor").detect()
        assert features.ffmpeg == Path("/usr/bin/ffmpeg")
        assert features.fluidsynth == Path("/usr/bin/fluidsynth")

    def test_soundfont_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HARMONIA_SOUNDFONT overrides the vendored soundfont.

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        font = tmp_path / "custom.sf3"
        font.write_text("x")
        monkeypatch.setenv("HARMONIA_SOUNDFONT", str(font))
        assert FeatureDetector(tmp_path / "vendor").detect().soundfont == font

    def test_soundfont_env_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale HARMONIA_SOUNDFONT path is ignored.

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setenv("HARMONIA_SOUNDFONT", str(tmp_path / "nope.sf2"))
        assert FeatureDetector(tmp_path / "vendor").detect().soundfont is None
