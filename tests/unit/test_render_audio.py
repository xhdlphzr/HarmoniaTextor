# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for render.audio."""

from __future__ import annotations

from pathlib import Path

import pytest

from harmoniatextor.render.audio import FeatureUnavailableError, synthesize_audio
from harmoniatextor.render.features import Features
from harmoniatextor.score.io import new_score

_EXPECTED_CALLS = 2


class TestAudio:
    """Audio synthesis pipeline."""

    def test_unavailable(self, tmp_path: Path) -> None:
        """Missing binaries raise a feature error."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        with pytest.raises(FeatureUnavailableError):
            synthesize_audio(score, tmp_path / "out.m4a", Features())

    def test_synthesize_m4a(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pipeline invokes FluidSynth then ffmpeg for M4A."""
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        features = Features(
            ffmpeg=tmp_path / "ffmpeg.exe",
            fluidsynth=tmp_path / "fluidsynth.exe",
            soundfont=tmp_path / "s.sf2",
        )
        out = synthesize_audio(score, tmp_path / "out.m4a", features, "m4a")
        assert out == tmp_path / "out.m4a"
        assert len(calls) == _EXPECTED_CALLS
        assert calls[1][-1] == str(out)
        assert "aac" in calls[1]
        assert not (tmp_path / "out.mid").exists()
        assert not (tmp_path / "out.wav").exists()

    def test_synthesize_mp3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pipeline encodes MP3 with libmp3lame."""
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        features = Features(
            ffmpeg=tmp_path / "ffmpeg.exe",
            fluidsynth=tmp_path / "fluidsynth.exe",
            soundfont=tmp_path / "s.sf2",
        )
        out = synthesize_audio(score, tmp_path / "out.mp3", features, "mp3")
        assert out == tmp_path / "out.mp3"
        assert "libmp3lame" in calls[1]
        assert not (tmp_path / "out.mid").exists()
        assert not (tmp_path / "out.wav").exists()

    def test_synthesize_serialises_same_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two exports of the same file both succeed (per-output lock reuse)."""
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        features = Features(
            ffmpeg=tmp_path / "ffmpeg.exe",
            fluidsynth=tmp_path / "fluidsynth.exe",
            soundfont=tmp_path / "s.sf2",
        )
        out = tmp_path / "same.m4a"
        synthesize_audio(score, out, features, "m4a")
        synthesize_audio(score, out, features, "m4a")
        assert len(calls) == _EXPECTED_CALLS * 2
