# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for render.audio."""

from __future__ import annotations

from pathlib import Path

import pytest
from music21 import metadata

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.render.audio import (
    FeatureUnavailableError,
    _apply_section_sizes,
    synthesize_audio,
)
from harmoniatextor.render.features import Features
from harmoniatextor.score.io import new_score
from harmoniatextor.score.streamops import ScoreEditor

_EXPECTED_CALLS = 2


class TestAudio:
    """Audio synthesis pipeline."""

    def test_unavailable(self, tmp_path: Path) -> None:
        """Missing binaries raise a feature error.

        Args:
            tmp_path: The pytest temporary path fixture.
        """
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        with pytest.raises(FeatureUnavailableError):
            synthesize_audio(score, tmp_path / "out.m4a", Features())

    def test_synthesize_m4a(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pipeline invokes FluidSynth then ffmpeg for M4A.

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
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
        assert calls[0][calls[0].index("-g") + 1] == "1.0"
        assert calls[1][calls[1].index("-af") + 1].startswith("loudnorm")
        assert calls[1][-1] == str(out)
        assert "aac" in calls[1]
        assert not (tmp_path / "out.mid").exists()
        assert not (tmp_path / "out.wav").exists()

    def test_synthesize_mp3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pipeline encodes MP3 with libmp3lame.

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
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
        """Two exports of the same file both succeed (per-output lock reuse).

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
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

    def test_section_sizes_noop(self) -> None:
        """No counts, an empty mapping and unit counts keep the score as is."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["flute"])
        assert _apply_section_sizes(score, None) is score
        assert _apply_section_sizes(score, {}) is score
        assert _apply_section_sizes(score, {"flute": 1}) is score

    def test_section_sizes_expand(self) -> None:
        """A section is expanded into quietened per-player copies."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["flute", "violin"]
        )
        ScoreEditor(score).write_line("flute", 1, [ThemeNote("C5", 1.0)])
        score.metadata = metadata.Metadata()
        score.metadata.title = "Section"
        for item in score.recurse().notes:
            item.volume.velocity = 90
        rendered = _apply_section_sizes(score, {"flute": 3})
        assert rendered.metadata is not None
        assert rendered.metadata.title == "Section"
        names = sorted(str(part.id or part.partName) for part in rendered.parts)
        assert names == ["flute~1", "flute~2", "flute~3", "violin"]
        copies = [part for part in rendered.parts if str(part.id).startswith("flute~")]
        for part in copies:
            velocities = [item.volume.velocity for item in part.recurse().notes]
            assert velocities == [52]
        assert len(score.parts) == 2
