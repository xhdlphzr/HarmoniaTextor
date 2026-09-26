# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for rendering, feature detection and export."""

from __future__ import annotations

from pathlib import Path

import pytest

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.render.audio import FeatureUnavailableError, synthesize_audio
from harmoniatextor.render.exporter import ExportService
from harmoniatextor.render.features import FeatureDetector, Features
from harmoniatextor.score.io import new_score, to_musicxml
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.service.service import CompositionService

_EXPECTED_CALLS = 2


def _movement_xml() -> str:
    """Build a standalone one-part movement melody."""
    score = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["violin"])
    ScoreEditor(score).write_line("violin", 1, [ThemeNote("C5", 1.0)])
    return to_musicxml(score)


class TestFeatureDetector:
    """Audio backend detection."""

    def test_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing binaries are reported as None."""
        monkeypatch.delenv("HARMONIA_SOUNDFONT", raising=False)
        monkeypatch.setattr("harmoniatextor.render.features.shutil.which", lambda _name: None)
        features = FeatureDetector(tmp_path / "vendor").detect()
        assert features.ffmpeg is None
        assert features.fluidsynth is None
        assert features.soundfont is None
        assert not features.audio_available
        assert not features.playback_available

    def test_vendor_executables(self, tmp_path: Path) -> None:
        """Binaries with a Windows suffix are detected in vendor/bin."""
        vendor = tmp_path / "vendor"
        (vendor / "bin").mkdir(parents=True)
        (vendor / "soundfonts").mkdir(parents=True)
        (vendor / "bin" / "ffmpeg.exe").write_text("x")
        (vendor / "bin" / "fluidsynth.exe").write_text("x")
        (vendor / "soundfonts" / "bach.sf2").write_text("x")
        features = FeatureDetector(vendor).detect()
        assert features.audio_available
        assert features.playback_available

    def test_vendor_suffixless(self, tmp_path: Path) -> None:
        """Binaries without a suffix are detected in vendor/bin."""
        vendor = tmp_path / "vendor"
        (vendor / "bin").mkdir(parents=True)
        (vendor / "bin" / "ffmpeg").write_text("x")
        (vendor / "bin" / "fluidsynth").write_text("x")
        features = FeatureDetector(vendor).detect()
        assert features.ffmpeg == vendor / "bin" / "ffmpeg"
        assert features.fluidsynth == vendor / "bin" / "fluidsynth"

    def test_path_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Binaries on the PATH are used when vendor/ is empty."""
        monkeypatch.setattr(
            "harmoniatextor.render.features.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        features = FeatureDetector(tmp_path / "vendor").detect()
        assert features.ffmpeg == Path("/usr/bin/ffmpeg")
        assert features.fluidsynth == Path("/usr/bin/fluidsynth")

    def test_soundfont_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """HARMONIA_SOUNDFONT overrides the vendored soundfont."""
        font = tmp_path / "custom.sf3"
        font.write_text("x")
        monkeypatch.setenv("HARMONIA_SOUNDFONT", str(font))
        assert FeatureDetector(tmp_path / "vendor").detect().soundfont == font

    def test_soundfont_env_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stale HARMONIA_SOUNDFONT path is ignored."""
        monkeypatch.setenv("HARMONIA_SOUNDFONT", str(tmp_path / "nope.sf2"))
        assert FeatureDetector(tmp_path / "vendor").detect().soundfont is None


class TestAudio:
    """Audio synthesis pipeline."""

    def test_unavailable(self, tmp_path: Path) -> None:
        """Missing binaries raise a feature error."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"])
        with pytest.raises(FeatureUnavailableError):
            synthesize_audio(score, tmp_path / "out.m4a", Features())

    def test_synthesize_m4a(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pipeline invokes FluidSynth then ffmpeg for M4A."""
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"])
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

    def test_synthesize_mp3(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pipeline encodes MP3 with libmp3lame."""
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"])
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


class TestExportService:
    """Export operations."""

    def test_exports(self, service: CompositionService, tmp_path: Path) -> None:
        """MusicXML and MIDI export succeed; audio degrades gracefully."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        exporter = ExportService(service, tmp_path / "vendor")
        assert exporter.export_musicxml(work.id, movement_id, tmp_path / "out").ok
        assert exporter.export_midi(work.id, movement_id, tmp_path / "out").ok
        for fmt in ("m4a", "mp3"):
            result = getattr(exporter, f"export_{fmt}")(work.id, movement_id, tmp_path / "out")
            assert not result.ok
            assert result.error
        assert not exporter.features().audio_available

    def test_export_audio_success(
        self, service: CompositionService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audio export succeeds when synthesis is available."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        exporter = ExportService(service, tmp_path / "vendor")

        def fake_synth(_score: object, out_path: Path, _features: object, _fmt: str) -> Path:
            out_path.write_text("audio")
            return out_path

        monkeypatch.setattr("harmoniatextor.render.exporter.synthesize_audio", fake_synth)
        for fmt in ("m4a", "mp3"):
            result = getattr(exporter, f"export_{fmt}")(work.id, movement_id, tmp_path / "out")
            assert result.ok
            assert result.path is not None
            assert result.path.suffix == f".{fmt}"

    def test_export_dispatch(self, service: CompositionService, tmp_path: Path) -> None:
        """The dispatcher routes known formats and rejects unknown ones."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        exporter = ExportService(service, tmp_path / "vendor")
        assert exporter.export(work.id, movement_id, tmp_path / "out", "musicxml").ok
        assert not exporter.export(work.id, movement_id, tmp_path / "out", "ogg").ok

    def test_export_work_single(self, service: CompositionService, tmp_path: Path) -> None:
        """A single-movement work yields one file."""
        work = service.create_work("Demo", "plain", "C")
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_work(work.id, "musicxml", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert result.path.suffix == ".musicxml"

    def test_export_work_zip(self, service: CompositionService, tmp_path: Path) -> None:
        """A multi-movement work yields a zip archive."""
        work = service.create_work("Demo", "sonata", "C")
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_work(work.id, "musicxml", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert result.path.suffix == ".zip"

    def test_export_work_error(self, service: CompositionService, tmp_path: Path) -> None:
        """A failing movement aborts the work export."""
        work = service.create_work("Demo", "sonata", "C")
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_work(work.id, "m4a", tmp_path / "out")
        assert not result.ok
        assert result.error

    def _movement_work(self, service: CompositionService) -> str:
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.submit_theme(work.id, "m01", _movement_xml(), check=False)
        return work.id

    def test_export_musicxml_merged_movements(
        self, service: CompositionService, tmp_path: Path
    ) -> None:
        """A work with composed movements exports its merged score."""
        work_id = self._movement_work(service)
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_musicxml(work_id, "m01", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert "score-partwise" in result.path.read_text(encoding="utf-8")

    def test_export_work_movements_single(
        self, service: CompositionService, tmp_path: Path
    ) -> None:
        """A composed work exports one merged file, not a zip."""
        work_id = self._movement_work(service)
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_work(work_id, "musicxml", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert result.path.name == f"{work_id}.musicxml"

    def test_export_work_movements_formats(
        self, service: CompositionService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Merged MIDI, audio and unsupported exports are handled."""
        work_id = self._movement_work(service)
        exporter = ExportService(service, tmp_path / "vendor")
        midi = exporter.export_work(work_id, "midi", tmp_path / "out")
        assert midi.ok
        assert midi.path is not None
        assert midi.path.suffix == ".mid"
        assert not exporter.export_work(work_id, "m4a", tmp_path / "out").ok
        assert not exporter.export_work(work_id, "ogg", tmp_path / "out").ok

        def fake_synth(_score: object, out_path: Path, _features: object, _fmt: str) -> Path:
            out_path.write_text("audio")
            return out_path

        monkeypatch.setattr("harmoniatextor.render.exporter.synthesize_audio", fake_synth)
        audio = exporter.export_work(work_id, "mp3", tmp_path / "out")
        assert audio.ok
        assert audio.path is not None
        assert audio.path.suffix == ".mp3"
