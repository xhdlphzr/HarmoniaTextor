# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Integration tests for exporting a work through the service and storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.render.exporter import ExportService
from harmoniatextor.score.io import new_score, to_musicxml
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.service.service import CompositionService


def _movement_xml() -> str:
    """Build a standalone one-part movement melody.

    Returns:
        The resulting text.
    """
    score = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["violin"])
    ScoreEditor(score).write_line("violin", 1, [ThemeNote("C5", 1.0)])
    return to_musicxml(score)


class TestExportService:
    """Export operations."""

    def test_exports(self, service: CompositionService, tmp_path: Path) -> None:
        """MusicXML and MIDI export succeed; audio degrades gracefully.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        exporter = ExportService(service, tmp_path / "vendor")
        assert exporter.export_musicxml(work.id, movement_id, tmp_path / "out").ok
        assert exporter.export_midi(work.id, movement_id, tmp_path / "out").ok
        for fmt in ("m4a", "mp3"):
            result = getattr(exporter, f"export_{fmt}")(
                work.id, movement_id, tmp_path / "out"
            )
            assert not result.ok
            assert result.error
        assert not exporter.features().audio_available

    def test_export_audio_success(
        self,
        service: CompositionService,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Audio export succeeds when synthesis is available.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        exporter = ExportService(service, tmp_path / "vendor")

        def fake_synth(
            _score: object, out_path: Path, _features: object, _fmt: str
        ) -> Path:
            out_path.write_text("audio")
            return out_path

        monkeypatch.setattr(
            "harmoniatextor.render.exporter.synthesize_audio", fake_synth
        )
        for fmt in ("m4a", "mp3"):
            result = getattr(exporter, f"export_{fmt}")(
                work.id, movement_id, tmp_path / "out"
            )
            assert result.ok
            assert result.path is not None
            assert result.path.suffix == f".{fmt}"

    def test_export_dispatch(self, service: CompositionService, tmp_path: Path) -> None:
        """The dispatcher routes known formats and rejects unknown ones.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        exporter = ExportService(service, tmp_path / "vendor")
        assert exporter.export(work.id, movement_id, tmp_path / "out", "musicxml").ok
        assert not exporter.export(work.id, movement_id, tmp_path / "out", "ogg").ok

    def test_export_work_single(
        self, service: CompositionService, tmp_path: Path
    ) -> None:
        """A single-movement work yields one file.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
        work = service.create_work("Demo", "plain", "C")
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_work(work.id, "musicxml", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert result.path.suffix == ".musicxml"

    def test_export_work_zip(self, service: CompositionService, tmp_path: Path) -> None:
        """A multi-movement work yields a zip archive.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
        work = service.create_work("Demo", "sonata", "C")
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_work(work.id, "musicxml", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert result.path.suffix == ".zip"

    def test_export_work_error(
        self, service: CompositionService, tmp_path: Path
    ) -> None:
        """A failing movement aborts the work export.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
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
        """A work with composed movements exports its merged score.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
        work_id = self._movement_work(service)
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_musicxml(work_id, "m01", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert "score-partwise" in result.path.read_text(encoding="utf-8")

    def test_export_work_movements_single(
        self, service: CompositionService, tmp_path: Path
    ) -> None:
        """A composed work exports one merged file, not a zip.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
        work_id = self._movement_work(service)
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_work(work_id, "musicxml", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert result.path.name == "Demo.musicxml"

    def test_export_uses_sanitised_title(
        self, service: CompositionService, tmp_path: Path
    ) -> None:
        """Exported files use the sanitised work title.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
        work = service.create_work("我的作品:第一首/测试", "plain", "C")
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_musicxml(
            work.id, work.movements[0].id, tmp_path / "out"
        )
        assert result.ok
        assert result.path is not None
        assert "我的作品第一首测试" in result.path.name
        assert all(char not in result.path.name for char in '<>:"/\\|?*')

    def test_export_work_uses_title(
        self, service: CompositionService, tmp_path: Path
    ) -> None:
        """The merged work export is named after the title.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
        work = service.create_work("标题:测试", "plain", "C")
        service.submit_theme(
            work.id, work.movements[0].id, _movement_xml(), check=False
        )
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_work(work.id, "musicxml", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert result.path.name == "标题测试.musicxml"

    def test_export_missing_movement_uses_title(
        self, service: CompositionService, tmp_path: Path
    ) -> None:
        """An unknown movement still exports under the work title.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
        """
        work = service.create_work("标题", "plain", "C")
        service.submit_theme(
            work.id, work.movements[0].id, _movement_xml(), check=False
        )
        exporter = ExportService(service, tmp_path / "vendor")
        result = exporter.export_musicxml(work.id, "nope", tmp_path / "out")
        assert result.ok
        assert result.path is not None
        assert result.path.name == "标题.musicxml"

    def test_export_work_movements_formats(
        self,
        service: CompositionService,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Merged MIDI, audio and unsupported exports are handled.

        Args:
            service: The composition service.
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        work_id = self._movement_work(service)
        exporter = ExportService(service, tmp_path / "vendor")
        midi = exporter.export_work(work_id, "midi", tmp_path / "out")
        assert midi.ok
        assert midi.path is not None
        assert midi.path.suffix == ".mid"
        assert not exporter.export_work(work_id, "m4a", tmp_path / "out").ok
        assert not exporter.export_work(work_id, "ogg", tmp_path / "out").ok

        def fake_synth(
            _score: object, out_path: Path, _features: object, _fmt: str
        ) -> Path:
            out_path.write_text("audio")
            return out_path

        monkeypatch.setattr(
            "harmoniatextor.render.exporter.synthesize_audio", fake_synth
        )
        audio = exporter.export_work(work_id, "mp3", tmp_path / "out")
        assert audio.ok
        assert audio.path is not None
        assert audio.path.suffix == ".mp3"
