# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Export service producing MusicXML, MIDI, M4A, MP3 and zip artifacts."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from music21 import stream

from harmoniatextor.render.audio import FeatureUnavailableError, synthesize_audio
from harmoniatextor.render.features import FeatureDetector, Features
from harmoniatextor.score.io import from_musicxml, to_musicxml
from harmoniatextor.service.service import CompositionService

__all__ = ["ExportResult", "ExportService", "safe_filename"]

_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_MAX_FILENAME = 120


def safe_filename(title: str, fallback: str) -> str:
    """Turn a work title into a filesystem-safe file name stem.

    Characters forbidden by Windows, macOS or Linux are removed, whitespace is
    collapsed, trailing dots and spaces are stripped, reserved device names are
    replaced by the fallback and the length is capped.

    Args:
        title: The raw title.
        fallback: Stem used when nothing usable remains.

    Returns:
        A safe file name stem.
    """
    cleaned = re.sub(r"\s+", " ", _UNSAFE_FILENAME.sub("", title)).strip().rstrip(".")
    if not cleaned or cleaned.upper() in _RESERVED_NAMES:
        return fallback
    return cleaned[:_MAX_FILENAME]


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Outcome of an export operation.

    Attributes:
        ok: Whether the export succeeded.
        path: Path of the produced artifact.
        error: Error message when the export failed.
    """

    ok: bool
    path: Path | None = None
    error: str = ""


class ExportService:
    """Export movements to downloadable artifacts.

    Attributes:
        service: Composition service providing the current score.
        vendor_root: Root of the vendored audio binaries.
    """

    def __init__(self, service: CompositionService, vendor_root: Path) -> None:
        """Initialise the export service.

        Args:
            service: Composition service.
            vendor_root: Root of the vendored binaries.
        """
        self.service = service
        self.vendor_root = vendor_root

    def features(self) -> Features:
        """Detect the available vendored features.

        Returns:
            The detected features.
        """
        return FeatureDetector(self.vendor_root).detect()

    def _title(self, work_id: str) -> str:
        """Return the filesystem-safe title of a work.

        Args:
            work_id: Work identifier.

        Returns:
            A file name stem based on the work title.
        """
        return safe_filename(self.service.get_work(work_id).title, work_id)

    def _movement_stem(self, work_id: str, movement_id: str) -> str:
        """Return a filesystem-safe stem for one movement of a work.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.

        Returns:
            ``"<title>-<movement>"``, both sanitised.
        """
        title = self._title(work_id)
        try:
            movement = self.service.get_movement(
                self.service.get_work(work_id), movement_id
            )
        except KeyError:
            return title
        return safe_filename(f"{title}-{movement.name}", f"{work_id}-{movement_id}")

    def export(
        self, work_id: str, movement_id: str, out_dir: Path, fmt: str
    ) -> ExportResult:
        """Export one movement in the requested format.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            out_dir: Output directory.
            fmt: One of ``musicxml``, ``midi``, ``m4a`` or ``mp3``.

        Returns:
            The export result.
        """
        handlers = {
            "musicxml": self.export_musicxml,
            "midi": self.export_midi,
            "m4a": self.export_m4a,
            "mp3": self.export_mp3,
        }
        handler = handlers.get(fmt)
        if handler is None:
            return ExportResult(ok=False, error=f"unsupported format: {fmt}")
        return handler(work_id, movement_id, out_dir)

    def _score(self, work_id: str, movement_id: str = "") -> stream.Score:
        """Return the score to export.

        A work with composed movements is exported as the merged full-work score;
        an empty work falls back to the movement's canonical score.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.

        Returns:
            The score to export.
        """
        merged = self.service.merged_musicxml(work_id)
        if merged:
            return from_musicxml(merged)
        return self.service.current_score(work_id, movement_id)

    def export_work(self, work_id: str, fmt: str, out_dir: Path) -> ExportResult:
        """Export a work, zipping several movements when nothing is composed.

        Args:
            work_id: Work identifier.
            fmt: Target format.
            out_dir: Output directory.

        Returns:
            The export result pointing at the single file or the zip archive.
        """
        work = self.service.get_work(work_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.service.merged_musicxml(work_id):
            return self._export_merged(work_id, out_dir, fmt)
        paths: list[Path] = []
        for movement in work.movements:
            result = self.export(work_id, movement.id, out_dir, fmt)
            if not result.ok or result.path is None:
                return result
            paths.append(result.path)
        if len(paths) == 1:
            return ExportResult(ok=True, path=paths[0])
        archive = out_dir / f"{self._title(work_id)}-{fmt}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in paths:
                bundle.write(path, arcname=path.name)
        return ExportResult(ok=True, path=archive)

    def _export_merged(self, work_id: str, out_dir: Path, fmt: str) -> ExportResult:
        """Export the merged full-work score as one artifact.

        Args:
            work_id: Work identifier.
            out_dir: Output directory.
            fmt: Target format.

        Returns:
            The export result.
        """
        stem = self._title(work_id)
        if fmt == "musicxml":
            path = out_dir / f"{stem}.musicxml"
            path.write_text(to_musicxml(self._score(work_id)), encoding="utf-8")
            return ExportResult(ok=True, path=path)
        if fmt == "midi":
            path = out_dir / f"{stem}.mid"
            self._score(work_id).write("midi", fp=str(path))  # type: ignore[no-untyped-call]
            return ExportResult(ok=True, path=path)
        if fmt in {"m4a", "mp3"}:
            path = out_dir / f"{stem}.{fmt}"
            try:
                synthesize_audio(self._score(work_id), path, self.features(), fmt)
            except FeatureUnavailableError as exc:
                return ExportResult(ok=False, error=str(exc))
            return ExportResult(ok=True, path=path)
        return ExportResult(ok=False, error=f"unsupported format: {fmt}")

    def export_musicxml(
        self, work_id: str, movement_id: str, out_dir: Path
    ) -> ExportResult:
        """Export the current score as MusicXML.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            out_dir: Output directory.

        Returns:
            The export result.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self._movement_stem(work_id, movement_id)}.musicxml"
        path.write_text(
            to_musicxml(self._score(work_id, movement_id)), encoding="utf-8"
        )
        return ExportResult(ok=True, path=path)

    def export_midi(
        self, work_id: str, movement_id: str, out_dir: Path
    ) -> ExportResult:
        """Export the current score as a MIDI file.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            out_dir: Output directory.

        Returns:
            The export result.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self._movement_stem(work_id, movement_id)}.mid"
        score = self._score(work_id, movement_id)
        score.write("midi", fp=str(path))  # type: ignore[no-untyped-call]  # music21
        return ExportResult(ok=True, path=path)

    def export_m4a(self, work_id: str, movement_id: str, out_dir: Path) -> ExportResult:
        """Export the current score as an M4A file.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            out_dir: Output directory.

        Returns:
            The export result; a friendly error is returned when the audio
            backend is unavailable.
        """
        return self._export_audio(work_id, movement_id, out_dir, "m4a")

    def export_mp3(self, work_id: str, movement_id: str, out_dir: Path) -> ExportResult:
        """Export the current score as an MP3 file.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            out_dir: Output directory.

        Returns:
            The export result; a friendly error is returned when the audio
            backend is unavailable.
        """
        return self._export_audio(work_id, movement_id, out_dir, "mp3")

    def _export_audio(
        self, work_id: str, movement_id: str, out_dir: Path, fmt: str
    ) -> ExportResult:
        """Export the current score as compressed audio.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            out_dir: Output directory.
            fmt: Target format, either ``"m4a"`` or ``"mp3"``.

        Returns:
            The export result.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self._movement_stem(work_id, movement_id)}.{fmt}"
        score = self._score(work_id, movement_id)
        try:
            synthesize_audio(score, path, self.features(), fmt)
        except FeatureUnavailableError as exc:
            return ExportResult(ok=False, error=str(exc))
        return ExportResult(ok=True, path=path)
