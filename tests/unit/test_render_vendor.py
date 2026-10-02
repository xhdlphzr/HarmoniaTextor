# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for render.vendor."""

from __future__ import annotations

import platform
import tarfile
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path

import pytest

from harmoniatextor.render import vendor

_EXPECTED_DOWNLOADS = 3


def _zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)


def _tar_xz(path: Path, members: dict[str, bytes]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, data in members.items():
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
        with tarfile.open(path, "w:xz") as bundle:
            for name in members:
                bundle.add(root / name, arcname=name)


def _ffmpeg_zip(target: Path, member: str = "ffmpeg.exe") -> None:
    _zip(target, {f"build/bin/{member}": b"x"})


def _fluidsynth_zip(target: Path) -> None:
    _zip(
        target,
        {
            "fluidsynth-2.6.0/bin/fluidsynth.exe": b"x",
            "fluidsynth-2.6.0/bin/": b"",
            "fluidsynth-2.6.0/share/README": b"y",
        },
    )


def _soundfont_zip(target: Path) -> None:
    _zip(target, {"GeneralUser-GS-main/GeneralUser-GS.sf2": b"z"})


def _ready_vendor(root: Path) -> None:
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "ffmpeg.exe").write_bytes(b"x")
    (root / "bin" / "fluidsynth.exe").write_bytes(b"x")
    (root / "soundfonts").mkdir(parents=True)
    (root / "soundfonts" / "GeneralUser-GS.sf2").write_bytes(b"x")


class TestDownload:
    """Raw download helper."""

    def test_writes_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The download helper fetches into the target path."""
        recorded: dict[str, str] = {}

        def fake_urlretrieve(url: str, target: str) -> None:
            recorded["url"] = url
            Path(target).write_bytes(b"x")

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
        vendor._download("https://example/x", tmp_path / "nested" / "f.bin")
        assert recorded["url"] == "https://example/x"
        assert (tmp_path / "nested" / "f.bin").read_bytes() == b"x"


class TestExtractExecutable:
    """Zip and tar executable extraction."""

    def test_zip(self, tmp_path: Path) -> None:
        """A zip member is extracted by basename."""
        archive = tmp_path / "a.zip"
        _zip(archive, {"pkg/bin/tool.exe": b"z"})
        assert vendor._extract_executable(archive, tmp_path / "bin", "tool.exe")
        assert (tmp_path / "bin" / "tool.exe").read_bytes() == b"z"

    def test_tar(self, tmp_path: Path) -> None:
        """A tar member is extracted by basename."""
        archive = tmp_path / "a.tar.xz"
        _tar_xz(archive, {"pkg/tool": b"z"})
        assert vendor._extract_executable(archive, tmp_path / "bin", "tool")
        assert (tmp_path / "bin" / "tool").read_bytes() == b"z"

    def test_zip_missing(self, tmp_path: Path) -> None:
        """A zip without the member reports failure."""
        archive = tmp_path / "a.zip"
        _zip(archive, {"pkg/x": b"z"})
        assert not vendor._extract_executable(archive, tmp_path / "bin", "tool")

    def test_tar_missing(self, tmp_path: Path) -> None:
        """A tar without the member reports failure."""
        archive = tmp_path / "a.tar.xz"
        _tar_xz(archive, {"pkg/x": b"z"})
        assert not vendor._extract_executable(archive, tmp_path / "bin", "tool")


class TestExtractSoundfont:
    """Soundfont extraction edge cases."""

    def test_missing_member(self, tmp_path: Path) -> None:
        """An archive without a .sf2 leaves no soundfont."""
        archive = tmp_path / "a.zip"
        _zip(archive, {"x/y.txt": b"z"})
        target = tmp_path / "out.sf2"
        vendor._extract_soundfont(archive, target)
        assert not target.exists()


class TestFetch:
    """The idempotent vendor fetch."""

    def test_windows_downloads_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """ffmpeg, FluidSynth and the soundfont are all fetched on Windows."""
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        calls: list[str] = []

        def fake_download(url: str, target: Path) -> None:
            calls.append(url)
            if "ffmpeg" in url:
                _ffmpeg_zip(target)
            elif "fluidsynth" in url:
                _fluidsynth_zip(target)
            else:
                _soundfont_zip(target)

        monkeypatch.setattr(vendor, "_download", fake_download)
        vendor.fetch(tmp_path)
        assert (tmp_path / "bin" / "ffmpeg.exe").is_file()
        assert (tmp_path / "bin" / "fluidsynth.exe").is_file()
        assert (tmp_path / "soundfonts" / "GeneralUser-GS.sf2").is_file()
        assert len(calls) == _EXPECTED_DOWNLOADS

    def test_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing binaries and soundfonts are kept."""
        _ready_vendor(tmp_path)
        calls: list[str] = []
        monkeypatch.setattr(vendor, "_download", lambda url, _target: calls.append(url))
        vendor.fetch(tmp_path)
        assert calls == []

    def test_unsupported_platform(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the soundfont is fetched on an unsupported platform."""
        monkeypatch.setattr(platform, "system", lambda: "Plan9")
        calls: list[str] = []

        def fake_download(url: str, target: Path) -> None:
            calls.append(url)
            _soundfont_zip(target)

        monkeypatch.setattr(vendor, "_download", fake_download)
        vendor.fetch(tmp_path)
        assert calls == [vendor.SOUNDFONT_ARCHIVE_URL]

    def test_linux_tar_ffmpeg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tar.xz ffmpeg build is extracted and made executable."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")

        def fake_download(_url: str, target: Path) -> None:
            if target.suffix == ".xz":
                _tar_xz(target, {"ffmpeg-6/ffmpeg": b"x"})
            else:
                _soundfont_zip(target)

        monkeypatch.setattr(vendor, "_download", fake_download)
        vendor.fetch(tmp_path)
        assert (tmp_path / "bin" / "ffmpeg").is_file()

    def test_ffmpeg_missing_member(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bad ffmpeg archive is ignored without crashing."""
        monkeypatch.setattr(platform, "system", lambda: "Windows")

        def fake_download(url: str, target: Path) -> None:
            if "ffmpeg" in url:
                _zip(target, {"pkg/x": b"z"})
            elif "fluidsynth" in url:
                _fluidsynth_zip(target)
            else:
                _soundfont_zip(target)

        monkeypatch.setattr(vendor, "_download", fake_download)
        vendor.fetch(tmp_path)
        assert not (tmp_path / "bin" / "ffmpeg.exe").exists()


class TestEnsureAsync:
    """Background one-off fetch."""

    def test_complete_backend_does_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A complete backend is left untouched."""
        _ready_vendor(tmp_path)
        started: list[dict[str, object]] = []
        monkeypatch.setattr(threading, "Thread", lambda **kwargs: started.append(kwargs))
        vendor.ensure_async(tmp_path)
        assert started == []

    def test_starts_thread_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fetch runs at most once per process."""
        vendor._FETCH_STARTED.clear()
        started: list[dict[str, object]] = []

        class FakeThread:
            """A thread double that records start requests."""

            def __init__(self, **kwargs: object) -> None:
                started.append(kwargs)

            def start(self) -> None:
                pass

        monkeypatch.setattr(threading, "Thread", FakeThread)
        vendor.ensure_async(tmp_path)
        vendor.ensure_async(tmp_path)
        assert len(started) == 1

    def test_safe_fetch_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A successful fetch is silent."""
        monkeypatch.setattr(vendor, "fetch", lambda _vendor: None)
        vendor._safe_fetch(tmp_path)

    def test_safe_fetch_swallows_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed fetch never raises."""

        def boom(_vendor: Path) -> None:
            raise RuntimeError("network")

        monkeypatch.setattr(vendor, "fetch", boom)
        vendor._safe_fetch(tmp_path)
