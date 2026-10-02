# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Download the audio backend binaries into the vendor directory.

On first launch the desktop app fetches ffmpeg, FluidSynth and a General MIDI
soundfont into ``~/.harmonia_textor/vendor`` so that m4a/mp3 export works on a
fresh machine.  The download is idempotent and runs once in a background thread;
a failure simply leaves audio unavailable instead of crashing the app.
"""

from __future__ import annotations

import platform
import tarfile
import threading
import urllib.request
import zipfile
from pathlib import Path

from harmoniatextor.render.features import FeatureDetector

__all__ = ["ensure_async", "fetch"]

FLUIDSYNTH_VERSION = "2.6.0"
FLUIDSYNTH_BASE = "https://github.com/FluidSynth/fluidsynth/releases/download"
FLUIDSYNTH_ASSETS: dict[str, str] = {
    "Windows": f"fluidsynth-v{FLUIDSYNTH_VERSION}-win10-x64-cpp11.zip",
}
SOUNDFONT_ARCHIVE_URL = "https://codeload.github.com/mrbumpy409/GeneralUser-GS/zip/refs/heads/main"
SOUNDFONT_NAME = "GeneralUser-GS.sf2"

#: Per platform: (download url, local archive name, executable basename).
FFMPEG_BUILDS: dict[str, tuple[str, str, str]] = {
    "Windows": (
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "ffmpeg-win64.zip",
        "ffmpeg.exe",
    ),
    "Darwin": ("https://evermeet.cx/ffmpeg/getrelease/zip", "ffmpeg-macos.zip", "ffmpeg"),
    "Linux": (
        "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
        "ffmpeg-linux.tar.xz",
        "ffmpeg",
    ),
}

_FETCH_LOCK = threading.Lock()
_FETCH_STARTED = threading.Event()


def _download(url: str, target: Path) -> None:
    """Download a URL to a local file.

    Args:
        url: Source URL.
        target: Destination file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)  # fixed https URLs


def _has_binary(bin_dir: Path, name: str) -> bool:
    """Return whether a vendored executable already exists.

    Args:
        bin_dir: ``vendor/bin`` directory.
        name: Executable basename, e.g. ``"ffmpeg.exe"``.

    Returns:
        ``True`` when the executable exists.
    """
    return (bin_dir / name).is_file()


def _extract_executable(archive: Path, bin_dir: Path, name: str) -> bool:
    """Extract one executable from a zip or tar archive.

    Args:
        archive: The downloaded archive.
        bin_dir: Destination ``vendor/bin`` directory.
        name: Executable basename to extract.

    Returns:
        ``True`` when the executable was found and written.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if Path(member).name == name:
                    (bin_dir / name).write_bytes(bundle.read(member))
                    return True
        return False
    with tarfile.open(archive) as bundle:
        for entry in bundle.getmembers():
            if entry.isfile() and Path(entry.name).name == name:
                handle = bundle.extractfile(entry)
                if handle is not None:
                    (bin_dir / name).write_bytes(handle.read())
                    return True
    return False


def _extract_binaries(archive: Path, bin_dir: Path) -> None:
    """Extract the ``bin/`` members of a FluidSynth archive.

    Args:
        archive: The downloaded zip archive.
        bin_dir: Destination ``vendor/bin`` directory.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if "/bin/" not in member or member.endswith("/"):
                continue
            (bin_dir / Path(member).name).write_bytes(bundle.read(member))


def _extract_soundfont(archive: Path, soundfont: Path) -> None:
    """Extract the first ``.sf2`` member of an archive.

    Args:
        archive: The downloaded zip archive.
        soundfont: Destination soundfont path.
    """
    soundfont.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.lower().endswith(".sf2"):
                soundfont.write_bytes(bundle.read(member))
                return


def _ensure_ffmpeg(vendor: Path, bin_dir: Path) -> None:
    """Download ffmpeg for the current platform when missing.

    Args:
        vendor: Vendor directory.
        bin_dir: ``vendor/bin`` directory.
    """
    system = platform.system()
    build = FFMPEG_BUILDS.get(system)
    if build is None:
        return
    if _has_binary(bin_dir, "ffmpeg") or _has_binary(bin_dir, "ffmpeg.exe"):
        return
    url, filename, member = build
    archive = vendor / filename
    _download(url, archive)
    if _extract_executable(archive, bin_dir, member) and system != "Windows":
        (bin_dir / member).chmod(0o755)
    archive.unlink(missing_ok=True)


def _ensure_fluidsynth(vendor: Path, bin_dir: Path) -> None:
    """Download FluidSynth for the current platform when missing.

    Args:
        vendor: Vendor directory.
        bin_dir: ``vendor/bin`` directory.
    """
    if _has_binary(bin_dir, "fluidsynth.exe") or _has_binary(bin_dir, "fluidsynth"):
        return
    asset = FLUIDSYNTH_ASSETS.get(platform.system())
    if asset is None:
        return
    archive = vendor / asset
    _download(f"{FLUIDSYNTH_BASE}/v{FLUIDSYNTH_VERSION}/{asset}", archive)
    _extract_binaries(archive, bin_dir)
    archive.unlink(missing_ok=True)


def _ensure_soundfont(vendor: Path) -> None:
    """Download the General MIDI soundfont when missing.

    Args:
        vendor: Vendor directory.
    """
    soundfont = vendor / "soundfonts" / SOUNDFONT_NAME
    if soundfont.exists():
        return
    archive = vendor / "GeneralUser-GS.zip"
    _download(SOUNDFONT_ARCHIVE_URL, archive)
    _extract_soundfont(archive, soundfont)
    archive.unlink(missing_ok=True)


def fetch(vendor: Path) -> None:
    """Download ffmpeg, FluidSynth and a soundfont when missing.

    The operation is idempotent: existing binaries and soundfonts are kept.

    Args:
        vendor: Vendor directory, normally ``~/.harmonia_textor/vendor``.
    """
    bin_dir = vendor / "bin"
    _ensure_ffmpeg(vendor, bin_dir)
    _ensure_fluidsynth(vendor, bin_dir)
    _ensure_soundfont(vendor)


def _safe_fetch(vendor: Path) -> None:
    """Run :func:`fetch` without ever raising.

    Args:
        vendor: Vendor directory.
    """
    try:
        fetch(vendor)
    except Exception:  # a failed download must not crash the app
        return


def ensure_async(vendor: Path) -> None:
    """Start a one-off background download when the audio backend is missing.

    Does nothing when ffmpeg, FluidSynth and a soundfont are all present.

    Args:
        vendor: Vendor directory.
    """
    features = FeatureDetector(vendor).detect()
    if (
        features.ffmpeg is not None
        and features.fluidsynth is not None
        and features.soundfont is not None
    ):
        return
    with _FETCH_LOCK:
        if _FETCH_STARTED.is_set():
            return
        _FETCH_STARTED.set()
    threading.Thread(target=_safe_fetch, args=(vendor,), daemon=True).start()
