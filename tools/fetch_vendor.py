# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Fetch the audio backend binaries into ``vendor/``.

M4A/MP3 export needs FluidSynth, a soundfont and ffmpeg.  This helper downloads
FluidSynth and a General MIDI soundfont into ``vendor/`` (which is git-ignored)
and reuses an ffmpeg that is already on the ``PATH`` when possible.
"""

from __future__ import annotations

import argparse
import platform
import sys
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

__all__ = ["fetch", "main"]

FLUIDSYNTH_VERSION = "2.6.0"
FLUIDSYNTH_BASE = "https://github.com/FluidSynth/fluidsynth/releases/download"
FLUIDSYNTH_ASSETS: dict[str, str] = {
    "Windows": f"fluidsynth-v{FLUIDSYNTH_VERSION}-win10-x64-cpp11.zip",
}
SOUNDFONT_ARCHIVE_URL = "https://codeload.github.com/mrbumpy409/GeneralUser-GS/zip/refs/heads/main"
SOUNDFONT_NAME = "GeneralUser-GS.sf2"


def _download(url: str, target: Path) -> None:
    """Download a URL to a local file.

    Args:
        url: Source URL.
        target: Destination file.
    """
    print(f"下载 {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)  # fixed https URLs


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


def fetch(vendor: Path) -> None:
    """Download the FluidSynth binaries and a soundfont into ``vendor``.

    Args:
        vendor: Vendor directory, normally ``vendor``.
    """
    asset = FLUIDSYNTH_ASSETS.get(platform.system())
    if asset is None:
        print("此平台没有预编译的 FluidSynth;请用系统包管理器安装后加入 PATH。")
    else:
        archive = vendor / asset
        _download(f"{FLUIDSYNTH_BASE}/v{FLUIDSYNTH_VERSION}/{asset}", archive)
        _extract_binaries(archive, vendor / "bin")
        archive.unlink()
    soundfont = vendor / "soundfonts" / SOUNDFONT_NAME
    if soundfont.exists():
        print(f"音色库已存在:{soundfont}")
    else:
        archive = vendor / "GeneralUser-GS.zip"
        _download(SOUNDFONT_ARCHIVE_URL, archive)
        _extract_soundfont(archive, soundfont)
        archive.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the vendor fetcher.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description="Fetch audio backend binaries into vendor/.")
    parser.add_argument("--vendor", default="vendor", help="Vendor directory (default: vendor).")
    args = parser.parse_args(argv)
    fetch(Path(args.vendor))
    print("完成。请确保 ffmpeg 在 vendor/bin 或 PATH 中。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
