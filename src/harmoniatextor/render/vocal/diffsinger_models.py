# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Download and locate optional DiffSinger ONNX voice models.

DiffSinger is an optional, high-quality singing backend.  The application uses
the FOF singer until a model has been downloaded; downloads run in a background
thread so the first audio export never blocks.  The models are personal-PC sized
(about 220 MB for the Chinese OpenCpop model and 555 MB for the English Tiger
voicebank) and are stored outside the repository under the user's data folder.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import urllib.request
import zipfile
from pathlib import Path

from harmoniatextor import config

__all__ = [
    "LANGUAGES",
    "ensure_async",
    "model_dir",
    "model_ready",
    "model_root",
]

#: Languages with a bundled/downloadable DiffSinger model.
LANGUAGES = frozenset({"zh", "en"})
#: All model directories: language models plus the male English voicebank.
MODELS = frozenset({"zh", "en", "en_male"})

_ZH_URL = (
    "https://github.com/HighCWu/DiffSinger-ONNX-Infer/releases/download/v0.1/model.zip"
)
_ZH_SHA256 = "598739ffde0b10a51dfef74ab0ec1eb2820941f4b81ada64079b733e895c9c77"
_EN_URL = (
    "https://github.com/spicytigermeat/tiger_diffsinger/releases/download/v106/"
    "TIGER_DS_v106_PACK.zip"
)
_EN_SHA256 = "b19d498f46f05f58f049ec2ec0ee2b569a01a4fbda41667922d7c773f1196c19"
_EN_MALE_URL = (
    "https://github.com/theaimeltus/DiffSinger-Ameko-Kero-Legacy-Archive-/"
    "releases/download/v1.3-MbPa0.8-Patisserie/"
    "AmekoKero_ver1.3_MbPa0.8_PatisserieBuild.zip"
)
_EN_MALE_SHA256 = "284e631df71b3710456d8abe5463f0671f8f0c0521dce4ad8304db7d812e4422"

_ZH_COMPLETE = ("config.yaml", "fs2.onnx", "p_sample.onnx", "vocoder.onnx")
_EN_COMPLETE = (
    "dsconfig.yaml",
    "dsacoustic/acoustic.onnx",
    "dsvocoder/tgm_hifigan.onnx",
)
_EN_MALE_COMPLETE = (
    "dsconfig.yaml",
    "onnx/millefeuille_b8.onnx",
    "dsvocoder/millefeuille_hifigan.onnx",
    "lang/millefeuille_b8.phonemes.json",
    "embs/millefeuille_b8.kero_neutral.emb",
)

_LOCK = threading.Lock()
_STARTED: set[str] = set()


def model_root() -> Path:
    """Return the DiffSinger model root directory.

    Returns:
        The model root, honouring ``HARMONIA_DIFFSINGER_DIR``.
    """
    override = os.environ.get("HARMONIA_DIFFSINGER_DIR")
    if override:
        return Path(override)
    return config.config_dir() / "diffsinger"


def model_dir(language: str) -> Path:
    """Return the directory of one model.

    Args:
        language: Model key (``zh``, ``en`` or ``en_male``).

    Returns:
        The model directory path (which may not exist yet).
    """
    return model_root() / language


def model_ready(language: str) -> bool:
    """Return whether a model is fully downloaded.

    Args:
        language: Model key (``zh``, ``en`` or ``en_male``).

    Returns:
        ``True`` when all required model files exist.
    """
    if language not in MODELS:
        return False
    required = (
        _ZH_COMPLETE
        if language == "zh"
        else _EN_MALE_COMPLETE
        if language == "en_male"
        else _EN_COMPLETE
    )
    base = model_dir(language)
    return all((base / relative).is_file() for relative in required)


def ensure_async() -> None:
    """Start background downloads for models that are missing on disk.

    This is deliberately presence-based rather than first-run based: every app
    start (and any explicit call) checks the actual files, so a deleted or
    partially downloaded model is fetched again.  Repeated calls are cheap and
    failures are swallowed; the next call retries after a transient failure.
    """
    for language in sorted(MODELS):
        if model_ready(language):
            continue
        with _LOCK:
            if language in _STARTED:
                continue
            _STARTED.add(language)
        thread = threading.Thread(
            target=_safe_download,
            args=(language,),
            name=f"diffsinger-{language}",
            daemon=True,
        )
        thread.start()


def _safe_download(language: str) -> None:
    """Download one model, ignoring failures and clearing the in-flight mark.

    Args:
        language: Language tag to download.
    """
    try:
        download(language)
    except Exception:  # noqa: BLE001, S110 - optional feature must never crash
        pass
    finally:
        with _LOCK:
            _STARTED.discard(language)


def download(language: str) -> Path:
    """Download, verify and extract one DiffSinger model synchronously.

    Args:
        language: Model key (``zh``, ``en`` or ``en_male``).

    Returns:
        The extracted model directory.

    Raises:
        ValueError: For an unknown language or a failed checksum.
    """
    if language not in MODELS:
        raise ValueError(f"Unsupported DiffSinger model: {language}")
    if model_ready(language):
        return model_dir(language)
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{language}.zip.part"
    url = _ZH_URL if language == "zh" else _EN_URL if language == "en" else _EN_MALE_URL
    expected = (
        _ZH_SHA256
        if language == "zh"
        else _EN_SHA256
        if language == "en"
        else _EN_MALE_SHA256
    )
    try:
        urllib.request.urlretrieve(url, archive)  # fixed https URL
        if _sha256(archive) != expected:
            raise ValueError(f"DiffSinger model checksum mismatch: {language}")
        target = model_dir(language)
        staging = root / f"{language}.unpack"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        if language == "zh":
            _extract_zh(archive, staging)
        elif language == "en":
            _extract_en(archive, staging)
        else:
            _extract_en_male(archive, staging)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        return target
    finally:
        archive.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zh(archive: Path, staging: Path) -> None:
    """Extract the OpenCpop zip into a flat model directory.

    Args:
        archive: Downloaded zip.
        staging: Destination staging directory.
    """
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            name = Path(member).name
            if not name:
                continue
            (staging / name).write_bytes(bundle.read(member))


def _extract_en(archive: Path, staging: Path) -> None:
    """Extract the Tiger pack and its nested voicebank zip.

    Args:
        archive: Downloaded outer zip.
        staging: Destination staging directory.
    """
    with zipfile.ZipFile(archive) as bundle:
        inner_name = next(
            (
                member
                for member in bundle.namelist()
                if member.replace("\\", "/").endswith("Voice Library/TIGER_DS_v106.zip")
            ),
            None,
        )
        if inner_name is None:
            raise ValueError("Tiger pack does not contain the voicebank zip")
        inner = staging.parent / "tiger-inner.zip"
        inner.write_bytes(bundle.read(inner_name))
    try:
        with zipfile.ZipFile(inner) as voicebank:
            voicebank.extractall(staging)
    finally:
        inner.unlink(missing_ok=True)


def _extract_en_male(archive: Path, staging: Path) -> None:
    """Extract the Ameko Kero voicebank, stripping its top-level folder.

    Args:
        archive: Downloaded zip.
        staging: Destination staging directory.
    """
    with zipfile.ZipFile(archive) as bundle:
        members = [
            member.replace("\\", "/")
            for member in bundle.namelist()
            if not member.endswith("/")
        ]
        if not members:
            raise ValueError("Ameko Kero pack is empty")
        first = members[0]
        prefix = f"{first.split('/', 1)[0]}/" if "/" in first else ""
        for member in members:
            relative = member.removeprefix(prefix)
            if not relative or relative.startswith("__MACOSX/"):
                continue
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
