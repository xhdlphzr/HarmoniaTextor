# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the optional DiffSinger model downloader."""

from __future__ import annotations

import io
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Self

import pytest

from harmoniatextor.render.vocal import diffsinger_models


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    """Write a zip archive with the given members."""
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)


def _zh_archive(path: Path) -> None:
    """Write a minimal OpenCpop-style archive."""
    _write_zip(
        path,
        {
            "model/": b"",
            "model/config.yaml": b"audio_sample_rate: 24000\n",
            "model/fs2.onnx": b"fs2",
            "model/p_sample.onnx": b"p",
            "model/vocoder.onnx": b"vocoder",
        },
    )


def _en_archive(path: Path) -> None:
    """Write a minimal Tiger pack with a nested voicebank archive."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as bundle:
        bundle.writestr("dsconfig.yaml", b"config")
        bundle.writestr("dsacoustic/acoustic.onnx", b"acoustic")
        bundle.writestr("dsvocoder/tgm_hifigan.onnx", b"vocoder")
    _write_zip(
        path,
        {
            "Pack/Voice Library/TIGER_DS_v106.zip": inner.getvalue(),
        },
    )


def _male_archive(path: Path) -> None:
    """Write a minimal Ameko Kero archive with a top-level folder."""
    _write_zip(
        path,
        {
            "AmekoKero/dsconfig.yaml": b"config",
            "AmekoKero/onnx/millefeuille_b8.onnx": b"acoustic",
            "AmekoKero/dsvocoder/millefeuille_hifigan.onnx": b"vocoder",
            "AmekoKero/lang/millefeuille_b8.phonemes.json": b"{}",
            "AmekoKero/embs/millefeuille_b8.kero_neutral.emb": b"emb",
            "AmekoKero/__MACOSX/junk": b"junk",
        },
    )


def test_model_root_honours_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model root can be redirected for tests and portable installs."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER_DIR", str(tmp_path / "models"))
    assert diffsinger_models.model_root() == tmp_path / "models"
    monkeypatch.delenv("HARMONIA_DIFFSINGER_DIR")
    assert diffsinger_models.model_root().name == "diffsinger"


def test_model_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Readiness requires every documented model file."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER_DIR", str(tmp_path))
    assert not diffsinger_models.model_ready("fr")
    assert not diffsinger_models.model_ready("zh")
    base = tmp_path / "zh"
    base.mkdir()
    for name in ("config.yaml", "fs2.onnx", "p_sample.onnx"):
        (base / name).write_text("x", encoding="utf-8")
    assert not diffsinger_models.model_ready("zh")
    (base / "vocoder.onnx").write_text("x", encoding="utf-8")
    assert diffsinger_models.model_ready("zh")
    assert not diffsinger_models.model_ready("en")


def test_download_zh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Chinese model archive is verified and extracted."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER_DIR", str(tmp_path))
    (tmp_path / "zh").mkdir()
    (tmp_path / "zh" / "stale").write_text("old", encoding="utf-8")
    source = tmp_path / "zh.zip"
    _zh_archive(source)
    monkeypatch.setattr(
        diffsinger_models, "_sha256", lambda _path: diffsinger_models._ZH_SHA256
    )

    def fake_urlretrieve(_url: str, target: Path) -> None:
        target.write_bytes(source.read_bytes())

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    result = diffsinger_models.download("zh")
    assert result == tmp_path / "zh"
    assert (result / "fs2.onnx").read_bytes() == b"fs2"
    assert not (tmp_path / "zh.zip.part").exists()


def test_download_en(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Tiger pack's nested voicebank zip is extracted."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER_DIR", str(tmp_path))
    source = tmp_path / "en.zip"
    _en_archive(source)
    monkeypatch.setattr(
        diffsinger_models, "_sha256", lambda _path: diffsinger_models._EN_SHA256
    )
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda _url, target: target.write_bytes(source.read_bytes()),
    )
    result = diffsinger_models.download("en")
    assert result == tmp_path / "en"
    assert (result / "dsacoustic/acoustic.onnx").read_bytes() == b"acoustic"
    assert not (tmp_path / "tiger-inner.zip").exists()


def test_download_male(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ameko Kero male voicebank is flattened into its model directory."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER_DIR", str(tmp_path))
    source = tmp_path / "male.zip"
    _male_archive(source)
    monkeypatch.setattr(
        diffsinger_models, "_sha256", lambda _path: diffsinger_models._EN_MALE_SHA256
    )
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda _url, target: target.write_bytes(source.read_bytes()),
    )
    result = diffsinger_models.download("en_male")
    assert result == tmp_path / "en_male"
    assert (result / "onnx/millefeuille_b8.onnx").read_bytes() == b"acoustic"
    assert not (result / "__MACOSX").exists()


def test_download_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown languages, bad checksums and malformed packs are rejected."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        diffsinger_models.download("fr")
    source = tmp_path / "zh.zip"
    _zh_archive(source)
    monkeypatch.setattr(diffsinger_models, "_sha256", lambda _path: "bad")
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda _url, target: target.write_bytes(source.read_bytes()),
    )
    with pytest.raises(ValueError, match="checksum"):
        diffsinger_models.download("zh")
    no_voicebank = tmp_path / "bad.zip"
    _write_zip(no_voicebank, {"Pack/readme.txt": b"x"})
    monkeypatch.setattr(
        diffsinger_models, "_sha256", lambda _path: diffsinger_models._EN_SHA256
    )
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda _url, target: target.write_bytes(no_voicebank.read_bytes()),
    )
    with pytest.raises(ValueError, match="voicebank"):
        diffsinger_models.download("en")
    empty = tmp_path / "empty.zip"
    _write_zip(empty, {})
    monkeypatch.setattr(
        diffsinger_models, "_sha256", lambda _path: diffsinger_models._EN_MALE_SHA256
    )
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda _url, target: target.write_bytes(empty.read_bytes()),
    )
    with pytest.raises(ValueError, match="empty"):
        diffsinger_models.download("en_male")


def test_download_short_circuits_ready_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Downloading an already installed model does nothing."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER_DIR", str(tmp_path))
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: True)
    assert diffsinger_models.download("zh") == tmp_path / "zh"


def test_sha256(tmp_path: Path) -> None:
    """The file digest is lowercase hexadecimal."""
    path = tmp_path / "file"
    path.write_bytes(b"hello")
    assert diffsinger_models._sha256(path) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_extract_zh_skips_empty_member_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zip entries without a basename are ignored."""

    class FakeBundle:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def namelist(self) -> list[str]:
            return ["", "config.yaml"]

        def read(self, member: str) -> bytes:
            return member.encode()

    monkeypatch.setattr(zipfile, "ZipFile", lambda _path: FakeBundle())
    staging = tmp_path / "staging"
    staging.mkdir()
    diffsinger_models._extract_zh(tmp_path / "missing.zip", staging)
    assert (staging / "config.yaml").read_bytes() == b"config.yaml"


def test_ensure_async(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing models are downloaded once through background threads."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER_DIR", str(tmp_path))
    monkeypatch.setattr(diffsinger_models, "_STARTED", set())
    calls: list[str] = []

    class FakeThread:
        def __init__(self, target: Any, args: tuple[Any, ...], **_kwargs: Any) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(threading, "Thread", FakeThread)
    ready: set[str] = set()

    def fake_download(language: str) -> None:
        calls.append(language)
        ready.add(language)

    monkeypatch.setattr(diffsinger_models, "download", fake_download)
    monkeypatch.setattr(
        diffsinger_models, "model_ready", lambda language: language in ready
    )

    diffsinger_models.ensure_async()
    diffsinger_models.ensure_async()
    assert calls == ["en", "en_male", "zh"]

    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: True)
    diffsinger_models.ensure_async()


def test_ensure_async_skips_inflight_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A language already being downloaded is not started twice."""
    monkeypatch.setattr(diffsinger_models, "_STARTED", {"zh"})
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: False)
    calls: list[str] = []

    class FakeThread:
        def __init__(self, target: Any, args: tuple[Any, ...], **_kwargs: Any) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            calls.append(self.args[0])

    monkeypatch.setattr(threading, "Thread", FakeThread)
    diffsinger_models.ensure_async()
    assert calls == ["en", "en_male"]


def test_safe_download_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed background download never escapes and can be retried."""
    monkeypatch.setattr(diffsinger_models, "_STARTED", {"zh"})

    def boom(_language: str) -> None:
        raise OSError("network down")

    monkeypatch.setattr(diffsinger_models, "download", boom)
    diffsinger_models._safe_download("zh")
    assert diffsinger_models._STARTED == set()
