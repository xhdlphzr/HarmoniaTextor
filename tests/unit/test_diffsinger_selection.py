# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for automatic DiffSinger/FOF backend selection."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from harmoniatextor.render.vocal import diffsinger, diffsinger_models
from harmoniatextor.render.vocal.singer import SungNote


class FakeBackend:
    """A backend returning a fixed buffer."""

    def __init__(
        self, value: float = 0.25, *, raise_error: bool = False, empty: bool = False
    ) -> None:
        """Initialise the fake backend.

        Args:
            value: Sample value to return.
            raise_error: Whether to raise instead of returning samples.
            empty: Whether to return an empty buffer.
        """
        self.value = value
        self.raise_error = raise_error
        self.empty = empty
        self.calls = 0

    def synthesize(self, _notes: object) -> np.ndarray:
        """Return the fixed buffer."""
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("model failed")
        if self.empty:
            return np.zeros(0, dtype=np.float64)
        return np.full(10, self.value)


def _note(language: str = "en", text: str = "love", midi: int = 60) -> SungNote:
    """Build a minimal sung note."""
    return SungNote(
        midi=midi, onset=0.0, duration=0.5, ipa="lʌv", text=text, language=language
    )


def test_normalise_language() -> None:
    """Language aliases are normalised."""
    assert diffsinger.normalise_language("en-US") == "en"
    assert diffsinger.normalise_language("zh_CN") == "zh"
    assert diffsinger.normalise_language("cmn") == "zh"
    assert diffsinger.normalise_language("fr") == ""


def test_male_max_midi(monkeypatch: pytest.MonkeyPatch) -> None:
    """The male/female pitch threshold is configurable."""
    monkeypatch.delenv("HARMONIA_DIFFSINGER_MALE_MAX_MIDI", raising=False)
    assert diffsinger._male_max_midi() == 64.0
    monkeypatch.setenv("HARMONIA_DIFFSINGER_MALE_MAX_MIDI", "127")
    assert diffsinger._male_max_midi() == 127.0
    monkeypatch.setenv("HARMONIA_DIFFSINGER_MALE_MAX_MIDI", "nope")
    assert diffsinger._male_max_midi() == 64.0


def test_english_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional Tiger steps/speaker settings are read safely."""
    monkeypatch.delenv("HARMONIA_DIFFSINGER_STEPS", raising=False)
    monkeypatch.delenv("HARMONIA_DIFFSINGER_SPEAKER", raising=False)
    assert diffsinger._english_options() == {}
    monkeypatch.setenv("HARMONIA_DIFFSINGER_STEPS", "24")
    monkeypatch.setenv("HARMONIA_DIFFSINGER_SPEAKER", "tiger_royal")
    assert diffsinger._english_options() == {
        "steps": 24,
        "speaker": "tiger_royal",
    }
    monkeypatch.setenv("HARMONIA_DIFFSINGER_STEPS", "nope")
    assert diffsinger._english_options() == {"speaker": "tiger_royal"}
    monkeypatch.setenv("HARMONIA_DIFFSINGER_STEPS", "0")
    assert diffsinger._english_options()["steps"] == 1


def test_load_passes_english_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The English backend receives the optional quality settings."""
    recorded: dict[str, object] = {}

    class OptionsFake:
        def __init__(self, model_dir: Path, **kwargs: object) -> None:
            recorded["dir"] = model_dir
            recorded.update(kwargs)

    _install_fake_backend(monkeypatch, "en", OptionsFake)
    monkeypatch.setattr(diffsinger_models, "model_dir", lambda language: Path(language))
    monkeypatch.setenv("HARMONIA_DIFFSINGER_STEPS", "20")
    monkeypatch.setenv("HARMONIA_DIFFSINGER_SPEAKER", "tiger_mystic")
    assert diffsinger.AutoVocalBackend()._load("en") is not None
    assert recorded == {
        "dir": Path("en"),
        "steps": 20,
        "speaker": "tiger_mystic",
    }


def test_load_male_strips_tiger_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Tiger speaker name is not passed to the male voicebank."""
    recorded: dict[str, object] = {}

    class OptionsFake:
        def __init__(self, model_dir: Path, **kwargs: object) -> None:
            recorded["dir"] = model_dir
            recorded.update(kwargs)

    _install_fake_backend(monkeypatch, "en", OptionsFake)
    monkeypatch.setattr(diffsinger_models, "model_dir", lambda language: Path(language))
    monkeypatch.setenv("HARMONIA_DIFFSINGER_STEPS", "20")
    monkeypatch.setenv("HARMONIA_DIFFSINGER_SPEAKER", "tiger_mystic")
    assert diffsinger.AutoVocalBackend()._load("en_male") is not None
    assert recorded == {"dir": Path("en_male"), "steps": 20}


def test_performance_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The feature can be disabled or gated by CPU count."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER", "0")
    assert not diffsinger._performance_ok()
    monkeypatch.setenv("HARMONIA_DIFFSINGER", "auto")
    monkeypatch.setenv("HARMONIA_DIFFSINGER_MIN_CORES", "1")
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert diffsinger._performance_ok()
    monkeypatch.setenv("HARMONIA_DIFFSINGER_MIN_CORES", "nope")
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert diffsinger._performance_ok()
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert not diffsinger._performance_ok()


def test_model_key_covers_chinese_and_male_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model selection handles Chinese and a male-only English install."""
    backend = diffsinger.AutoVocalBackend()
    monkeypatch.setattr(
        diffsinger_models, "model_ready", lambda language: language == "zh"
    )
    assert backend._model_key([_note("zh", text="你")], "zh") == "zh"
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: False)
    assert backend._model_key([_note("zh", text="你")], "zh") == ""
    monkeypatch.setattr(
        diffsinger_models, "model_ready", lambda language: language == "en_male"
    )
    assert backend._model_key([_note("en", midi=68)], "en") == "en_male"


def test_empty_notes() -> None:
    """Empty note lists render silence without loading a model."""
    assert diffsinger.AutoVocalBackend().synthesize([]).size == 0


def test_mixed_or_unsupported_language_falls_back() -> None:
    """Mixed languages and unsupported tags use the FOF fallback."""
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend()
    assert backend.synthesize([_note("en"), _note("zh")]).size > 0
    assert backend.synthesize([_note("fr")]).size > 0
    assert backend.synthesize([_note("en", text="")]).size > 0


def test_performance_disabled_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disabled DiffSinger feature uses FOF."""
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend()
    monkeypatch.setenv("HARMONIA_DIFFSINGER", "off")
    assert backend.synthesize([_note("en")]).size > 0


def test_missing_model_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing model uses FOF; downloads are started by app startup instead."""
    monkeypatch.setattr(diffsinger, "_performance_ok", lambda: True)
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: False)
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend()
    assert backend.synthesize([_note("en")]).size > 0


def test_ready_model_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ready model is loaded and its samples are returned."""
    monkeypatch.setattr(diffsinger, "_performance_ok", lambda: True)
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: True)
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend()
    fake = FakeBackend(value=0.5)
    monkeypatch.setattr(backend, "_load", lambda _language: fake)
    samples = backend.synthesize([_note("en")])
    assert samples[0] == 0.5
    assert fake.calls == 1


def test_low_voice_uses_male_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A low English voice selects the male voicebank."""
    monkeypatch.setattr(diffsinger, "_performance_ok", lambda: True)
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: True)
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend()
    calls: list[str] = []

    def load(key: str) -> FakeBackend:
        calls.append(key)
        return FakeBackend(value=0.4)

    monkeypatch.setattr(backend, "_load", load)
    assert backend.synthesize([_note("en", midi=50)])[0] == 0.4
    assert calls == ["en_male"]


def test_male_model_missing_uses_tiger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the male model a low voice falls back to the Tiger model."""
    monkeypatch.setattr(diffsinger, "_performance_ok", lambda: True)
    monkeypatch.setattr(
        diffsinger_models,
        "model_ready",
        lambda language: language in {"en", "en_male"} and language == "en",
    )
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend()
    calls: list[str] = []

    def load(key: str) -> FakeBackend:
        calls.append(key)
        return FakeBackend(value=0.3)

    monkeypatch.setattr(backend, "_load", load)
    backend.synthesize([_note("en", midi=50)])
    assert calls == ["en"]


def test_male_model_failure_uses_tiger(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken male model falls through to the Tiger model."""
    monkeypatch.setattr(diffsinger, "_performance_ok", lambda: True)
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: True)
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend(value=0.1)
    calls: list[str] = []

    def load(key: str) -> FakeBackend | None:
        calls.append(key)
        if key == "en_male":
            return None
        return FakeBackend(value=0.6)

    monkeypatch.setattr(backend, "_load", load)
    assert backend.synthesize([_note("en", midi=50)])[0] == 0.6
    assert calls == ["en_male", "en"]


def test_chinese_uses_male_ameko(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chinese lines use the Ameko male model when it is installed."""
    monkeypatch.setattr(diffsinger, "_performance_ok", lambda: True)
    monkeypatch.setattr(
        diffsinger_models, "model_ready", lambda language: language == "en_male"
    )
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend()
    calls: list[str] = []

    def load(key: str) -> FakeBackend:
        calls.append(key)
        return FakeBackend(value=0.45)

    monkeypatch.setattr(backend, "_load", load)
    assert backend.synthesize([_note("zh", text="你", midi=50)])[0] == 0.45
    assert calls == ["zh_male"]


def test_load_chinese_male_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Chinese male backend is constructed with Ameko's model directory."""
    recorded: dict[str, object] = {}

    class Fake:
        def __init__(self, model_dir: Path, **kwargs: object) -> None:
            recorded["dir"] = model_dir
            recorded.update(kwargs)

    module = types.ModuleType("harmoniatextor.render.vocal.diffsinger_zh_ameko")
    backend_name = "AmekoChineseDiffSingerBackend"
    setattr(module, backend_name, Fake)
    monkeypatch.setitem(
        sys.modules, "harmoniatextor.render.vocal.diffsinger_zh_ameko", module
    )
    monkeypatch.setattr(diffsinger_models, "model_dir", lambda language: Path(language))
    monkeypatch.setenv("HARMONIA_DIFFSINGER_STEPS", "16")
    monkeypatch.setenv("HARMONIA_DIFFSINGER_SPEAKER", "tiger_royal")
    assert diffsinger.AutoVocalBackend()._load("zh_male") is not None
    assert recorded == {"dir": Path("en_male"), "steps": 16}


def test_no_english_model_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """No installed English model uses FOF."""
    monkeypatch.setattr(diffsinger, "_performance_ok", lambda: True)
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: False)
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend(value=0.1)
    assert backend.synthesize([_note("en", midi=50)])[0] == 0.1


@pytest.mark.parametrize(
    "fake",
    [
        FakeBackend(raise_error=True),
        FakeBackend(value=float("nan")),
    ],
)
def test_model_failure_falls_back(
    fake: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model errors or invalid buffers fall back to FOF."""
    monkeypatch.setattr(diffsinger, "_performance_ok", lambda: True)
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: True)
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend(value=0.1)
    monkeypatch.setattr(backend, "_load", lambda _language: fake)
    samples = backend.synthesize([_note("en")])
    assert samples[0] == 0.1


def test_model_empty_buffer_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty model buffer falls back to FOF."""
    monkeypatch.setattr(diffsinger, "_performance_ok", lambda: True)
    monkeypatch.setattr(diffsinger_models, "model_ready", lambda _language: True)
    backend = diffsinger.AutoVocalBackend()
    backend._fallback = FakeBackend(value=0.1)
    monkeypatch.setattr(backend, "_load", lambda _language: FakeBackend(empty=True))
    assert backend.synthesize([_note("en")])[0] == 0.1


def test_male_options_strips_tiger_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ameko male backend only receives step/quality options."""
    monkeypatch.setenv("HARMONIA_DIFFSINGER_STEPS", "16")
    monkeypatch.setenv("HARMONIA_DIFFSINGER_SPEAKER", "tiger_royal")
    assert diffsinger._male_options() == {"steps": 16}


def _install_fake_backend(
    monkeypatch: pytest.MonkeyPatch, language: str, factory: object
) -> None:
    """Install a fake backend module for one language."""
    module_name = f"harmoniatextor.render.vocal.diffsinger_{language}"
    module = types.ModuleType(module_name)
    class_name = (
        "ChineseDiffSingerBackend" if language == "zh" else "EnglishDiffSingerBackend"
    )
    setattr(module, class_name, factory)
    monkeypatch.setitem(sys.modules, module_name, module)


def test_load_success_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Language backends are discovered, constructed once and cached."""
    instances: list[Any] = []

    class Fake:
        def __init__(self, model_dir: Path) -> None:
            self.model_dir = model_dir
            instances.append(self)

    _install_fake_backend(monkeypatch, "zh", Fake)
    _install_fake_backend(monkeypatch, "en", Fake)
    monkeypatch.setattr(diffsinger_models, "model_dir", lambda language: Path(language))
    backend = diffsinger.AutoVocalBackend()
    zh = backend._load("zh")
    assert zh is backend._load("zh")
    en = backend._load("en")
    assert en is not None
    assert len(instances) == 2
    assert instances[0].model_dir == Path("zh")


def test_load_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing/broken optional backend is reported as unavailable."""
    monkeypatch.setitem(sys.modules, "harmoniatextor.render.vocal.diffsinger_zh", None)
    assert diffsinger.AutoVocalBackend()._load("zh") is None
    monkeypatch.setitem(sys.modules, "harmoniatextor.render.vocal.diffsinger_en", None)
    assert diffsinger.AutoVocalBackend()._load("en") is None


def test_load_constructor_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that raises while loading is unavailable."""

    class Broken:
        def __init__(self, _model_dir: Path) -> None:
            raise RuntimeError("bad model")

    _install_fake_backend(monkeypatch, "zh", Broken)
    _install_fake_backend(monkeypatch, "en", Broken)
    assert diffsinger.AutoVocalBackend()._load("zh") is None
    assert diffsinger.AutoVocalBackend()._load("en") is None
