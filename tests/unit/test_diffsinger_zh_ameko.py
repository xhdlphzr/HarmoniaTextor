# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for the Ameko Kero Chinese DiffSinger backend."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from harmoniatextor.render.vocal import diffsinger_zh_ameko
from harmoniatextor.render.vocal.diffsinger_en import (
    HOP_SIZE,
    _load_dictionary,
    _seconds_to_frames,
)
from harmoniatextor.render.vocal.diffsinger_zh_ameko import (
    DEFAULT_DEPTH,
    DEFAULT_STEPS,
    AmekoChineseDiffSingerBackend,
    _language_path,
    _phones_for_text,
    _pinyin_syllables,
)
from harmoniatextor.render.vocal.singer import SAMPLE_RATE, SungNote

#: Chinese phones used by the fake voicebank.
_ZH_PHONES = (
    "zh/a",
    "zh/ao",
    "zh/cl",
    "zh/e",
    "zh/h",
    "zh/i",
    "zh/iE",
    "zh/ir",
    "zh/j",
    "zh/n",
    "zh/sh",
)
#: Fake phoneme table mapping symbol to token id.
_MODERN_PHONEMES: dict[str, int] = {
    "AP": 0,
    "SP": 1,
    **{phone: index for index, phone in enumerate(_ZH_PHONES, start=2)},
}
#: Fake modern acoustic model inputs.
_MODERN_INPUTS = (
    "tokens",
    "languages",
    "durations",
    "f0",
    "gender",
    "velocity",
    "spk_embed",
    "depth",
    "steps",
)
#: Fake acoustic model inputs without the language embedding.
_LEGACY_INPUTS = tuple(name for name in _MODERN_INPUTS if name != "languages")
#: Fake root configuration of the Ameko layout.
_CONFIG = """acoustic: onnx/millefeuille_b8.onnx
phonemes: lang/millefeuille_b8.phonemes.json
languages: lang/millefeuille_b8.languages.json
use_lang_id: true
vocoder: dsvocoder
speakers:
- embs/millefeuille_b8.kero_fly
- embs/millefeuille_b8.kero_neutral
"""
#: Fake Chinese pinyin dictionary in the Ameko voicebank format.
_DICTIONARY = """entries:
- {grapheme: "a", phonemes: [zh/a]}
- {grapheme: "ni", phonemes: [zh/n, zh/i]}
- {grapheme: "hao", phonemes: [zh/h, zh/ao]}
- {grapheme: "shi", phonemes: [zh/sh, zh/ir]}
- {grapheme: "jie", phonemes: [zh/j, zh/iE]}
- {grapheme: "-", phonemes: [SP]}
- {grapheme: "cl", phonemes: [cl]}
replacements:
- {from: cl, to: zh/cl}
- {from: a, to: zh/a}
"""


class _SessionInput:
    """Minimal stand-in for an ONNX Runtime model input."""

    def __init__(self, name: str) -> None:
        """Store the input name.

        Args:
            name: The input tensor name.
        """
        self.name = name


class _FakeAcoustic:
    """Fake acoustic model returning a constant mel spectrogram."""

    def __init__(self, inputs: Sequence[str] = _MODERN_INPUTS) -> None:
        """Initialise the recorder and its input names.

        Args:
            inputs: Names reported by :meth:`get_inputs`.
        """
        self.feeds: list[dict[str, NDArray[Any]]] = []
        self.names = tuple(inputs)
        self.non_finite = False

    def get_inputs(self) -> list[_SessionInput]:
        """Return the configured model inputs.

        Returns:
            One fake input per configured name.
        """
        return [_SessionInput(name) for name in self.names]

    def run(
        self,
        output_names: Sequence[str],
        feed: dict[str, NDArray[Any]],
    ) -> list[NDArray[np.float32]]:
        """Record the feed and return a mel spectrogram.

        Args:
            output_names: Requested output names.
            feed: Acoustic model inputs.

        Returns:
            A constant ``[1, frames, 128]`` mel spectrogram.
        """
        assert list(output_names) == ["mel"]
        self.feeds.append(feed)
        frames = int(np.sum(feed["durations"]))
        mel = np.full((1, frames, 128), 0.25, dtype=np.float32)
        if self.non_finite:
            mel[0, 0, 0] = np.nan
            mel[0, 0, 1] = np.inf
        return [mel]


class _FakeVocoder:
    """Fake HiFi-GAN vocoder returning a constant waveform."""

    def __init__(self) -> None:
        """Initialise the recorder."""
        self.feeds: list[dict[str, NDArray[Any]]] = []
        self.value = 0.5
        self.non_finite = False

    def run(
        self,
        output_names: Sequence[str],
        feed: dict[str, NDArray[Any]],
    ) -> list[NDArray[np.float32]]:
        """Record the feed and return a waveform.

        Args:
            output_names: Requested output names.
            feed: Vocoder inputs.

        Returns:
            A constant ``[1, frames * hop]`` waveform.
        """
        assert list(output_names) == ["waveform"]
        self.feeds.append(feed)
        frames = int(feed["mel"].shape[1])
        wave = np.full((1, frames * HOP_SIZE), self.value, dtype=np.float32)
        if self.non_finite:
            wave[0, 0] = np.nan
            wave[0, 1] = -np.inf
        return [wave]


def _install_sessions(
    monkeypatch: pytest.MonkeyPatch,
    acoustic: _FakeAcoustic,
    vocoder: _FakeVocoder,
) -> list[Path]:
    """Replace the ONNX session factory with the fake models.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        acoustic: Fake acoustic session.
        vocoder: Fake vocoder session.

    Returns:
        The model paths passed to the factory, in call order.
    """
    created: list[Path] = []

    def factory(path: Path) -> _FakeAcoustic | _FakeVocoder:
        created.append(path)
        return acoustic if path.parent.name != "dsvocoder" else vocoder

    monkeypatch.setattr(diffsinger_zh_ameko, "_create_session", factory)
    return created


def _embedding(value: float) -> bytes:
    """Return a 256-float little-endian speaker embedding.

    Args:
        value: Value repeated in every component.

    Returns:
        The raw embedding bytes.
    """
    return np.full(256, value, dtype="<f4").tobytes()


@pytest.fixture
def ameko_voicebank(tmp_path: Path) -> Path:
    """Create a minimal fake Ameko Kero voicebank.

    Args:
        tmp_path: The pytest temporary path fixture.

    Returns:
        The voicebank root.
    """
    base = tmp_path / "ameko"
    (base / "onnx").mkdir(parents=True)
    (base / "onnx" / "millefeuille_b8.onnx").write_bytes(b"acoustic")
    lang = base / "lang"
    lang.mkdir()
    (lang / "millefeuille_b8.phonemes.json").write_text(
        json.dumps(_MODERN_PHONEMES), encoding="utf-8"
    )
    (lang / "millefeuille_b8.languages.json").write_text(
        json.dumps({"en": 2, "zh": 13}), encoding="utf-8"
    )
    durations = base / "dsdur"
    durations.mkdir()
    (durations / "dsdict-zh.yaml").write_text(_DICTIONARY, encoding="utf-8")
    vocoder = base / "dsvocoder"
    vocoder.mkdir()
    (vocoder / "generic_hifigan.onnx").write_bytes(b"generic")
    (vocoder / "millefeuille_hifigan.onnx").write_bytes(b"vocoder")
    embs = base / "embs"
    embs.mkdir()
    (embs / "millefeuille_b8.kero_fly.emb").write_bytes(_embedding(0.375))
    (embs / "millefeuille_b8.kero_neutral.emb").write_bytes(_embedding(0.125))
    (base / "dsconfig.yaml").write_text(_CONFIG, encoding="utf-8")
    return base


@pytest.fixture
def backend_pair(
    monkeypatch: pytest.MonkeyPatch,
    ameko_voicebank: Path,
) -> tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder]:
    """Build a Chinese Ameko backend wired to fake ONNX sessions.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.

    Returns:
        The backend, its fake acoustic session and its fake vocoder.
    """
    acoustic = _FakeAcoustic()
    vocoder = _FakeVocoder()
    _install_sessions(monkeypatch, acoustic, vocoder)
    return AmekoChineseDiffSingerBackend(ameko_voicebank), acoustic, vocoder


def _note(midi: int, onset: float, duration: float, text: str) -> SungNote:
    """Build a Chinese sung note.

    Args:
        midi: MIDI pitch number.
        onset: Start time in seconds.
        duration: Duration in seconds.
        text: Chinese syllable text.

    Returns:
        The sung note.
    """
    return SungNote(
        midi=midi,
        onset=onset,
        duration=duration,
        ipa="",
        text=text,
        language="zh",
    )


def test_create_session_without_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """Building a backend without onnxruntime raises a clear error.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(RuntimeError, match="onnxruntime"):
        AmekoChineseDiffSingerBackend(ameko_voicebank)


def test_backend_loads_voicebank(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """The root configuration resolves models, dictionary, speaker and language.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    created = _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    backend = AmekoChineseDiffSingerBackend(ameko_voicebank)
    assert created == [
        ameko_voicebank / "onnx" / "millefeuille_b8.onnx",
        ameko_voicebank / "dsvocoder" / "millefeuille_hifigan.onnx",
    ]
    assert backend._phoneme_ids["zh/a"] == _MODERN_PHONEMES["zh/a"]
    assert backend._dictionary.words["ni"] == ["zh/n", "zh/i"]
    assert backend._dictionary.replacements["cl"] == "zh/cl"
    assert backend._symbol_prefix == "zh/"
    assert backend._language_id == 13
    assert backend._steps == DEFAULT_STEPS
    assert backend._depth == DEFAULT_DEPTH
    assert np.allclose(backend._speaker, 0.125)


def test_vocoder_selection_prefers_matching_model(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """The voicebank's own vocoder wins over an unrelated generic model.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    created = _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    AmekoChineseDiffSingerBackend(ameko_voicebank)
    assert created[-1].name == "millefeuille_hifigan.onnx"


def test_default_layout_without_root_config(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """The Ameko layout is found even without a root dsconfig.yaml.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    (ameko_voicebank / "dsconfig.yaml").unlink()
    created = _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    backend = AmekoChineseDiffSingerBackend(ameko_voicebank)
    assert created == [
        ameko_voicebank / "onnx" / "millefeuille_b8.onnx",
        ameko_voicebank / "dsvocoder" / "millefeuille_hifigan.onnx",
    ]
    assert backend._language_id == 13


def test_config_must_be_mapping(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """A root configuration without a mapping is rejected.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    (ameko_voicebank / "dsconfig.yaml").write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        AmekoChineseDiffSingerBackend(ameko_voicebank)


def test_missing_chinese_language(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """A language table without Chinese is rejected.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    (ameko_voicebank / "lang" / "millefeuille_b8.languages.json").write_text(
        json.dumps({"en": 2}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Chinese"):
        AmekoChineseDiffSingerBackend(ameko_voicebank)


def test_legacy_model_without_language_input(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """A model without a languages input is fed without the language array.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    acoustic = _FakeAcoustic(_LEGACY_INPUTS)
    _install_sessions(monkeypatch, acoustic, _FakeVocoder())
    backend = AmekoChineseDiffSingerBackend(ameko_voicebank)
    assert backend._language_id is None
    backend.synthesize([_note(60, 0.0, 0.2, "你")])
    assert "languages" not in acoustic.feeds[0]


def test_missing_dictionary(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """A voicebank without the Chinese dictionary cannot be loaded.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    (ameko_voicebank / "dsdur" / "dsdict-zh.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        AmekoChineseDiffSingerBackend(ameko_voicebank)


def test_missing_speaker_embeddings(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """A voicebank without any speaker embedding cannot be loaded.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    for path in (ameko_voicebank / "embs").glob("*.emb"):
        path.unlink()
    with pytest.raises(ValueError, match="no speaker embeddings"):
        AmekoChineseDiffSingerBackend(ameko_voicebank)


def test_pinyin_syllables() -> None:
    """Chinese text converts to toneless syllables and blank text to nothing."""
    assert _pinyin_syllables("你好") == ["ni", "hao"]
    assert _pinyin_syllables("  ") == []
    assert _pinyin_syllables("!") == ["!"]


def test_phones_for_text(ameko_voicebank: Path) -> None:
    """Dictionary hits, misses and empty text all produce phones.

    Args:
        ameko_voicebank: The fake voicebank fixture.
    """
    dictionary = _load_dictionary(ameko_voicebank / "dsdur" / "dsdict-zh.yaml")
    assert _phones_for_text(dictionary, "你好") == ["zh/n", "zh/i", "zh/h", "zh/ao"]
    assert _phones_for_text(dictionary, "安") == ["zh/a"]
    assert _phones_for_text(dictionary, "") == ["zh/a"]


def test_language_path_derived_from_phonemes(ameko_voicebank: Path) -> None:
    """Without a languages key the table is derived from the phoneme table.

    Args:
        ameko_voicebank: The fake voicebank fixture.
    """
    phonemes = ameko_voicebank / "lang" / "millefeuille_b8.phonemes.json"
    assert _language_path(ameko_voicebank, {}, phonemes) == (
        ameko_voicebank / "lang" / "millefeuille_b8.languages.json"
    )
    assert _language_path(ameko_voicebank, {"languages": "lang/x.json"}, phonemes) == (
        ameko_voicebank / "lang" / "x.json"
    )


def test_token_id_resolution(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Exact, replacement, prefixed and unknown phones resolve to token ids.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, _acoustic, _vocoder = backend_pair
    assert backend._token_id("zh/a") == _MODERN_PHONEMES["zh/a"]
    assert backend._token_id("cl") == _MODERN_PHONEMES["zh/cl"]
    assert backend._token_id("n") == _MODERN_PHONEMES["zh/n"]
    assert backend._token_id("zzz") == _MODERN_PHONEMES["zh/a"]


def test_empty_notes(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """No notes produce no samples and no model calls.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    samples = backend.synthesize([])
    assert samples.size == 0
    assert samples.dtype == np.float64
    assert acoustic.feeds == []


def test_zero_frames(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Notes that round to zero frames produce no samples.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    assert backend.synthesize([_note(60, 0.0, 0.001, "你")]).size == 0
    assert acoustic.feeds == []


def test_zero_pitch_is_unvoiced(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """A non-positive MIDI pitch produces an all-zero f0 curve.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    backend.synthesize([_note(0, 0.0, 0.2, "你")])
    assert not np.any(acoustic.feeds[0]["f0"])


def test_tokens_durations_and_language(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Chinese text becomes zh phones, frame durations and a language array.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    notes = [
        _note(60, 0.0, 0.5, "你"),
        _note(62, 0.5, 0.5, "好"),
        _note(64, 1.0, 0.5, "世"),
        _note(65, 1.5, 0.6, "界"),
    ]
    samples = backend.synthesize(notes)
    feed = acoustic.feeds[0]
    expected_tokens = [
        _MODERN_PHONEMES["zh/n"],
        _MODERN_PHONEMES["zh/i"],
        _MODERN_PHONEMES["zh/h"],
        _MODERN_PHONEMES["zh/ao"],
        _MODERN_PHONEMES["zh/sh"],
        _MODERN_PHONEMES["zh/ir"],
        _MODERN_PHONEMES["zh/j"],
        _MODERN_PHONEMES["zh/iE"],
    ]
    assert list(feed["tokens"][0]) == expected_tokens
    assert int(np.sum(feed["durations"])) == _seconds_to_frames(2.1) == 181
    languages = feed["languages"]
    assert languages.shape == (1, 8)
    assert np.all(languages == 13)
    f0 = feed["f0"][0]
    assert np.allclose(f0[:43], 440.0 * 2.0 ** ((60 - 69) / 12.0))
    assert np.allclose(f0[43:86], 440.0 * 2.0 ** ((62 - 69) / 12.0))
    assert samples.size == math.ceil(2.1 * SAMPLE_RATE)
    assert samples.dtype == np.float64
    assert samples.ndim == 1


def test_multi_character_text(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """All syllables of a multi-character note become phones of that note.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    backend.synthesize([_note(60, 0.0, 0.5, "你好")])
    feed = acoustic.feeds[0]
    assert list(feed["tokens"][0]) == [
        _MODERN_PHONEMES["zh/n"],
        _MODERN_PHONEMES["zh/i"],
        _MODERN_PHONEMES["zh/h"],
        _MODERN_PHONEMES["zh/ao"],
    ]
    assert list(feed["durations"][0]) == [11, 11, 11, 10]


def test_missing_pinyin_falls_back(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """A syllable missing from the dictionary falls back to ``zh/a``.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    backend.synthesize([_note(60, 0.0, 0.5, "安")])
    assert list(acoustic.feeds[0]["tokens"][0]) == [_MODERN_PHONEMES["zh/a"]]


def test_empty_text_falls_back_without_pypinyin(
    monkeypatch: pytest.MonkeyPatch,
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Empty text falls back to ``zh/a`` without importing pypinyin.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        backend_pair: The Ameko backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    monkeypatch.setitem(sys.modules, "pypinyin", None)
    backend.synthesize([_note(60, 0.0, 0.5, "")])
    assert list(acoustic.feeds[0]["tokens"][0]) == [_MODERN_PHONEMES["zh/a"]]


def test_missing_pypinyin(
    monkeypatch: pytest.MonkeyPatch,
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Chinese text without pypinyin raises a clear runtime error.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        backend_pair: The Ameko backend fixture.
    """
    backend, _acoustic, _vocoder = backend_pair
    monkeypatch.setitem(sys.modules, "pypinyin", None)
    with pytest.raises(RuntimeError, match="pypinyin"):
        _pinyin_syllables("你")
    with pytest.raises(RuntimeError, match="pypinyin"):
        backend.synthesize([_note(60, 0.0, 0.5, "你")])


def test_speaker_and_diffusion_settings(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """Speaker, steps and depth reach the acoustic model.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    acoustic = _FakeAcoustic()
    _install_sessions(monkeypatch, acoustic, _FakeVocoder())
    backend = AmekoChineseDiffSingerBackend(
        ameko_voicebank,
        steps=4,
        depth=0.3,
        speaker="millefeuille_b8.kero_fly",
    )
    assert np.allclose(backend._speaker, 0.375)
    backend.synthesize([_note(60, 0.0, 0.2, "你")])
    feed = acoustic.feeds[0]
    assert int(feed["steps"]) == 4
    assert float(feed["depth"]) == pytest.approx(0.3)
    assert np.allclose(feed["spk_embed"][0], 0.375)


def test_custom_speaker_paths_and_unknown_speaker(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """Absolute speaker paths work and unknown speakers are rejected.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake voicebank fixture.
    """
    _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    absolute = ameko_voicebank / "embs" / "millefeuille_b8.kero_neutral.emb"
    backend = AmekoChineseDiffSingerBackend(ameko_voicebank, speaker=str(absolute))
    assert np.allclose(backend._speaker, 0.125)
    with pytest.raises(ValueError, match="not found"):
        AmekoChineseDiffSingerBackend(ameko_voicebank, speaker="nobody")


def test_acoustic_and_vocoder_feeds(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """The acoustic feed and the vocoder feed have the expected shapes.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, acoustic, vocoder = backend_pair
    backend.synthesize([_note(69, 0.0, 0.5, "你")])
    feed = acoustic.feeds[0]
    frames = int(np.sum(feed["durations"]))
    assert feed["tokens"].shape == (1, 2)
    assert feed["durations"].shape == (1, 2)
    assert feed["f0"].shape == (1, frames)
    assert feed["gender"].shape == (1, frames)
    assert not np.any(feed["gender"])
    assert np.allclose(feed["velocity"], 1.0)
    assert feed["spk_embed"].shape == (1, frames, 256)
    assert feed["depth"].shape == ()
    assert feed["steps"].shape == ()
    vocoder_feed = vocoder.feeds[0]
    assert vocoder_feed["mel"].shape == (1, frames, 128)
    assert np.array_equal(vocoder_feed["f0"], feed["f0"])


def test_non_finite_values_are_sanitised(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """NaN/inf values from the models never reach the caller.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, acoustic, vocoder = backend_pair
    acoustic.non_finite = True
    vocoder.non_finite = True
    samples = backend.synthesize([_note(60, 0.0, 0.2, "你")])
    assert samples.size > 0
    assert np.all(np.isfinite(samples))


def test_output_length_and_leading_silence(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """The output covers the score and places the line at the first onset.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, _acoustic, vocoder = backend_pair
    vocoder.value = 1.0
    samples = backend.synthesize([_note(60, 0.2, 0.1, "你")])
    lead = round(0.2 * SAMPLE_RATE)
    assert samples.size == math.ceil((0.2 + 0.1) * SAMPLE_RATE)
    assert not np.any(samples[:lead])
    assert samples[lead] == pytest.approx(1.0)


def test_deterministic(
    backend_pair: tuple[AmekoChineseDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Repeated synthesis with the same sessions is bit-identical.

    Args:
        backend_pair: The Ameko backend fixture.
    """
    backend, _acoustic, _vocoder = backend_pair
    notes = [_note(60, 0.0, 0.2, "你"), _note(62, 0.2, 0.2, "好")]
    assert np.array_equal(backend.synthesize(notes), backend.synthesize(notes))
