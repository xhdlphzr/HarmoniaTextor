# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for the English DiffSinger ONNX backend."""

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

from harmoniatextor.render.vocal import diffsinger_en
from harmoniatextor.render.vocal.diffsinger_en import (
    EnglishDiffSingerBackend,
    _distribute_phones,
    _fallback_phones,
    _find_vocoder,
    _group_words,
    _load_dictionary,
    _load_languages,
    _load_phonemes,
    _load_speaker_embedding,
    _read_json,
    _read_yaml,
    _resolve_speaker,
    _seconds_to_frames,
    _split_counts,
    _split_frames,
    _word_text,
)
from harmoniatextor.render.vocal.singer import SAMPLE_RATE, SungNote

_PHONEME_LINES = (
    "<PAD>",
    "AP",
    "SP",
    "a",
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ax",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "v",
    "w",
    "y",
    "z",
    "zh",
)
_PHONEME_IDS = {phone: index for index, phone in enumerate(_PHONEME_LINES)}
_DICTIONARY = """
entries:
- {grapheme: "the", phonemes: [dh, ax]}
- {grapheme: "lord", phonemes: [l, ao, r, d]}
- {grapheme: "is", phonemes: [ih, z]}
- {grapheme: "my", phonemes: [m, ay]}
- {grapheme: "shelter", phonemes: [sh, eh, l, t, er]}
- {grapheme: "alien", phonemes: [zzz]}
- {grapheme: "a", phonemes: [ax]}
- {grapheme: "silent", phonemes: []}
"""
_ACOUSTIC_MODEL = "acoustic.onnx"
_VOCODER_MODEL = "tgm_hifigan.onnx"
_LEGACY_INPUTS = (
    "tokens",
    "durations",
    "f0",
    "gender",
    "velocity",
    "spk_embed",
    "depth",
    "steps",
)
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
_EN_PHONES = (
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ax",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "dx",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "v",
    "w",
    "y",
    "z",
    "zh",
    "cl",
)
_MODERN_PHONEMES: dict[str, int] = {
    "AP": 1,
    "SP": 2,
    **{f"en/{phone}": index for index, phone in enumerate(_EN_PHONES, start=3)},
}
_MODERN_DICTIONARY = (
    "entries:\n"
    '- {grapheme: "-", phonemes: [SP]}\n'
    "replacements:\n"
    + "".join(f"  - {{from: {phone}, to: en/{phone}}}\n" for phone in _EN_PHONES)
)
_MODERN_CONFIG = """acoustic: onnx/millefeuille_b8.onnx
phonemes: lang/millefeuille_b8.phonemes.json
languages: lang/millefeuille_b8.languages.json
use_lang_id: true
vocoder: dsvocoder
speakers:
- embs/millefeuille_b8.kero_fly
- embs/millefeuille_b8.kero_neutral
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

    def __init__(self, inputs: Sequence[str] = _LEGACY_INPUTS) -> None:
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
        wave = np.full(
            (1, frames * diffsinger_en.HOP_SIZE), self.value, dtype=np.float32
        )
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

    monkeypatch.setattr(diffsinger_en, "_create_session", factory)
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
def voicebank(tmp_path: Path) -> Path:
    """Create a minimal fake Tiger-style voicebank directory.

    Args:
        tmp_path: The pytest temporary path fixture.

    Returns:
        The voicebank root.
    """
    base = tmp_path / "voicebank"
    acoustic = base / "dsacoustic"
    acoustic.mkdir(parents=True)
    (acoustic / "phonemes.txt").write_text("\n".join(_PHONEME_LINES), encoding="utf-8")
    (acoustic / _ACOUSTIC_MODEL).write_bytes(b"acoustic")
    default = np.linspace(-0.5, 0.5, num=256, dtype=np.float32)
    (acoustic / "tiger_fresh.emb").write_bytes(default.astype("<f4").tobytes())
    royal = np.full(256, 0.25, dtype=np.float32)
    (acoustic / "tiger_royal.emb").write_bytes(royal.astype("<f4").tobytes())
    vocoder = base / "dsvocoder"
    vocoder.mkdir(parents=True)
    (vocoder / _VOCODER_MODEL).write_bytes(b"vocoder")
    durations = base / "dsdur"
    durations.mkdir(parents=True)
    (durations / "dsdict-en.yaml").write_text(_DICTIONARY, encoding="utf-8")
    return base


@pytest.fixture
def ameko_voicebank(tmp_path: Path) -> Path:
    """Create a minimal fake Millefeuille/Ameko-style voicebank.

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
        json.dumps({"en": 2, "fr": 1}), encoding="utf-8"
    )
    durations = base / "dsdur"
    durations.mkdir()
    (durations / "dsdict-en.yaml").write_text(_MODERN_DICTIONARY, encoding="utf-8")
    vocoder = base / "dsvocoder"
    vocoder.mkdir()
    (vocoder / "generic_hifigan.onnx").write_bytes(b"generic")
    (vocoder / "millefeuille_hifigan.onnx").write_bytes(b"vocoder")
    embs = base / "embs"
    embs.mkdir()
    (embs / "millefeuille_b8.kero_fly.emb").write_bytes(_embedding(0.375))
    (embs / "millefeuille_b8.kero_neutral.emb").write_bytes(_embedding(0.125))
    (base / "dsconfig.yaml").write_text(_MODERN_CONFIG, encoding="utf-8")
    return base


@pytest.fixture
def backend_pair(
    monkeypatch: pytest.MonkeyPatch,
    voicebank: Path,
) -> tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder]:
    """Build a Tiger-style backend wired to fake ONNX sessions.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        voicebank: The fake voicebank fixture.

    Returns:
        The backend, its fake acoustic session and its fake vocoder.
    """
    acoustic = _FakeAcoustic()
    vocoder = _FakeVocoder()
    _install_sessions(monkeypatch, acoustic, vocoder)
    return EnglishDiffSingerBackend(voicebank), acoustic, vocoder


@pytest.fixture
def ameko_backend(
    monkeypatch: pytest.MonkeyPatch,
    ameko_voicebank: Path,
) -> tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder]:
    """Build an Ameko-style backend wired to fake ONNX sessions.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake Millefeuille voicebank fixture.

    Returns:
        The backend, its fake acoustic session and its fake vocoder.
    """
    acoustic = _FakeAcoustic(_MODERN_INPUTS)
    vocoder = _FakeVocoder()
    _install_sessions(monkeypatch, acoustic, vocoder)
    return EnglishDiffSingerBackend(ameko_voicebank), acoustic, vocoder


def _note(
    midi: int,
    onset: float,
    duration: float,
    text: str = "",
    syllabic: str = "single",
) -> SungNote:
    """Build an English sung note.

    Args:
        midi: MIDI pitch number.
        onset: Start time in seconds.
        duration: Duration in seconds.
        text: Syllable text.
        syllabic: Syllabic role.

    Returns:
        The sung note.
    """
    return SungNote(
        midi=midi,
        onset=onset,
        duration=duration,
        ipa="",
        text=text,
        language="en",
        syllabic=syllabic,
    )


def test_create_session_without_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, voicebank: Path
) -> None:
    """Building a backend without onnxruntime raises a clear error.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        voicebank: The fake voicebank fixture.
    """
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(RuntimeError, match="onnxruntime"):
        EnglishDiffSingerBackend(voicebank)


def test_create_session_with_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session factory passes the CPU provider to onnxruntime.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    calls: list[tuple[str, list[str]]] = []

    def session(model: str, providers: list[str]) -> str:
        calls.append((model, providers))
        return "session"

    runtime = type("Runtime", (), {"InferenceSession": staticmethod(session)})
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    result = diffsinger_en._create_session(Path("acoustic.onnx"))
    assert result == "session"
    assert calls == [("acoustic.onnx", ["CPUExecutionProvider"])]


def test_backend_loads_voicebank(
    monkeypatch: pytest.MonkeyPatch, voicebank: Path
) -> None:
    """The legacy layout resolves model paths and the phoneme table.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        voicebank: The fake voicebank fixture.
    """
    created = _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    backend = EnglishDiffSingerBackend(voicebank)
    assert created == [
        voicebank / "dsacoustic" / _ACOUSTIC_MODEL,
        voicebank / "dsvocoder" / _VOCODER_MODEL,
    ]
    assert backend._phoneme_ids["dh"] == _PHONEME_IDS["dh"]
    assert backend._dictionary.words["lord"] == ["l", "ao", "r", "d"]
    assert backend._dictionary.replacements == {}
    assert backend._symbol_prefix == ""
    assert backend._language_id is None


def test_tiger_root_config_is_honoured(
    monkeypatch: pytest.MonkeyPatch, voicebank: Path
) -> None:
    """A Tiger bank with a root dsconfig.yaml uses its configured paths.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        voicebank: The fake voicebank fixture.
    """
    (voicebank / "dsconfig.yaml").write_text(
        "acoustic: dsacoustic/acoustic.onnx\n"
        "phonemes: dsacoustic/phonemes.txt\n"
        "vocoder: dsvocoder\n"
        "speakers:\n- dsacoustic/tiger_fresh\n",
        encoding="utf-8",
    )
    created = _install_sessions(monkeypatch, _FakeAcoustic(), _FakeVocoder())
    backend = EnglishDiffSingerBackend(voicebank)
    assert created == [
        voicebank / "dsacoustic" / _ACOUSTIC_MODEL,
        voicebank / "dsvocoder" / _VOCODER_MODEL,
    ]
    assert backend._speaker[0] == pytest.approx(-0.5)


def test_ameko_layout_is_detected(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """The root-config layout loads JSON phonemes, languages and speakers.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake Millefeuille voicebank fixture.
    """
    acoustic = _FakeAcoustic(_MODERN_INPUTS)
    created = _install_sessions(monkeypatch, acoustic, _FakeVocoder())
    backend = EnglishDiffSingerBackend(ameko_voicebank)
    assert created == [
        ameko_voicebank / "onnx" / "millefeuille_b8.onnx",
        ameko_voicebank / "dsvocoder" / "millefeuille_hifigan.onnx",
    ]
    assert backend._phoneme_ids["en/aa"] == _MODERN_PHONEMES["en/aa"]
    assert backend._dictionary.replacements["dh"] == "en/dh"
    assert backend._dictionary.words["-"] == ["SP"]
    assert backend._symbol_prefix == "en/"
    assert backend._language_id == 2
    assert np.allclose(backend._speaker, 0.125)


def test_ameko_tokens_and_language_input(
    ameko_backend: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Heuristic phones are prefixed and the language id is passed per token.

    Args:
        ameko_backend: The Ameko backend fixture.
    """
    backend, acoustic, _vocoder = ameko_backend
    backend.synthesize([_note(69, 0.0, 0.5, "the"), _note(67, 0.5, 0.5, "lord")])
    feed = acoustic.feeds[0]
    assert list(feed["tokens"][0]) == [
        _MODERN_PHONEMES["en/th"],
        _MODERN_PHONEMES["en/eh"],
        _MODERN_PHONEMES["en/l"],
        _MODERN_PHONEMES["en/ao"],
        _MODERN_PHONEMES["en/r"],
        _MODERN_PHONEMES["en/d"],
    ]
    languages = feed["languages"]
    assert languages.shape == (1, 6)
    assert np.all(languages == 2)
    assert int(feed["durations"].sum()) == 86


def test_ameko_speaker_selection(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """Speaker names, absolute paths and unknown speakers are handled.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake Millefeuille voicebank fixture.
    """
    _install_sessions(monkeypatch, _FakeAcoustic(_MODERN_INPUTS), _FakeVocoder())
    fly = EnglishDiffSingerBackend(ameko_voicebank, speaker="millefeuille_b8.kero_fly")
    assert np.allclose(fly._speaker, 0.375)
    absolute = ameko_voicebank / "embs" / "millefeuille_b8.kero_neutral.emb"
    neutral = EnglishDiffSingerBackend(ameko_voicebank, speaker=str(absolute))
    assert np.allclose(neutral._speaker, 0.125)
    with pytest.raises(ValueError, match="not found"):
        EnglishDiffSingerBackend(ameko_voicebank, speaker="nobody")


def test_ameko_without_english_language(
    monkeypatch: pytest.MonkeyPatch, ameko_voicebank: Path
) -> None:
    """A bank whose language table lacks English is rejected.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        ameko_voicebank: The fake Millefeuille voicebank fixture.
    """
    (ameko_voicebank / "lang" / "millefeuille_b8.languages.json").write_text(
        json.dumps({"fr": 1}), encoding="utf-8"
    )
    _install_sessions(monkeypatch, _FakeAcoustic(_MODERN_INPUTS), _FakeVocoder())
    with pytest.raises(ValueError, match="English"):
        EnglishDiffSingerBackend(ameko_voicebank)


def test_ameko_token_mapping(
    ameko_backend: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Exact, replacement and unknown phones map to the right token ids.

    Args:
        ameko_backend: The Ameko backend fixture.
    """
    backend, _acoustic, _vocoder = ameko_backend
    assert backend._token_id("en/aa") == _MODERN_PHONEMES["en/aa"]
    assert backend._token_id("dh") == _MODERN_PHONEMES["en/dh"]
    assert backend._token_id("zzz") == _MODERN_PHONEMES["AP"]


def test_legacy_token_mapping(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """The legacy table maps bare phones and falls back to AP.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, _acoustic, _vocoder = backend_pair
    assert backend._token_id("dh") == _PHONEME_IDS["dh"]
    assert backend._token_id("zzz") == _PHONEME_IDS["AP"]


def test_speaker_embedding_is_used(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
    voicebank: Path,
) -> None:
    """The chosen speaker embedding is repeated over every frame.

    Args:
        backend_pair: The Tiger backend fixture.
        voicebank: The fake voicebank fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    backend.synthesize([_note(69, 0.0, 0.2, "the")])
    feed = acoustic.feeds[0]
    expected = np.frombuffer(
        (voicebank / "dsacoustic" / "tiger_fresh.emb").read_bytes(), dtype="<f4"
    )
    frames = int(np.sum(feed["durations"]))
    assert feed["spk_embed"].shape == (1, frames, 256)
    assert np.allclose(feed["spk_embed"][0, :, :], expected, atol=1e-6)


def test_custom_speaker_steps_and_depth(
    monkeypatch: pytest.MonkeyPatch, voicebank: Path
) -> None:
    """Speaker selection and diffusion settings reach the models.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        voicebank: The fake voicebank fixture.
    """
    acoustic = _FakeAcoustic()
    _install_sessions(monkeypatch, acoustic, _FakeVocoder())
    backend = EnglishDiffSingerBackend(
        voicebank, steps=4, depth=0.3, speaker="tiger_royal"
    )
    backend.synthesize([_note(69, 0.0, 0.2, "a")])
    feed = acoustic.feeds[0]
    assert int(feed["steps"]) == 4
    assert float(feed["depth"]) == pytest.approx(0.3)
    assert np.allclose(feed["spk_embed"][0], 0.25)


def test_word_grouping() -> None:
    """Syllabic roles and melisma notes form word-sized runs."""
    notes = [
        _note(60, 0.0, 0.1, "The"),
        _note(60, 0.1, 0.1, "shel", "begin"),
        _note(60, 0.2, 0.1, "ter", "end"),
        _note(60, 0.3, 0.1, ""),
        _note(60, 0.4, 0.1, "Lord"),
    ]
    assert _group_words(notes) == [[0], [1, 2, 3], [4]]
    assert _group_words([_note(60, 0.0, 0.1, "", "end")]) == [[0]]
    assert _group_words([]) == []


def test_word_text_strips_punctuation() -> None:
    """Joined word text is lower-case and only keeps letters and apostrophes."""
    notes = [_note(60, 0.0, 0.1, " The,"), _note(60, 0.1, 0.1, "Lord's!")]
    assert _word_text(notes, [0, 1]) == "thelord's"


def test_fallback_phones() -> None:
    """The heuristic maps digraphs, letters and skips unknown characters."""
    assert _fallback_phones("shepherd") == ["sh", "eh", "f", "er", "d"]
    assert _fallback_phones("the") == ["th", "eh"]
    assert _fallback_phones("a1") == ["ae"]
    assert _fallback_phones("") == ["a"]


def test_split_helpers() -> None:
    """Frame and phone splits cover even, empty and proportional cases."""
    assert _split_frames(10, 3) == [4, 3, 3]
    assert _split_frames(0, 2) == [0, 0]
    assert _split_frames(10, 0) == []
    assert _split_counts(4, [1.0, 1.0, 1.0]) == [2, 1, 1]
    assert _split_counts(3, [10.0, 1.0, 1.0]) == [1, 1, 1]
    assert _split_counts(3, [0.0, 0.0]) == [2, 1]


def test_distribute_phones() -> None:
    """Phones split over notes, with melisma notes repeating the last phone."""
    assert _distribute_phones(["a", "b", "c", "d", "e", "f"], [30, 10]) == [
        ["a", "b", "c", "d", "e"],
        ["f"],
    ]
    assert _distribute_phones(["a", "b"], [10, 20]) == [["a"], ["b"]]
    assert _distribute_phones(["a", "b"], [10, 10, 10, 10]) == [
        ["a"],
        ["b"],
        ["b"],
        ["b"],
    ]


def test_seconds_to_frames() -> None:
    """Seconds convert to rounded 512-sample frames at 44.1 kHz."""
    assert _seconds_to_frames(0.0) == 0
    assert _seconds_to_frames(1.0) == round(SAMPLE_RATE / 512)


def test_empty_notes(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """No notes produce no samples.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    samples = backend.synthesize([])
    assert samples.size == 0
    assert samples.dtype == np.float64
    assert acoustic.feeds == []


def test_zero_duration_notes(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Notes that round to zero frames produce no samples.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, _acoustic, _vocoder = backend_pair
    assert backend.synthesize([_note(69, 0.0, 0.0, "a")]).size == 0


def test_frames_and_pitch(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Durations, f0, gender, velocity and the output length follow the notes.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    notes = [_note(69, 0.0, 0.5, "the"), _note(81, 0.5, 0.5, "lord")]
    samples = backend.synthesize(notes)
    feed = acoustic.feeds[0]
    durations = feed["durations"][0]
    assert int(durations.sum()) == _seconds_to_frames(1.0) == 86
    assert list(durations) == [22, 21, 11, 11, 11, 10]
    assert list(feed["tokens"][0]) == [
        _PHONEME_IDS["dh"],
        _PHONEME_IDS["ax"],
        _PHONEME_IDS["l"],
        _PHONEME_IDS["ao"],
        _PHONEME_IDS["r"],
        _PHONEME_IDS["d"],
    ]
    f0 = feed["f0"][0]
    assert np.allclose(f0[:43], 440.0)
    assert np.allclose(f0[43:], 880.0)
    assert not np.any(feed["gender"])
    assert np.allclose(feed["velocity"], 1.0)
    assert samples.size == math.ceil(1.0 * SAMPLE_RATE)
    assert samples[0] == pytest.approx(0.5)
    assert samples[-1] == 0.0


def test_zero_pitch_note_is_unvoiced(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """A non-positive MIDI pitch produces an all-zero (unvoiced) f0 curve.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    backend.synthesize([_note(0, 0.0, 0.2, "a")])
    assert not np.any(acoustic.feeds[0]["f0"])


def test_word_fragments_join_for_dictionary_lookup(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """A begin/end syllable pair is looked up as one joined word.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    backend.synthesize(
        [_note(69, 0.0, 0.5, "shel", "begin"), _note(69, 0.5, 0.5, "ter", "end")]
    )
    tokens = list(acoustic.feeds[0]["tokens"][0])
    assert tokens == [
        _PHONEME_IDS["sh"],
        _PHONEME_IDS["eh"],
        _PHONEME_IDS["l"],
        _PHONEME_IDS["t"],
        _PHONEME_IDS["er"],
    ]
    assert int(acoustic.feeds[0]["durations"].sum()) == 86


def test_dictionary_fallback_and_melisma(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Unknown words use the heuristic and melisma notes repeat the last phone.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    backend.synthesize(
        [_note(69, 0.0, 0.2, "zorp"), _note(69, 0.2, 0.2, "my")]
        + [_note(69, 0.4, 0.2, ""), _note(69, 0.6, 0.2, "")]
    )
    tokens = list(acoustic.feeds[0]["tokens"][0])
    assert tokens == [
        _PHONEME_IDS["z"],
        _PHONEME_IDS["ao"],
        _PHONEME_IDS["r"],
        _PHONEME_IDS["p"],
        _PHONEME_IDS["m"],
        _PHONEME_IDS["ay"],
        _PHONEME_IDS["ay"],
    ]


def test_unknown_phone_uses_fallback_token(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """A dictionary phone outside the phoneme table maps to the AP token.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    backend.synthesize([_note(69, 0.0, 0.2, "alien")])
    assert list(acoustic.feeds[0]["tokens"][0]) == [_PHONEME_IDS["AP"]]


def test_empty_text_uses_vowel(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """A word with no text falls back to a neutral vowel, never silence.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, acoustic, _vocoder = backend_pair
    backend.synthesize([_note(69, 0.0, 0.2, "")])
    assert list(acoustic.feeds[0]["tokens"][0]) == [_PHONEME_IDS["a"]]


def test_non_finite_values_are_sanitised(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """NaN/inf values from the models never reach the caller.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, acoustic, vocoder = backend_pair
    acoustic.non_finite = True
    vocoder.non_finite = True
    samples = backend.synthesize([_note(69, 0.0, 0.2, "the")])
    assert samples.size > 0
    assert np.all(np.isfinite(samples))


def test_leading_silence_covers_onset(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """The output starts with silence up to the first note onset.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, _acoustic, vocoder = backend_pair
    vocoder.value = 1.0
    samples = backend.synthesize([_note(69, 0.2, 0.1, "a")])
    lead = round(0.2 * SAMPLE_RATE)
    assert samples.size == math.ceil((0.2 + 0.1) * SAMPLE_RATE)
    assert not np.any(samples[:lead])
    assert samples[lead] == pytest.approx(1.0)


def test_deterministic(
    backend_pair: tuple[EnglishDiffSingerBackend, _FakeAcoustic, _FakeVocoder],
) -> None:
    """Repeated synthesis with the same sessions is bit-identical.

    Args:
        backend_pair: The Tiger backend fixture.
    """
    backend, _acoustic, _vocoder = backend_pair
    notes = [_note(69, 0.0, 0.2, "the"), _note(67, 0.2, 0.2, "lord")]
    first = backend.synthesize(notes)
    second = backend.synthesize(notes)
    assert np.array_equal(first, second)


def test_read_yaml_requires_mapping(tmp_path: Path) -> None:
    """A YAML file without a mapping is rejected.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        _read_yaml(path)


def test_read_json_requires_object(tmp_path: Path) -> None:
    """A JSON file without an object is rejected.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    path = tmp_path / "list.json"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        _read_json(path)


def test_empty_phonemes_rejected(tmp_path: Path) -> None:
    """An empty phoneme table is rejected.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    path = tmp_path / "phonemes.txt"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no phonemes"):
        _load_phonemes(path)


def test_empty_languages_rejected(tmp_path: Path) -> None:
    """A language table without languages is rejected.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    path = tmp_path / "languages.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no languages"):
        _load_languages(path)


def test_empty_dictionary_rejected(tmp_path: Path) -> None:
    """A dictionary without usable entries is rejected.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    path = tmp_path / "dictionary.yaml"
    path.write_text("entries: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no dictionary"):
        _load_dictionary(path)


def test_dictionary_with_only_replacements(tmp_path: Path) -> None:
    """A replacement table alone is a valid dictionary.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    path = tmp_path / "dictionary.yaml"
    path.write_text(
        "replacements:\n- {from: dh, to: en/dh}\n- {from: '', to: ''}\n",
        encoding="utf-8",
    )
    dictionary = _load_dictionary(path)
    assert dictionary.words == {}
    assert dictionary.replacements == {"dh": "en/dh"}


def test_malformed_speaker_embedding(tmp_path: Path) -> None:
    """A speaker embedding with the wrong number of floats is rejected.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    path = tmp_path / "speaker.emb"
    path.write_bytes(np.zeros(2, dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="256"):
        _load_speaker_embedding(path)


def test_resolve_speaker_fallback_and_empty(tmp_path: Path) -> None:
    """The default falls back to the first embedding, or fails when none exist.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    fallback = tmp_path / "embs"
    fallback.mkdir()
    (fallback / "aaa.emb").write_bytes(_embedding(0.0))
    assert _resolve_speaker(tmp_path, {}, None) == fallback / "aaa.emb"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no speaker embeddings"):
        _resolve_speaker(empty, {}, None)


def test_find_vocoder_selection(tmp_path: Path) -> None:
    """A single model wins, matching names win, otherwise the first is used.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    single = tmp_path / "single"
    single.mkdir()
    (single / "vocoder.onnx").write_bytes(b"v")
    assert _find_vocoder(tmp_path / "acoustic.onnx", single) == single / "vocoder.onnx"
    multi = tmp_path / "multi"
    multi.mkdir()
    (multi / "generic_hifigan.onnx").write_bytes(b"g")
    (multi / "millefeuille_hifigan.onnx").write_bytes(b"m")
    assert (
        _find_vocoder(tmp_path / "millefeuille_b8.onnx", multi)
        == multi / "millefeuille_hifigan.onnx"
    )
    assert (
        _find_vocoder(tmp_path / "acoustic.onnx", multi)
        == multi / "generic_hifigan.onnx"
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no ONNX"):
        _find_vocoder(tmp_path / "acoustic.onnx", empty)
