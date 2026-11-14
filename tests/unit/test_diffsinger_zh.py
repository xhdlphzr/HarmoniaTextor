# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for the Chinese DiffSinger ONNX backend."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

from harmoniatextor.render.vocal.diffsinger_pinyin import (
    PHONE_IDS,
    PHONES,
    PINYIN_TO_PHONES,
)
from harmoniatextor.render.vocal.diffsinger_zh import (
    ChineseDiffSingerBackend,
    _build_sequence,
    _load_config,
    _phones,
    _pinyin,
    _resample,
    _sanitise,
)
from harmoniatextor.render.vocal.singer import SAMPLE_RATE, SungNote

#: Number of mel bins in a fake model configuration.
_SPEC_BINS = 80
#: Fake model files created by the session factory.
_MODEL_FILES = ("fs2.onnx", "q_sample.onnx", "p_sample.onnx", "pe.onnx", "vocoder.onnx")


class _FakeSession:
    """A deterministic stand-in for an ONNX Runtime session."""

    def __init__(
        self,
        name: str,
        *,
        frames: int = 8,
        samples: int = 24000,
        passthrough: bool = False,
        wav: NDArray[Any] | None = None,
    ) -> None:
        """Store the fake model configuration.

        Args:
            name: Name of the ONNX file this session stands in for.
            frames: Number of mel frames the fake model produces.
            samples: Number of waveform samples the fake vocoder produces.
            passthrough: Whether the diffusion sampler returns its input as is.
            wav: Optional explicit waveform returned by the fake vocoder.
        """
        self.name = name
        self.frames = frames
        self.samples = samples
        self.passthrough = passthrough
        self.wav = wav
        self.feeds: list[dict[str, Any]] = []

    @property
    def mel(self) -> NDArray[np.float32]:
        """Return the deterministic mel spectrogram the fake model emits."""
        values = np.linspace(-1.0, 1.0, self.frames * _SPEC_BINS, dtype=np.float32)
        return values.reshape(1, self.frames, _SPEC_BINS)

    def run(self, output_names: list[str], feeds: dict[str, Any]) -> list[Any]:
        """Return deterministic outputs for the requested names.

        Args:
            output_names: ONNX output names requested by the caller.
            feeds: ONNX input arrays.

        Returns:
            One fake array per requested output.
        """
        self.feeds.append(dict(feeds))
        return [self._output(name, feeds) for name in output_names]

    def _output(self, name: str, feeds: dict[str, Any]) -> NDArray[Any]:
        """Return the fake array for one output name.

        Args:
            name: ONNX output name.
            feeds: ONNX input arrays.

        Returns:
            The fake output array.

        Raises:
            AssertionError: If the output name is not part of the fake model.
        """
        if name == "decoder_inp":
            values = np.linspace(0.0, 1.0, self.frames * 4, dtype=np.float32)
            return values.reshape(1, self.frames, 4)
        if name == "mel_out":
            return self.mel
        if name == "x_next":
            base = feeds["x_start"] if "x_start" in feeds else feeds["x"]
            if self.passthrough:
                return np.asarray(base, dtype=np.float32)
            noise = np.asarray(feeds["noise"], dtype=np.float32)
            return np.asarray(base, dtype=np.float32) + np.float32(0.01) * noise
        if name == "f0_denorm_pred":
            return np.full((1, self.frames), 440.0, dtype=np.float32)
        if name == "wav_out":
            if self.wav is not None:
                return self.wav
            return np.linspace(-0.5, 0.5, self.samples, dtype=np.float32).reshape(1, -1)
        raise AssertionError(f"unexpected fake output {name}")


def _model_dir(
    tmp_path: Path,
    *,
    k_steps: int = 3,
    spec_min: float = -1.0,
    spec_max: float = 1.0,
) -> Path:
    """Create a temporary fake DiffSinger model directory.

    Args:
        tmp_path: The pytest temporary path fixture.
        k_steps: Number of diffusion steps in the fake configuration.
        spec_min: Minimum mel value in the fake configuration.
        spec_max: Maximum mel value in the fake configuration.

    Returns:
        Path of the fake model directory.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {
        "K_step": k_steps,
        "audio_sample_rate": 24000,
        "spec_min": [spec_min] * _SPEC_BINS,
        "spec_max": [spec_max] * _SPEC_BINS,
    }
    (model_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return model_dir


def _sessions(
    *,
    frames: int = 8,
    samples: int = 24000,
    passthrough: bool = False,
    wav: NDArray[Any] | None = None,
) -> dict[str, _FakeSession]:
    """Create a complete set of fake sessions for the model chain.

    Args:
        frames: Number of mel frames the fake model produces.
        samples: Number of waveform samples the fake vocoder produces.
        passthrough: Whether the diffusion sampler returns its input as is.
        wav: Optional explicit waveform returned by the fake vocoder.

    Returns:
        Mapping of ONNX file names to fake sessions.
    """
    return {
        name: _FakeSession(
            name,
            frames=frames,
            samples=samples,
            passthrough=passthrough,
            wav=wav,
        )
        for name in _MODEL_FILES
    }


def _patch_sessions(
    monkeypatch: pytest.MonkeyPatch, sessions: Mapping[str, _FakeSession]
) -> None:
    """Patch the backend's session factory to return the fake sessions.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        sessions: Mapping of ONNX file names to fake sessions.
    """

    def factory(path: Path) -> _FakeSession:
        return sessions[path.name]

    monkeypatch.setattr(ChineseDiffSingerBackend, "_session", staticmethod(factory))


def _install_pypinyin(
    monkeypatch: pytest.MonkeyPatch, mapping: Mapping[str, list[str]]
) -> None:
    """Install a fake ``pypinyin`` module into :data:`sys.modules`.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        mapping: Text-to-syllables mapping used by the fake module.
    """

    def lazy_pinyin(text: str, *, style: object) -> list[str]:
        del style
        return list(mapping.get(text, []))

    module: Any = types.ModuleType("pypinyin")
    module.Style = types.SimpleNamespace(NORMAL="normal")
    module.lazy_pinyin = lazy_pinyin
    monkeypatch.setitem(sys.modules, "pypinyin", module)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """Return a temporary fake DiffSinger model directory.

    Args:
        tmp_path: The pytest temporary path fixture.

    Returns:
        Path of the fake model directory.
    """
    return _model_dir(tmp_path)


def test_data_tables_match_model_tokens() -> None:
    """The embedded tables only reference phones the model knows."""
    assert PHONES[:2] == ("AP", "SP")
    assert len(PHONES) == len(PHONE_IDS) == 60
    unknown = {
        phone
        for mapping in PINYIN_TO_PHONES.values()
        for phone in mapping.split()
        if phone not in PHONE_IDS
    }
    assert unknown == {"ng"}


def test_resample_degenerate() -> None:
    """Empty buffers and degenerate rates are returned unchanged."""
    empty = np.zeros(0, dtype=np.float64)
    assert _resample(empty, 24000).size == 0
    samples = np.array([0.0, 1.0])
    assert _resample(samples, 0) is samples
    assert _resample(samples, SAMPLE_RATE) is samples


def test_resample_length() -> None:
    """Linear resampling scales the sample count to the target rate."""
    samples = np.linspace(0.0, 1.0, 2400)
    result = _resample(samples, 24000)
    assert result.size == 4410
    assert result.dtype == np.float64


def test_resample_linear_values() -> None:
    """Linear interpolation keeps the endpoint values."""
    result = _resample(np.array([0.0, 1.0]), 2, 4)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0, 1.0])


def test_sanitise() -> None:
    """Non-finite samples become silence and the rest is clipped."""
    samples = np.array([np.nan, np.inf, -np.inf, 2.0, -2.0, 0.5])
    result = _sanitise(samples)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 1.0, -1.0, 0.5])
    assert result.dtype == np.float64


def test_load_config(model_dir: Path) -> None:
    """The configuration is parsed into a mapping."""
    config = _load_config(model_dir / "config.yaml")
    assert config["K_step"] == 3
    assert config["audio_sample_rate"] == 24000


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    """A configuration that is not a mapping is rejected."""
    path = tmp_path / "config.yaml"
    path.write_text("not a mapping\n", encoding="utf-8")
    with pytest.raises(TypeError, match="not a mapping"):
        _load_config(path)


def test_session_factory_uses_cpu_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default factory creates CPU-only ONNX Runtime sessions."""
    created: list[tuple[str, list[str]]] = []

    class _InferenceSession:
        """Record the session arguments for the test."""

        def __init__(self, path: str, *, providers: list[str]) -> None:
            """Record the path and providers of a fake session."""
            created.append((path, providers))

    fake: Any = types.SimpleNamespace(InferenceSession=_InferenceSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    session = ChineseDiffSingerBackend._session(Path("unit.onnx"))
    assert isinstance(session, _InferenceSession)
    assert created == [("unit.onnx", ["CPUExecutionProvider"])]


def test_missing_onnxruntime(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing onnxruntime package raises a clear runtime error."""
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, None))
    with pytest.raises(RuntimeError, match="onnxruntime"):
        ChineseDiffSingerBackend._session(Path("unit.onnx"))
    with pytest.raises(RuntimeError, match="onnxruntime"):
        ChineseDiffSingerBackend(model_dir)


def test_module_import_needs_no_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the backend does not import its optional dependencies."""
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, None))
    monkeypatch.setitem(sys.modules, "pypinyin", cast(Any, None))
    module = importlib.import_module("harmoniatextor.render.vocal.diffsinger_zh")
    assert module.ChineseDiffSingerBackend is ChineseDiffSingerBackend


def test_missing_config(tmp_path: Path) -> None:
    """A model directory without configuration fails to load."""
    with pytest.raises(FileNotFoundError):
        ChineseDiffSingerBackend(tmp_path / "absent")


def test_constructor_reads_config(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The constructor reads the sample rate, diffusion steps and mel ranges."""
    sessions = _sessions()
    _patch_sessions(monkeypatch, sessions)
    backend = ChineseDiffSingerBackend(model_dir)
    assert backend._sample_rate == 24000
    assert backend._k_steps == 3
    assert backend._spec_min.shape == (_SPEC_BINS,)
    assert backend._spec_max.shape == (_SPEC_BINS,)
    assert backend._fs2 is sessions["fs2.onnx"]
    assert backend._vocoder is sessions["vocoder.onnx"]


def test_constructor_requests_every_model_file(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session factory is asked for every ONNX file in the chain."""
    names: list[str] = []

    def factory(path: Path) -> _FakeSession:
        names.append(path.name)
        return _FakeSession(path.name)

    monkeypatch.setattr(ChineseDiffSingerBackend, "_session", staticmethod(factory))
    ChineseDiffSingerBackend(model_dir)
    assert names == list(_MODEL_FILES)


def test_build_sequence_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinyin is converted to phones carrying the note's pitch and duration."""
    _install_pypinyin(monkeypatch, {"你": ["ni"], "好": ["hao"]})
    notes = [
        SungNote(midi=60, onset=0.0, duration=0.5, ipa="ni", text="你"),
        SungNote(midi=62, onset=0.5, duration=0.5, ipa="xau", text="好"),
    ]
    tokens, pitches, durations = _build_sequence(notes)
    assert tokens.tolist() == [
        [PHONE_IDS["n"], PHONE_IDS["i"], PHONE_IDS["h"], PHONE_IDS["ao"]]
    ]
    assert pitches.tolist() == [[60, 60, 62, 62]]
    np.testing.assert_allclose(durations, [[0.5, 0.5, 0.5, 0.5]], rtol=1e-6)
    assert durations.dtype == np.float32


def test_build_sequence_multi_char_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the first pinyin syllable of multi-character text is used."""
    _install_pypinyin(monkeypatch, {"你好": ["ni", "hao"]})
    note = SungNote(midi=60, onset=0.0, duration=0.5, ipa="", text="你好")
    tokens, _, _ = _build_sequence([note])
    assert tokens.tolist() == [[PHONE_IDS["n"], PHONE_IDS["i"]]]


def test_build_sequence_unknown_pinyin(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown pinyin syllable falls back to the neutral phone."""
    _install_pypinyin(monkeypatch, {"x": ["zzz"]})
    note = SungNote(midi=60, onset=0.0, duration=0.5, ipa="", text="x")
    tokens, _, _ = _build_sequence([note])
    assert tokens.tolist() == [[PHONE_IDS["a"]]]


def test_build_sequence_empty_text_uses_ipa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty text uses the note's IPA as the pinyin key without pypinyin."""
    monkeypatch.setitem(sys.modules, "pypinyin", cast(Any, None))
    note = SungNote(midi=60, onset=0.0, duration=0.5, ipa="ni", text="")
    tokens, _, _ = _build_sequence([note])
    assert tokens.tolist() == [[PHONE_IDS["n"], PHONE_IDS["i"]]]


def test_build_sequence_empty_text_and_ipa() -> None:
    """A note without text and IPA falls back to the neutral phone."""
    note = SungNote(midi=60, onset=0.0, duration=0.5, ipa="", text="")
    assert _pinyin(note) == "a"
    assert _phones(note) == ["a"]


def test_pinyin_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty pypinyin result falls back to the neutral phone."""
    _install_pypinyin(monkeypatch, {})
    note = SungNote(midi=60, onset=0.0, duration=0.5, ipa="", text="你")
    assert _pinyin(note) == "a"


def test_pinyin_uses_normal_style(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pinyin conversion strips tone marks through ``Style.NORMAL``."""
    style = object()
    seen: list[object] = []

    def lazy_pinyin(text: str, *, style: object) -> list[str]:
        del text
        seen.append(style)
        return ["NI"]

    module: Any = types.ModuleType("pypinyin")
    module.Style = types.SimpleNamespace(NORMAL=style)
    module.lazy_pinyin = lazy_pinyin
    monkeypatch.setitem(sys.modules, "pypinyin", module)
    note = SungNote(midi=60, onset=0.0, duration=0.5, ipa="", text="你")
    assert _pinyin(note) == "ni"
    assert seen == [style]


def test_missing_pypinyin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chinese text without pypinyin raises a clear runtime error."""
    monkeypatch.setitem(sys.modules, "pypinyin", cast(Any, None))
    note = SungNote(midi=60, onset=0.0, duration=0.5, ipa="", text="你")
    with pytest.raises(RuntimeError, match="pypinyin"):
        _pinyin(note)


def test_empty_notes(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No notes produce an empty buffer without touching the model."""
    sessions = _sessions()
    _patch_sessions(monkeypatch, sessions)
    backend = ChineseDiffSingerBackend(model_dir)
    samples = backend.synthesize([])
    assert samples.shape == (0,)
    assert samples.dtype == np.float64
    assert all(not session.feeds for session in sessions.values())


def test_synthesize_feeds_model(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-phone note produces the expected token, pitch and duration."""
    sessions = _sessions()
    _patch_sessions(monkeypatch, sessions)
    backend = ChineseDiffSingerBackend(model_dir)
    backend.synthesize([SungNote(midi=60, onset=0.0, duration=0.5, ipa="a")])
    feed = sessions["fs2.onnx"].feeds[0]
    assert feed["txt_tokens"].tolist() == [[PHONE_IDS["a"]]]
    assert feed["pitch_midi"].tolist() == [[60]]
    assert feed["is_slur"].tolist() == [[0]]
    np.testing.assert_allclose(feed["midi_dur"], [[0.5]], rtol=1e-6)


def test_synthesize_covers_score(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output is 44.1 kHz and at least as long as the score."""
    sessions = _sessions(samples=24000)
    _patch_sessions(monkeypatch, sessions)
    backend = ChineseDiffSingerBackend(model_dir)
    samples = backend.synthesize([SungNote(midi=60, onset=0.0, duration=1.0, ipa="a")])
    assert samples.dtype == np.float64
    assert samples.ndim == 1
    assert samples.size == SAMPLE_RATE


def test_synthesize_pads_short_output(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short model output is padded with silence to cover the score."""
    sessions = _sessions(samples=240)
    _patch_sessions(monkeypatch, sessions)
    backend = ChineseDiffSingerBackend(model_dir)
    samples = backend.synthesize([SungNote(midi=60, onset=0.0, duration=1.0, ipa="a")])
    assert samples.size == SAMPLE_RATE
    assert np.count_nonzero(samples[441:]) == 0


def test_synthesize_keeps_tail(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model output longer than the score is kept, tail included."""
    sessions = _sessions(samples=48000)
    _patch_sessions(monkeypatch, sessions)
    backend = ChineseDiffSingerBackend(model_dir)
    samples = backend.synthesize([SungNote(midi=60, onset=0.0, duration=0.5, ipa="a")])
    assert samples.size == 2 * SAMPLE_RATE


def test_synthesize_sanitises(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NaN, infinities and out-of-range model output become valid samples."""
    wav = np.array([[np.nan, np.inf, -np.inf, 2.0, -2.0, 0.25]], dtype=np.float32)
    _patch_sessions(monkeypatch, _sessions(wav=wav))
    backend = ChineseDiffSingerBackend(model_dir)
    samples = backend.synthesize([SungNote(midi=60, onset=0.0, duration=1.0, ipa="a")])
    assert np.all(np.isfinite(samples))
    assert float(samples.max()) <= 1.0
    assert float(samples.min()) >= -1.0
    assert samples[0] == 0.0


def test_synthesize_is_deterministic(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixed diffusion seed makes repeated renders identical."""
    _patch_sessions(monkeypatch, _sessions())
    backend = ChineseDiffSingerBackend(model_dir)
    notes = [SungNote(midi=60, onset=0.0, duration=0.5, ipa="a", text="")]
    np.testing.assert_array_equal(backend.synthesize(notes), backend.synthesize(notes))


def test_synthesize_missing_pypinyin(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthesising Chinese text without pypinyin raises a runtime error."""
    _patch_sessions(monkeypatch, _sessions())
    monkeypatch.setitem(sys.modules, "pypinyin", cast(Any, None))
    backend = ChineseDiffSingerBackend(model_dir)
    note = SungNote(midi=60, onset=0.0, duration=0.5, ipa="", text="你")
    with pytest.raises(RuntimeError, match="pypinyin"):
        backend.synthesize([note])


def test_diffusion_step_feeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sampler receives the configured steps and the right tensor shapes."""
    model_dir = _model_dir(tmp_path, k_steps=4)
    sessions = _sessions(frames=6)
    _patch_sessions(monkeypatch, sessions)
    backend = ChineseDiffSingerBackend(model_dir)
    backend.synthesize([SungNote(midi=60, onset=0.0, duration=0.5, ipa="a")])
    q_feeds = sessions["q_sample.onnx"].feeds
    assert [feed["t"].tolist() for feed in q_feeds] == [[3]]
    p_feeds = sessions["p_sample.onnx"].feeds
    assert [feed["t"].tolist() for feed in p_feeds] == [[3], [2], [1], [0]]
    assert p_feeds[0]["cond"].shape == (1, 4, 6)
    assert p_feeds[0]["x"].shape == (1, 1, _SPEC_BINS, 6)


def test_normalisation_roundtrip(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normalising and denormalising the mel restores the model's mel."""
    sessions = _sessions(frames=5, passthrough=True)
    _patch_sessions(monkeypatch, sessions)
    backend = ChineseDiffSingerBackend(model_dir)
    backend.synthesize([SungNote(midi=60, onset=0.0, duration=0.5, ipa="a")])
    fs2 = sessions["fs2.onnx"]
    q_feeds = sessions["q_sample.onnx"].feeds
    np.testing.assert_allclose(
        q_feeds[0]["x_start"][0, 0], fs2.mel[0].T, rtol=1e-5, atol=1e-6
    )
    pe_feeds = sessions["pe.onnx"].feeds
    np.testing.assert_allclose(pe_feeds[0]["mel_input"], fs2.mel, rtol=1e-5, atol=1e-6)
    vocoder_feeds = sessions["vocoder.onnx"].feeds
    np.testing.assert_allclose(
        vocoder_feeds[0]["mel_out"], fs2.mel, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(vocoder_feeds[0]["f0"], np.full((1, 5), 440.0))
