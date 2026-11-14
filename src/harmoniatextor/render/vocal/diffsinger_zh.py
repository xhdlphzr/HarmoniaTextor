# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Chinese DiffSinger ONNX vocal backend.

The backend renders Chinese :class:`~harmoniatextor.render.vocal.singer.SungNote`
notes with a DiffSinger MIDI acoustic model exported to ONNX.  A note's Chinese
text is converted to pinyin, the pinyin to OpenCpop phonemes, and the resulting
phone sequence is fed to the model's acoustic chain: the FastSpeech2 network
(``fs2``), the shallow-diffusion sampler (``q_sample``/``p_sample``), the pitch
extractor (``pe``) and the neural vocoder.

``onnxruntime`` and ``pypinyin`` are optional dependencies.  Importing this
module never requires them, so the model-free FOF singer keeps working in a
plain install; constructing the backend without ``onnxruntime`` or synthesising
Chinese text without ``pypinyin`` raises a clear :class:`RuntimeError` instead.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from harmoniatextor.render.vocal.diffsinger_pinyin import PHONE_IDS, PINYIN_TO_PHONES
from harmoniatextor.render.vocal.singer import SAMPLE_RATE, SungNote

__all__ = ["ChineseDiffSingerBackend"]

#: Seed of the diffusion noise, fixed so a render is deterministic.
_SEED = 0
#: Phone used when a syllable cannot be mapped to the model's vocabulary.
_FALLBACK_PHONE = "a"
#: Names of the ONNX files that make up the acoustic model.
_MODEL_FILES = ("fs2.onnx", "q_sample.onnx", "p_sample.onnx", "pe.onnx", "vocoder.onnx")

_Config = dict[str, Any]
_Array = NDArray[Any]
_FloatArray = NDArray[np.float64]
_IntArray = NDArray[np.int64]


def _import_optional(name: str) -> Any:
    """Import an optional dependency, raising a clear error when it is missing.

    Args:
        name: Import name of the package.

    Returns:
        The imported module.

    Raises:
        RuntimeError: If the package is not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        message = (
            f"the '{name}' package is required for the Chinese DiffSinger backend; "
            f"install it with 'uv pip install {name}'"
        )
        raise RuntimeError(message) from exc


def _load_config(path: Path) -> _Config:
    """Read a DiffSinger ``config.yaml``.

    Args:
        path: Path of the configuration file.

    Returns:
        The parsed configuration mapping.

    Raises:
        TypeError: If the file does not contain a mapping.
    """
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"DiffSinger config {path} is not a mapping")
    return cast(_Config, loaded)


def _pinyin(note: SungNote) -> str:
    """Return the pinyin syllable used to look up a note's phones.

    The note's Chinese ``text`` is converted with :mod:`pypinyin`.  When the text
    is empty the note's IPA is tried as the pinyin key, and ``a`` is the last
    resort.

    Args:
        note: The note to convert.

    Returns:
        A lower-case pinyin syllable without tone marks.
    """
    text = note.text.strip()
    if not text:
        return note.ipa.strip() or _FALLBACK_PHONE
    pypinyin = _import_optional("pypinyin")
    syllables: list[str] = list(pypinyin.lazy_pinyin(text, style=pypinyin.Style.NORMAL))
    if not syllables:
        return _FALLBACK_PHONE
    return str(syllables[0]).lower()


def _phones(note: SungNote) -> list[str]:
    """Return the model's phone tokens for one note.

    Args:
        note: The note to convert.

    Returns:
        The space-split OpenCpop phones, or ``["a"]`` for an unknown syllable.
    """
    mapping = PINYIN_TO_PHONES.get(_pinyin(note))
    if mapping is None:
        return [_FALLBACK_PHONE]
    return mapping.split()


def _build_sequence(
    notes: Sequence[SungNote],
) -> tuple[_IntArray, _IntArray, NDArray[np.float32]]:
    """Build the per-phone model inputs from a sequence of sung notes.

    Each note contributes one model note window: all of its phones share the
    note's pitch, duration and a zero ``is_slur``, so every note starts a new
    note window.  The model uses the note duration carried by the syllable-final
    vowel to lay out the melody, so the full note duration is repeated on every
    phone token of the note.

    Args:
        notes: Notes to sing, in performance order.

    Returns:
        ``(token ids, MIDI pitches, durations)`` arrays shaped ``(1, phones)``.
    """
    token_ids: list[int] = []
    pitches: list[int] = []
    durations: list[float] = []
    for note in notes:
        for phone in _phones(note):
            token_ids.append(PHONE_IDS.get(phone, PHONE_IDS[_FALLBACK_PHONE]))
            pitches.append(note.midi)
            durations.append(note.duration)
    return (
        np.asarray([token_ids], dtype=np.int64),
        np.asarray([pitches], dtype=np.int64),
        np.asarray([durations], dtype=np.float32),
    )


def _run(
    session: Any, outputs: Sequence[str], feeds: Mapping[str, _Array]
) -> list[Any]:
    """Run one ONNX Runtime session.

    Args:
        session: An object exposing the ONNX Runtime ``run`` method.
        outputs: Names of the outputs to fetch.
        feeds: Mapping of input names to input arrays.

    Returns:
        The requested outputs, in the requested order.
    """
    return list(session.run(list(outputs), dict(feeds)))


def _sanitise(samples: _FloatArray) -> _FloatArray:
    """Replace non-finite samples with silence and clip them to ``[-1, 1]``.

    Args:
        samples: Mono samples to clean.

    Returns:
        The cleaned samples.
    """
    cleaned = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(cleaned, -1.0, 1.0)


def _resample(
    samples: _FloatArray, source_rate: int, target_rate: int = SAMPLE_RATE
) -> _FloatArray:
    """Resample a mono buffer with linear interpolation.

    Args:
        samples: Mono samples at ``source_rate``.
        source_rate: Sample rate of ``samples`` in Hz.
        target_rate: Requested sample rate in Hz.

    Returns:
        Mono float64 samples at ``target_rate``.
    """
    if samples.size == 0 or source_rate <= 0 or source_rate == target_rate:
        return samples
    length = round(samples.size * target_rate / source_rate)
    positions = np.arange(length, dtype=np.float64) * (source_rate / target_rate)
    source = np.arange(samples.size, dtype=np.float64)
    return cast(_FloatArray, np.interp(positions, source, samples))


class ChineseDiffSingerBackend:
    """A DiffSinger ONNX vocal backend for Chinese lyrics.

    The backend loads a model directory that contains ``config.yaml`` and the
    five ONNX files listed in :data:`_MODEL_FILES`.  A render is deterministic:
    the same notes always produce the same samples.
    """

    def __init__(self, model_dir: Path) -> None:
        """Load the configuration and open the ONNX sessions.

        Args:
            model_dir: Directory holding ``config.yaml`` and the model's ONNX files.

        Raises:
            FileNotFoundError: If ``config.yaml`` does not exist.
            RuntimeError: If ``onnxruntime`` is not installed.
        """
        self._model_dir = Path(model_dir)
        config = _load_config(self._model_dir / "config.yaml")
        self._sample_rate = int(config["audio_sample_rate"])
        self._k_steps = int(config["K_step"])
        self._spec_min = np.asarray(config["spec_min"], dtype=np.float32)
        self._spec_max = np.asarray(config["spec_max"], dtype=np.float32)
        sessions = [self._session(self._model_dir / name) for name in _MODEL_FILES]
        self._fs2, self._q_sample, self._p_sample, self._pe, self._vocoder = sessions

    @staticmethod
    def _session(path: Path) -> Any:
        """Create a CPU ONNX Runtime session for one model file.

        Args:
            path: Path of the ``.onnx`` file.

        Returns:
            An ONNX Runtime inference session.

        Raises:
            RuntimeError: If ``onnxruntime`` is not installed.
        """
        onnxruntime = _import_optional("onnxruntime")
        return onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

    def synthesize(self, notes: Sequence[SungNote]) -> _FloatArray:
        """Render Chinese notes to mono samples.

        Args:
            notes: Notes to sing, in performance order.

        Returns:
            Mono float64 samples at :data:`SAMPLE_RATE`, at least as long as
            ``max(onset + duration)`` seconds.  A short model tail is kept.
        """
        if not notes:
            return np.zeros(0, dtype=np.float64)
        txt_tokens, pitch_midi, midi_dur = _build_sequence(notes)
        decoder_inp, mel = self._encode(txt_tokens, pitch_midi, midi_dur)
        mel = self._diffuse(decoder_inp, mel)
        wav = self._vocode(mel)
        samples = _sanitise(np.asarray(wav, dtype=np.float64).reshape(-1))
        samples = _resample(samples, self._sample_rate)
        span = int(
            np.ceil(max(note.onset + note.duration for note in notes) * SAMPLE_RATE)
        )
        if samples.size < span:
            samples = np.pad(samples, (0, span - samples.size))
        return samples

    def _encode(
        self,
        txt_tokens: _IntArray,
        pitch_midi: _IntArray,
        midi_dur: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Run the FastSpeech2 model.

        Args:
            txt_tokens: Phone token ids, shaped ``(1, phones)``.
            pitch_midi: MIDI pitch per phone, shaped ``(1, phones)``.
            midi_dur: Duration in seconds per phone, shaped ``(1, phones)``.

        Returns:
            The decoder features and the mel spectrogram, both shaped
            ``(1, frames, ...)``.
        """
        outputs = _run(
            self._fs2,
            ["decoder_inp", "mel_out"],
            {
                "txt_tokens": txt_tokens,
                "pitch_midi": pitch_midi,
                "midi_dur": midi_dur,
                "is_slur": np.zeros_like(txt_tokens, dtype=np.int64),
            },
        )
        decoder_inp = np.asarray(outputs[0], dtype=np.float32)
        mel = np.asarray(outputs[1], dtype=np.float32)
        return decoder_inp, mel

    def _diffuse(
        self, decoder_inp: NDArray[np.float32], mel: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Normalise the mel, run the diffusion sampler and denormalise it.

        Args:
            decoder_inp: Decoder features from :meth:`_encode`.
            mel: Mel spectrogram from :meth:`_encode`.

        Returns:
            The denormalised mel spectrogram shaped ``(1, frames, bins)``.
        """
        normalised = (mel - self._spec_min) / (
            self._spec_max - self._spec_min
        ) * 2.0 - 1.0
        current = normalised.transpose([0, 2, 1])[:, None, :, :].astype(np.float32)
        rng = np.random.default_rng(_SEED)
        noise = rng.standard_normal(current.shape).astype(np.float32)
        current = np.asarray(
            _run(
                self._q_sample,
                ["x_next"],
                {
                    "x_start": current,
                    "noise": noise,
                    "t": np.asarray([self._k_steps - 1], dtype=np.int64),
                },
            )[0],
            dtype=np.float32,
        )
        condition = decoder_inp.transpose([0, 2, 1])
        for step in reversed(range(self._k_steps)):
            noise = rng.standard_normal(current.shape).astype(np.float32)
            current = np.asarray(
                _run(
                    self._p_sample,
                    ["x_next"],
                    {
                        "x": current,
                        "noise": noise,
                        "t": np.asarray([step], dtype=np.int64),
                        "cond": condition,
                    },
                )[0],
                dtype=np.float32,
            )
        denormalised = current[:, 0].transpose([0, 2, 1])
        denormalised = (denormalised + 1.0) / 2.0 * (
            self._spec_max - self._spec_min
        ) + self._spec_min
        return cast(NDArray[np.float32], denormalised.astype(np.float32))

    def _vocode(self, mel: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict f0 and render a waveform with the vocoder.

        Args:
            mel: Denormalised mel spectrogram shaped ``(1, frames, bins)``.

        Returns:
            The mono waveform shaped ``(1, samples)``.
        """
        f0 = np.asarray(
            _run(self._pe, ["f0_denorm_pred"], {"mel_input": mel})[0],
            dtype=np.float32,
        )
        return np.asarray(
            _run(self._vocoder, ["wav_out"], {"mel_out": mel, "f0": f0})[0],
            dtype=np.float32,
        )
