# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Chinese singing synthesis with the Ameko Kero DiffSinger voicebank.

The Ameko Kero bank is a male Millefeuille-format DiffSinger voicebank that also
ships a Chinese phoneme set (symbols prefixed with ``zh/``) and a pinyin
dictionary (``dsdur/dsdict-zh.yaml``).  This module implements the Chinese
frontend for it:

* every note's Chinese text is converted to toneless pinyin with :mod:`pypinyin`;
* each pinyin syllable is looked up in the bank's dictionary (whose
  ``grapheme`` entries are pinyin syllables) to get its ``zh/...`` phones;
* the note's frames are split evenly over those phones;
* the shared DiffSinger acoustic model turns tokens, durations and an ``f0``
  curve into a mel spectrogram, and the HiFi-GAN vocoder turns that into a
  44.1 kHz waveform.

Voicebank discovery and the ONNX helpers are reused from
:mod:`harmoniatextor.render.vocal.diffsinger_en`.  ``onnxruntime`` and
``pypinyin`` are imported lazily, so importing this module never requires them;
a clear :class:`RuntimeError` is raised when they are needed but missing.
Notes are assumed to arrive in performance order with contiguous durations;
their onsets are only used to place the rendered line on the output timeline
and to size the returned buffer.
"""

from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from harmoniatextor.render.vocal.diffsinger_en import (
    NUM_MEL_BINS,
    SPEAKER_DIM,
    _create_session,
    _Dictionary,
    _find_vocoder,
    _load_dictionary,
    _load_languages,
    _load_phonemes,
    _load_speaker_embedding,
    _read_yaml,
    _resolve_speaker,
    _seconds_to_frames,
    _split_frames,
)
from harmoniatextor.render.vocal.singer import SAMPLE_RATE, SungNote

__all__ = ["DEFAULT_DEPTH", "DEFAULT_STEPS", "AmekoChineseDiffSingerBackend"]

#: Conservative diffusion step count: enough quality for a singing stem without
#: making CPU inference slow.
DEFAULT_STEPS = 12
#: Diffusion depth, matching the voicebank's configured ``max_depth``.
DEFAULT_DEPTH = 0.6
#: Language tag and id key of the Chinese frontend.
_LANGUAGE = "zh"
#: Phone symbol used for syllables that have no dictionary entry.
_FALLBACK_PHONE = "zh/a"
#: Root-config paths of the Millefeuille/Ameko layout, used when a bank has no
#: root ``dsconfig.yaml`` of its own.
_DEFAULT_ACOUSTIC = "onnx/millefeuille_b8.onnx"
_DEFAULT_PHONEMES = "lang/millefeuille_b8.phonemes.json"
_DEFAULT_VOCODER = "dsvocoder"

_FloatArray = NDArray[np.float64]


def _import_pypinyin() -> Any:
    """Import the optional ``pypinyin`` package.

    The import suppresses pypinyin's own ``codecs.open`` deprecation warning so
    strict warning filters (``-W error``) cannot turn a lazy import into an
    error.

    Returns:
        The imported module.

    Raises:
        RuntimeError: When the package is not installed.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return cast(Any, importlib.import_module("pypinyin"))
    except ImportError as exc:
        raise RuntimeError(
            "pypinyin is required for the Chinese Ameko DiffSinger backend"
        ) from exc


def _pinyin_syllables(text: str) -> list[str]:
    """Convert Chinese text to lower-case, toneless pinyin syllables.

    Args:
        text: Note text.  Non-Chinese characters are passed through unchanged
            by :mod:`pypinyin`.

    Returns:
        One non-empty syllable per character, in order; empty for blank text.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    pypinyin = _import_pypinyin()
    syllables = pypinyin.lazy_pinyin(cleaned, style=pypinyin.Style.NORMAL)
    return [
        str(syllable).strip().lower() for syllable in syllables if str(syllable).strip()
    ]


def _phones_for_text(dictionary: _Dictionary, text: str) -> list[str]:
    """Return the model phone symbols for one note's text.

    Args:
        dictionary: The voicebank's Chinese pinyin dictionary.
        text: Note text.

    Returns:
        A non-empty phone list.  A syllable without a dictionary entry (and a
        note without any pinyin) falls back to ``zh/a`` so nothing is silent.
    """
    phones: list[str] = []
    for syllable in _pinyin_syllables(text):
        phones.extend(dictionary.words.get(syllable) or [_FALLBACK_PHONE])
    return phones or [_FALLBACK_PHONE]


def _language_path(base: Path, config: dict[str, Any], phonemes: Path) -> Path:
    """Return the voicebank's language table path.

    Args:
        base: Voicebank root.
        config: Root ``dsconfig.yaml`` mapping, or an empty mapping when absent.
        phonemes: Phoneme table path, used to derive the sibling JSON name.

    Returns:
        The configured language table path, or the table next to the phonemes.
    """
    if "languages" in config:
        return base / str(config["languages"])
    stem = phonemes.name.removesuffix(".phonemes.json")
    return phonemes.parent / f"{stem}.languages.json"


class AmekoChineseDiffSingerBackend:
    """Chinese singing synthesis with the Ameko Kero DiffSinger voicebank.

    The frontend is deterministic: the same notes always produce the same
    tokens, durations and ``f0`` curve.  The acoustic graphs themselves may
    contain random operators, so the neural waveform is not guaranteed to be
    bit-identical between calls even for identical inputs.
    """

    def __init__(
        self,
        voicebank_dir: Path,
        *,
        steps: int = DEFAULT_STEPS,
        depth: float = DEFAULT_DEPTH,
        speaker: str | None = None,
    ) -> None:
        """Load the voicebank's Chinese frontend, speaker and ONNX models.

        Args:
            voicebank_dir: Unpacked Ameko Kero voicebank directory.
            steps: Diffusion sampling steps handed to the acoustic model.
            depth: Diffusion depth in ``(0, max_depth]``.
            speaker: Speaker name, voicebank-relative ``.emb`` path, or absolute
                ``.emb`` path; ``None`` selects the bank's neutral speaker.

        Raises:
            RuntimeError: When ``onnxruntime`` is not installed.
            ValueError: When a voicebank file is missing, malformed, or has no
                Chinese language id.
        """
        base = Path(voicebank_dir)
        config_path = base / "dsconfig.yaml"
        config = _read_yaml(config_path) if config_path.is_file() else {}
        acoustic = base / str(config.get("acoustic", _DEFAULT_ACOUSTIC))
        phonemes = base / str(config.get("phonemes", _DEFAULT_PHONEMES))
        languages_path = _language_path(base, config, phonemes)
        languages = _load_languages(languages_path) if languages_path.is_file() else {}
        self._steps = steps
        self._depth = depth
        self._phoneme_ids = _load_phonemes(phonemes)
        self._dictionary = _load_dictionary(base / "dsdur" / "dsdict-zh.yaml")
        self._speaker = _load_speaker_embedding(_resolve_speaker(base, config, speaker))
        self._fallback_id = self._phoneme_ids.get(
            _FALLBACK_PHONE,
            self._phoneme_ids.get("AP", next(iter(self._phoneme_ids.values()))),
        )
        self._symbol_prefix = (
            f"{_LANGUAGE}/"
            if any(symbol.startswith(f"{_LANGUAGE}/") for symbol in self._phoneme_ids)
            else ""
        )
        self._acoustic = _create_session(acoustic)
        inputs: set[str] = {item.name for item in self._acoustic.get_inputs()}
        self._language_id: int | None = None
        if "languages" in inputs:
            try:
                self._language_id = languages[_LANGUAGE]
            except KeyError as exc:
                raise ValueError(
                    f"voicebank has no Chinese language id in {base}"
                ) from exc
        vocoder_dir = base / str(config.get("vocoder", _DEFAULT_VOCODER))
        self._vocoder = _create_session(_find_vocoder(acoustic, vocoder_dir))

    def _token_id(self, phone: str) -> int:
        """Map one phone symbol to a model token id.

        Args:
            phone: Phone symbol from the Chinese dictionary.

        Returns:
            The model token id.  Unknown phones fall back to the dictionary's
            replacement table, the model's language prefix, then ``zh/a``.
        """
        token = self._phoneme_ids.get(phone)
        if token is not None:
            return token
        replacement = self._dictionary.replacements.get(phone)
        if replacement is not None and replacement in self._phoneme_ids:
            return self._phoneme_ids[replacement]
        return self._phoneme_ids.get(f"{self._symbol_prefix}{phone}", self._fallback_id)

    def synthesize(self, notes: Sequence[SungNote]) -> _FloatArray:
        """Render Chinese notes to mono float samples at 44.1 kHz.

        Args:
            notes: Notes to sing, in performance order.

        Returns:
            Mono float64 samples covering ``max(onset + duration)`` seconds.
        """
        if not notes:
            return np.zeros(0, dtype=np.float64)
        lengths = [max(0.0, float(note.duration)) for note in notes]
        boundaries = [0.0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        frames = [
            _seconds_to_frames(boundaries[index + 1])
            - _seconds_to_frames(boundaries[index])
            for index in range(len(notes))
        ]
        pitches = [
            0.0 if note.midi <= 0 else 440.0 * 2.0 ** ((note.midi - 69) / 12.0)
            for note in notes
        ]
        tokens: list[int] = []
        durations: list[int] = []
        f0: list[float] = []
        for index, note in enumerate(notes):
            phones = _phones_for_text(self._dictionary, note.text)
            for phone, count in zip(phones, _split_frames(frames[index], len(phones))):
                tokens.append(self._token_id(phone))
                durations.append(count)
            f0.extend([pitches[index]] * frames[index])
        total_frames = sum(durations)
        if total_frames <= 0:
            return np.zeros(0, dtype=np.float64)
        f0_array = np.asarray(f0, dtype=np.float32).reshape(1, total_frames)
        speaker = self._speaker.astype(np.float32)
        spk_embed = np.ascontiguousarray(
            np.broadcast_to(speaker, (1, total_frames, SPEAKER_DIM))
        )
        feeds: dict[str, NDArray[Any]] = {
            "tokens": np.asarray([tokens], dtype=np.int64),
            "durations": np.asarray([durations], dtype=np.int64),
            "f0": f0_array,
            "gender": np.zeros((1, total_frames), dtype=np.float32),
            "velocity": np.ones((1, total_frames), dtype=np.float32),
            "spk_embed": spk_embed,
            "depth": np.asarray(self._depth, dtype=np.float32),
            "steps": np.asarray(self._steps, dtype=np.int64),
        }
        if self._language_id is not None:
            feeds["languages"] = np.full(
                (1, len(tokens)), self._language_id, dtype=np.int64
            )
        mel = np.asarray(self._acoustic.run(["mel"], feeds)[0], dtype=np.float32)
        mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        mel = mel.reshape(-1, NUM_MEL_BINS)[np.newaxis]
        wave = np.asarray(
            self._vocoder.run(["waveform"], {"mel": mel, "f0": f0_array})[0],
            dtype=np.float32,
        )
        samples = np.nan_to_num(
            wave.reshape(-1).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        end = math.ceil(
            max(float(note.onset) + max(0.0, float(note.duration)) for note in notes)
            * SAMPLE_RATE
        )
        start = min(round(max(0.0, float(notes[0].onset)) * SAMPLE_RATE), end)
        output = np.zeros(max(end, 0), dtype=np.float64)
        stop = min(output.size, start + samples.size)
        output[start:stop] = samples[: stop - start]
        return output
