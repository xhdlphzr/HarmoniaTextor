# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Automatic selection between DiffSinger ONNX models and the FOF singer.

DiffSinger gives a much more natural voice than the built-in FOF synthesis, but
it needs a model and a CPU/GPU that can run diffusion inference.  This backend
is deliberately defensive: it only uses DiffSinger for Chinese/English lyrics
when the matching model is present and the machine meets a conservative CPU
threshold, and silently falls back to :class:`FOFSinger` otherwise (or when a
model fails to load).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from harmoniatextor.render.vocal import diffsinger_models
from harmoniatextor.render.vocal.singer import FOFSinger, SungNote, VocalBackend

__all__ = ["AutoVocalBackend", "normalise_language"]

_FALLBACK_CORES = 4
#: English voices whose average pitch is below this use the male voicebank.
#: Covers bass and tenor lines; soprano/alto stay on the female Tiger bank.
_MALE_MAX_MIDI = 64.0
_LANGUAGE_ALIASES = {
    "zh": "zh",
    "cmn": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "chinese": "zh",
    "en": "en",
    "eng": "en",
    "en-us": "en",
    "en-gb": "en",
    "english": "en",
}


class _LanguageBackend(Protocol):
    """A model-backed DiffSinger backend."""

    def synthesize(self, notes: Sequence[SungNote]) -> NDArray[np.float64]:
        """Render notes to mono samples."""
        raise NotImplementedError


def normalise_language(language: str) -> str:
    """Normalise a language tag to ``zh``/``en``.

    Args:
        language: Raw language tag from the score.

    Returns:
        The normalised tag, or an empty string when unsupported.
    """
    return _LANGUAGE_ALIASES.get(language.strip().lower().replace("_", "-"), "")


def _performance_ok() -> bool:
    """Return whether this machine should attempt DiffSinger inference.

    Returns:
        ``True`` when the feature is enabled and the CPU has enough cores.
    """
    if os.environ.get("HARMONIA_DIFFSINGER", "auto").lower() in {"0", "off", "false"}:
        return False
    try:
        minimum = int(os.environ.get("HARMONIA_DIFFSINGER_MIN_CORES", _FALLBACK_CORES))
    except ValueError:
        minimum = _FALLBACK_CORES
    return (os.cpu_count() or 1) >= minimum


def _male_max_midi() -> float:
    """Return the pitch threshold below which the male model is used.

    Returns:
        The configured MIDI threshold (default 64, roughly E4).
    """
    raw = os.environ.get("HARMONIA_DIFFSINGER_MALE_MAX_MIDI", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return _MALE_MAX_MIDI


def _english_options() -> dict[str, Any]:
    """Read optional English Tiger quality knobs from the environment.

    Returns:
        Keyword arguments for :class:`EnglishDiffSingerBackend`.
    """
    options: dict[str, Any] = {}
    steps = os.environ.get("HARMONIA_DIFFSINGER_STEPS", "").strip()
    if steps:
        try:
            options["steps"] = max(1, int(steps))
        except ValueError:
            pass
    speaker = os.environ.get("HARMONIA_DIFFSINGER_SPEAKER", "").strip()
    if speaker:
        options["speaker"] = speaker
    return options


def _male_options() -> dict[str, Any]:
    """Return quality options for the Ameko Kero male voicebank.

    Returns:
        Keyword arguments safe for the male voicebank (no Tiger speaker name).
    """
    options = _english_options()
    options.pop("speaker", None)
    return options


class AutoVocalBackend:
    """A vocal backend that picks DiffSinger when it can, FOF otherwise."""

    def __init__(self) -> None:
        """Initialise the selector with a lazily loaded FOF fallback."""
        self._fallback: VocalBackend = FOFSinger()
        self._backends: dict[str, _LanguageBackend] = {}

    def synthesize(self, notes: Sequence[SungNote]) -> NDArray[np.float64]:
        """Render notes with DiffSinger when possible, else FOF.

        Args:
            notes: Notes to sing.

        Returns:
            Mono float samples at the FOF sample rate.
        """
        if not notes:
            return np.zeros(0, dtype=np.float64)
        language = self._language(notes)
        if language and _performance_ok():
            key = self._model_key(notes, language)
            candidates = [key]
            if key == "zh_male":
                candidates.append("zh")
            elif key == "en_male":
                candidates.append("en")
            for candidate in candidates:
                backend = self._load(candidate)
                if backend is None:
                    continue
                try:
                    samples = np.asarray(backend.synthesize(notes), dtype=np.float64)
                except Exception:  # noqa: BLE001 - FOF is the safety net
                    samples = np.zeros(0, dtype=np.float64)
                if samples.size > 0 and np.all(np.isfinite(samples)):
                    return samples
        return self._fallback.synthesize(notes)

    def _model_key(self, notes: Sequence[SungNote], language: str) -> str:
        """Choose the concrete model for a line.

        English lines below the male-pitch threshold use the Ameko Kero male
        voicebank; higher lines use the Tiger voicebank.  Missing models simply
        fall through to whichever English model is installed, then FOF.

        Args:
            notes: Notes to inspect.
            language: Normalised language (``zh``/``en``).

        Returns:
            A model key, or an empty string when no model is installed.
        """
        if language == "zh":
            return (
                "zh_male"
                if diffsinger_models.model_ready("en_male")
                else ("zh" if diffsinger_models.model_ready("zh") else "")
            )
        average = sum(note.midi for note in notes) / len(notes)
        if average < _male_max_midi() and diffsinger_models.model_ready("en_male"):
            return "en_male"
        if diffsinger_models.model_ready("en"):
            return "en"
        if diffsinger_models.model_ready("en_male"):
            return "en_male"
        return ""

    def _language(self, notes: Sequence[SungNote]) -> str:
        """Return the single supported language of the notes, if any.

        Args:
            notes: Notes to inspect.

        Returns:
            ``zh``, ``en`` or an empty string for mixed/unsupported lyrics.
        """
        languages = {
            normalise_language(note.language) for note in notes if note.text.strip()
        }
        if len(languages) != 1:
            return ""
        language = languages.pop()
        return language if language in diffsinger_models.LANGUAGES else ""

    def _load(self, language: str) -> _LanguageBackend | None:
        """Load and cache a model backend.

        Args:
            language: Model key (``zh``, ``en`` or ``en_male``).

        Returns:
            The backend, or ``None`` when its optional dependency is missing.
        """
        cached = self._backends.get(language)
        if cached is not None:
            return cached
        try:
            if language == "zh":
                from harmoniatextor.render.vocal.diffsinger_zh import (
                    ChineseDiffSingerBackend,
                )

                backend: _LanguageBackend = ChineseDiffSingerBackend(
                    diffsinger_models.model_dir("zh")
                )
            elif language == "zh_male":
                from harmoniatextor.render.vocal.diffsinger_zh_ameko import (
                    AmekoChineseDiffSingerBackend,
                )

                backend = AmekoChineseDiffSingerBackend(
                    diffsinger_models.model_dir("en_male"),
                    **_male_options(),
                )
            else:
                from harmoniatextor.render.vocal.diffsinger_en import (
                    EnglishDiffSingerBackend,
                )

                options = _english_options()
                if language != "en":
                    options.pop("speaker", None)
                backend = EnglishDiffSingerBackend(
                    diffsinger_models.model_dir(language), **options
                )
        except Exception:  # noqa: BLE001 - optional dependency/model failure
            return None
        self._backends[language] = backend
        return backend
