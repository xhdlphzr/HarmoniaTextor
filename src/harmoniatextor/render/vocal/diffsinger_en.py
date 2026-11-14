# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""An English DiffSinger ONNX singing backend.

DiffSinger renders singing from phoneme tokens, per-token durations and a
per-frame fundamental-frequency curve.  This module implements the small amount
of frontend needed for English DiffSinger voicebanks:

* consecutive :class:`~harmoniatextor.render.vocal.singer.SungNote` objects are
  grouped into words using their ``syllabic`` role and text;
* each word is looked up in the voicebank's phoneme dictionary (falling back to
  a deterministic built-in grapheme-to-ARPAbet heuristic);
* the word's phones are distributed over its notes, and its frames are
  distributed over the phones;
* the acoustic model turns tokens into a mel spectrogram and the HiFi-GAN
  vocoder turns that into a 44.1 kHz waveform.

Two OpenUTAU voicebank layouts are detected automatically.  A Tiger v106 style
bank has no root ``dsconfig.yaml`` and uses::

    voicebank/
        dsacoustic/acoustic.onnx
        dsacoustic/phonemes.txt
        dsacoustic/<speaker>.emb
        dsdur/dsdict-en.yaml
        dsvocoder/*.onnx

A newer Millefeuille-style bank (for example Ameko Kero) has a root
``dsconfig.yaml`` with paths to everything else::

    voicebank/
        dsconfig.yaml
        onnx/<voicebank>.onnx
        lang/<voicebank>.phonemes.json
        lang/<voicebank>.languages.json
        embs/<voicebank>.<speaker>.emb
        dsdur/dsdict-en.yaml
        dsvocoder/*.onnx

When a root configuration exists its ``acoustic``/``phonemes``/``languages``/
``vocoder`` keys are honoured.  JSON phoneme tables map symbol to id and use
language prefixes such as ``en/aa``; their ``dsdict-en.yaml`` frequently lists
only special entries plus a bare-to-prefixed ``replacements`` table, so the
built-in heuristic supplies the word phones and the replacement/prefix step maps
them into the model's symbol space.  When the acoustic model has a ``languages``
input, the English language id from ``languages.json`` is sent for every token.

``onnxruntime`` is an optional dependency: importing this module never requires
it, and a clear :class:`RuntimeError` is raised when a backend is built without
it.  The backend assumes notes arrive in performance order with contiguous
durations (as :func:`~harmoniatextor.render.vocal.extract.vocal_lines` produces
them); note onsets are only used to place the rendered line on the output
timeline and to size the returned buffer.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from harmoniatextor.render.vocal.singer import SAMPLE_RATE, SungNote

__all__ = ["DEFAULT_DEPTH", "DEFAULT_STEPS", "EnglishDiffSingerBackend"]

#: Audio frames per second of the DiffSinger mel spectrogram.
HOP_SIZE = 512
#: Number of mel bins produced by the acoustic model and consumed by the vocoder.
NUM_MEL_BINS = 128
#: Size of one speaker embedding stored in the voicebank's ``.emb`` files.
SPEAKER_DIM = 256
#: Conservative diffusion step count: enough quality for a singing stem without
#: making CPU inference slow.
DEFAULT_STEPS = 12
#: Diffusion depth, matching the voicebanks' configured ``max_depth``.
DEFAULT_DEPTH = 0.6
#: Preferred default speaker; when a bank has no such embedding a neutral (or
#: first discovered) speaker is used instead.
DEFAULT_SPEAKER = "tiger_fresh"
#: Language prefix and id used by the English frontend.
_LANGUAGE = "en"

#: ``syllabic`` values that continue the previous word.
_CONTINUATION = frozenset({"middle", "end"})
#: Vowel used when a word has no letters to convert.
_DEFAULT_VOWEL = "a"
#: Grapheme pairs that map to one or two phones.  A pair is always tested
#: before its first letter, so ``th`` becomes ``th`` rather than ``t`` + ``hh``.
_G2P_RULES: dict[str, tuple[str, ...]] = {
    "th": ("th",),
    "sh": ("sh",),
    "ch": ("ch",),
    "ph": ("f",),
    "wh": ("w",),
    "ck": ("k",),
    "ng": ("ng",),
    "qu": ("k", "w"),
    "ee": ("iy",),
    "ea": ("iy",),
    "oo": ("uw",),
    "ou": ("aw",),
    "ow": ("ow",),
    "oa": ("ow",),
    "oi": ("oy",),
    "oy": ("oy",),
    "au": ("ao",),
    "aw": ("ao",),
    "ai": ("ey",),
    "ay": ("ey",),
    "ei": ("ey",),
    "ie": ("iy",),
    "ar": ("aa", "r"),
    "er": ("er",),
    "ir": ("er",),
    "ur": ("er",),
    "or": ("ao", "r"),
}
#: Single-grapheme fallback mapping, roughly General American ARPAbet.
_G2P_LETTERS: dict[str, tuple[str, ...]] = {
    "a": ("ae",),
    "b": ("b",),
    "c": ("k",),
    "d": ("d",),
    "e": ("eh",),
    "f": ("f",),
    "g": ("g",),
    "h": ("hh",),
    "i": ("ih",),
    "j": ("jh",),
    "k": ("k",),
    "l": ("l",),
    "m": ("m",),
    "n": ("n",),
    "o": ("aa",),
    "p": ("p",),
    "q": ("k",),
    "r": ("r",),
    "s": ("s",),
    "t": ("t",),
    "u": ("ah",),
    "v": ("v",),
    "w": ("w",),
    "x": ("k", "s"),
    "y": ("iy",),
    "z": ("z",),
}

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _Dictionary:
    """A voicebank's word entries and symbol replacement table.

    Attributes:
        words: Word grapheme to bare phone list.
        replacements: Bare phone to model-specific symbol, used by banks whose
            phone table is language-prefixed.
    """

    words: dict[str, list[str]]
    replacements: dict[str, str]


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        TypeError: When the file does not contain a mapping.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected a YAML mapping in {path}")
    return cast("dict[str, Any]", data)


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``.

    Args:
        path: File to read.

    Returns:
        The parsed object.

    Raises:
        TypeError: When the file does not contain a JSON object.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return cast("dict[str, Any]", data)


def _load_phonemes(path: Path) -> dict[str, int]:
    """Load the voicebank's phoneme table.

    Args:
        path: ``phonemes.txt`` (one symbol per line in model id order) or a
            ``phonemes.json`` object mapping symbol to token id.

    Returns:
        Mapping from phoneme symbol to its model token id.

    Raises:
        ValueError: When the table contains no phonemes.
    """
    if path.suffix.lower() == ".json":
        phonemes = {
            str(symbol): int(token) for symbol, token in _read_json(path).items()
        }
    else:
        lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
        phones = [line for line in lines if line]
        phonemes = {phone: index for index, phone in enumerate(phones)}
    if not phonemes:
        raise ValueError(f"no phonemes in {path}")
    return phonemes


def _load_languages(path: Path) -> dict[str, int]:
    """Load a DiffSinger language table.

    Args:
        path: ``languages.json`` mapping language tag to model language id.

    Returns:
        Mapping from language tag to its model id.

    Raises:
        ValueError: When the table contains no languages.
    """
    languages = {str(key): int(value) for key, value in _read_json(path).items()}
    if not languages:
        raise ValueError(f"no languages in {path}")
    return languages


def _load_dictionary(path: Path) -> _Dictionary:
    """Load word entries and symbol replacements from ``dsdict-en.yaml``.

    Args:
        path: ``dsdict-en.yaml``.  Tiger banks provide word entries; newer banks
            may only provide special entries plus a ``replacements`` table
            mapping bare ARPAbet symbols to language-prefixed symbols.

    Returns:
        The word dictionary and replacement table.

    Raises:
        ValueError: When the file has neither word entries nor replacements.
    """
    data = _read_yaml(path)
    words: dict[str, list[str]] = {}
    for item in data.get("entries", []):
        grapheme = str(item.get("grapheme", "")).strip().lower()
        phones = [str(phone) for phone in item.get("phonemes", [])]
        if grapheme and phones:
            words[grapheme] = phones
    replacements: dict[str, str] = {}
    for item in data.get("replacements", []):
        source = str(item.get("from", ""))
        target = str(item.get("to", ""))
        if source and target:
            replacements[source] = target
    if not words and not replacements:
        raise ValueError(f"no dictionary entries in {path}")
    return _Dictionary(words=words, replacements=replacements)


def _load_speaker_embedding(path: Path) -> _FloatArray:
    """Load a raw little-endian float32 speaker embedding.

    Args:
        path: ``.emb`` file holding 256 float32 values.

    Returns:
        The embedding as a float64 vector.

    Raises:
        ValueError: When the file does not hold exactly 256 floats.
    """
    embedding = np.frombuffer(path.read_bytes(), dtype="<f4").astype(np.float64)
    if embedding.size != SPEAKER_DIM:
        raise ValueError(
            f"expected {SPEAKER_DIM} floats in {path}, got {embedding.size}"
        )
    return embedding


def _speaker_paths(base: Path, config: dict[str, Any]) -> list[Path]:
    """Return the existing speaker embeddings of a voicebank.

    Configured speakers come first, then any ``embs/*.emb`` and
    ``dsacoustic/*.emb`` files that were not listed.

    Args:
        base: Voicebank root.
        config: Root ``dsconfig.yaml`` mapping, or an empty mapping when absent.

    Returns:
        Paths to existing ``.emb`` files.
    """
    candidates: dict[Path, None] = {}
    for entry in config.get("speakers", []):
        path = base / str(entry)
        if path.suffix != ".emb":
            path = path.parent / f"{path.name}.emb"
        if path.is_file():
            candidates.setdefault(path, None)
    for pattern in ("embs/*.emb", "dsacoustic/*.emb"):
        for path in sorted(base.glob(pattern)):
            candidates.setdefault(path, None)
    return list(candidates)


def _resolve_speaker(base: Path, config: dict[str, Any], requested: str | None) -> Path:
    """Resolve the speaker embedding to use.

    Args:
        base: Voicebank root.
        config: Root ``dsconfig.yaml`` mapping, or an empty mapping when absent.
        requested: Speaker name, voicebank-relative ``.emb`` path or absolute
            ``.emb`` path; ``None`` selects the bank's default.

    Returns:
        Path to the speaker embedding.

    Raises:
        ValueError: When the requested speaker does not exist or the voicebank
            contains no speaker embeddings.
    """
    speakers = _speaker_paths(base, config)
    if requested is None:
        for path in speakers:
            if path.stem == DEFAULT_SPEAKER:
                return path
        for path in speakers:
            if "neutral" in path.stem.lower():
                return path
        if speakers:
            return speakers[0]
        raise ValueError(f"no speaker embeddings in {base}")
    cleaned = requested.strip()
    wanted = cleaned.replace("\\", "/")
    for path in speakers:
        relative = str(path.relative_to(base)).replace("\\", "/")
        keys = {path.name, path.stem, relative, relative.removesuffix(".emb")}
        if wanted in keys:
            return path
    direct = base / cleaned
    if direct.is_file():
        return direct
    raise ValueError(f"speaker {requested!r} not found in {base}")


def _find_vocoder(acoustic: Path, directory: Path) -> Path:
    """Select the vocoder model belonging to an acoustic model.

    Args:
        acoustic: Acoustic ONNX path, used to recognise the voicebank's own
            vocoder among several models.
        directory: Vocoder directory that should contain ``*.onnx`` models.

    Returns:
        The vocoder model path.

    Raises:
        ValueError: When the directory contains no ONNX model.
    """
    models = sorted(directory.glob("*.onnx"))
    if not models:
        raise ValueError(f"no ONNX model in {directory}")
    if len(models) == 1:
        return models[0]
    prefix = acoustic.stem.split("_")[0].lower()
    for model in models:
        if prefix and prefix in model.stem.lower():
            return model
    return models[0]


def _create_session(model_path: Path) -> Any:
    """Create a CPU ONNX Runtime inference session.

    Args:
        model_path: Path to the ``.onnx`` model.

    Returns:
        The inference session.

    Raises:
        RuntimeError: When the optional ``onnxruntime`` package is missing.
    """
    try:
        runtime = cast("Any", importlib.import_module("onnxruntime"))
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for the English DiffSinger backend"
        ) from exc
    return runtime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _group_words(notes: Sequence[SungNote]) -> list[list[int]]:
    """Group note indices into word-sized runs.

    A note continues the previous word when its ``syllabic`` role is
    ``middle``/``end`` or when it has no text (a melisma continuation).  Every
    other note starts a new word.

    Args:
        notes: Notes in performance order.

    Returns:
        One list of note indices per word.
    """
    groups: list[list[int]] = []
    for index, note in enumerate(notes):
        text = note.text.strip()
        continuation = note.syllabic in _CONTINUATION or (
            not text and note.syllabic != "begin"
        )
        if groups and continuation:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _word_text(notes: Sequence[SungNote], indices: Sequence[int]) -> str:
    """Join a word's note texts into a lower-case dictionary key.

    Args:
        notes: Notes in performance order.
        indices: Indices belonging to the word.

    Returns:
        The concatenated letters and apostrophes of the word.
    """
    return "".join(
        character
        for index in indices
        for character in notes[index].text.strip().lower()
        if character.isalpha() or character == "'"
    )


def _fallback_phones(word: str) -> list[str]:
    """Convert an unknown word to ARPAbet-like phones with a small heuristic.

    This is deliberately not a real G2P model: it prevents a dictionary miss
    from silencing a word, while avoiding a new runtime dependency.

    Args:
        word: Lower-case word with punctuation removed.

    Returns:
        A non-empty phone list; a neutral vowel when the word has no letters.
    """
    phones: list[str] = []
    index = 0
    while index < len(word):
        pair = word[index : index + 2]
        if pair in _G2P_RULES:
            phones.extend(_G2P_RULES[pair])
            index += 2
        else:
            phones.extend(_G2P_LETTERS.get(word[index], ()))
            index += 1
    return phones or [_DEFAULT_VOWEL]


def _split_counts(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` items over weighted slots, giving every slot one item.

    Args:
        total: Number of items to allocate; at least ``len(weights)``.
        weights: Relative size of each slot (for example note frame counts).

    Returns:
        A count per slot that sums to ``total`` and is at least one.
    """
    count = len(weights)
    positive = [max(float(weight), 0.0) for weight in weights]
    total_weight = sum(positive)
    if total_weight <= 0.0:
        positive = [1.0] * count
        total_weight = float(count)
    shares = [total * weight / total_weight for weight in positive]
    counts = [max(1, int(share)) for share in shares]
    while sum(counts) < total:
        index = max(
            range(count), key=lambda position: shares[position] - counts[position]
        )
        counts[index] += 1
    while sum(counts) > total:
        index = max(range(count), key=counts.__getitem__)
        counts[index] -= 1
    return counts


def _split_frames(total: int, count: int) -> list[int]:
    """Split a note's frames evenly over its phones.

    Args:
        total: Frames available for the note.
        count: Number of phones in the note.

    Returns:
        One frame count per phone; all zeros when the note has no duration.
    """
    if count <= 0:
        return []
    if total <= 0:
        return [0] * count
    base, remainder = divmod(total, count)
    counts = [base] * count
    for index in range(remainder):
        counts[index] += 1
    return counts


def _distribute_phones(phones: Sequence[str], frames: Sequence[int]) -> list[list[str]]:
    """Assign a word's phones to its notes.

    When there are at least as many phones as notes, phones are split
    proportionally to note length.  When there are fewer phones than notes (a
    melisma), each extra note repeats the word's last phone.

    Args:
        phones: The word's phones, in order.
        frames: Frame length of each note in the word.

    Returns:
        One phone list per note, in order.
    """
    if len(phones) >= len(frames):
        counts = _split_counts(len(phones), [float(frame) for frame in frames])
        chunks: list[list[str]] = []
        start = 0
        for count in counts:
            chunks.append(list(phones[start : start + count]))
            start += count
        return chunks
    chunks = [[phone] for phone in phones]
    chunks.extend([[phones[-1]]] * (len(frames) - len(phones)))
    return chunks


def _seconds_to_frames(seconds: float) -> int:
    """Convert seconds to a rounded frame index.

    Args:
        seconds: Time in seconds.

    Returns:
        The equivalent (rounded) frame count.
    """
    return round(seconds * SAMPLE_RATE / HOP_SIZE)


class EnglishDiffSingerBackend:
    """English singing synthesis with a DiffSinger ONNX voicebank.

    The frontend is deterministic: the same notes always yield the same tokens,
    durations and ``f0`` curve.  The acoustic graphs themselves may contain
    random operators, so the neural waveform is not guaranteed to be
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
        """Load the voicebank's text frontend, speaker and ONNX models.

        Args:
            voicebank_dir: Unpacked OpenUTAU DiffSinger voicebank directory.
            steps: Diffusion sampling steps handed to the acoustic model.
            depth: Diffusion depth in ``(0, max_depth]``.
            speaker: Speaker name, voicebank-relative ``.emb`` path, or absolute
                ``.emb`` path; ``None`` selects the bank's default speaker.

        Raises:
            RuntimeError: When ``onnxruntime`` is not installed.
            ValueError: When a voicebank file is missing, malformed, or has no
                English language id.
        """
        base = Path(voicebank_dir)
        config_path = base / "dsconfig.yaml"
        config = _read_yaml(config_path) if config_path.is_file() else {}
        acoustic = base / str(config.get("acoustic", "dsacoustic/acoustic.onnx"))
        phonemes = base / str(config.get("phonemes", "dsacoustic/phonemes.txt"))
        vocoder_dir = base / str(config.get("vocoder", "dsvocoder"))
        languages = (
            _load_languages(base / str(config["languages"]))
            if "languages" in config
            else {}
        )
        self._steps = steps
        self._depth = depth
        self._phoneme_ids = _load_phonemes(phonemes)
        self._dictionary = _load_dictionary(base / "dsdur" / "dsdict-en.yaml")
        self._speaker = _load_speaker_embedding(_resolve_speaker(base, config, speaker))
        self._fallback_id = self._phoneme_ids.get(
            "AP", next(iter(self._phoneme_ids.values()))
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
                    f"voicebank has no English language id in {base}"
                ) from exc
        self._vocoder = _create_session(_find_vocoder(acoustic, vocoder_dir))

    def _phones(self, word: str) -> list[str]:
        """Return the phones for one word.

        Args:
            word: Lower-case dictionary key.

        Returns:
            The dictionary entry, or a heuristic fallback when unknown.
        """
        phones = self._dictionary.words.get(word)
        if phones is not None:
            return phones
        return _fallback_phones(word)

    def _token_id(self, phone: str) -> int:
        """Map one phone to a model token id.

        Args:
            phone: Bare ARPAbet symbol from the dictionary or heuristic.

        Returns:
            The model token id.  Unknown phones fall back to the dictionary's
            replacement table, the model's language prefix, then ``AP``.
        """
        token = self._phoneme_ids.get(phone)
        if token is not None:
            return token
        replacement = self._dictionary.replacements.get(phone)
        if replacement is not None and replacement in self._phoneme_ids:
            return self._phoneme_ids[replacement]
        return self._phoneme_ids.get(f"{self._symbol_prefix}{phone}", self._fallback_id)

    def synthesize(self, notes: Sequence[SungNote]) -> _FloatArray:
        """Render English notes to mono float samples at 44.1 kHz.

        Args:
            notes: Notes to sing, in performance order.

        Returns:
            Mono float64 samples covering ``max(onset + duration)`` seconds.
        """
        if not notes:
            return np.zeros(0, dtype=np.float64)
        groups = _group_words(notes)
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
        for group in groups:
            word = _word_text(notes, group)
            chunks = _distribute_phones(
                self._phones(word), [frames[index] for index in group]
            )
            for index, chunk in zip(group, chunks):
                phone_frames = _split_frames(frames[index], len(chunk))
                for phone, count in zip(chunk, phone_frames):
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
