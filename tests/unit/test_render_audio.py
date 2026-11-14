# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for render.audio."""

from __future__ import annotations

import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from music21 import converter, metadata, stream
from numpy.typing import NDArray

from harmoniatextor.domain.models import Syllable, ThemeNote
from harmoniatextor.render.audio import (
    FeatureUnavailableError,
    _apply_section_sizes,
    _fit_vocal_level,
    _instrumental_score,
    _is_silent,
    _vocal_stems,
    _write_wav,
    synthesize_audio,
)
from harmoniatextor.render.features import Features
from harmoniatextor.render.vocal.singer import SungNote
from harmoniatextor.score.io import new_score
from harmoniatextor.score.lyrics import attach_lyrics
from harmoniatextor.score.streamops import ScoreEditor

_EXPECTED_CALLS = 2


def _read_wav(path: Path) -> NDArray[np.float64]:
    """Read a 16-bit mono WAV file back into floats.

    Args:
        path: WAV file to read.

    Returns:
        The samples in ``[-1, 1]``.
    """
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def _keep_intermediates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[list[list[str]], Path]:
    """Replace the audio subprocesses and keep the scratch directory.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: The pytest temporary path fixture.

    Returns:
        The recorded subprocess calls and the scratch directory.
    """
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> None:
        calls.append(args)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
    monkeypatch.setattr(
        "harmoniatextor.render.audio.tempfile.mkdtemp", lambda prefix="": str(scratch)
    )
    monkeypatch.setattr(
        "harmoniatextor.render.audio.shutil.rmtree", lambda *args, **kwargs: None
    )
    return calls, scratch


class TestAudio:
    """Audio synthesis pipeline."""

    def test_unavailable(self, tmp_path: Path) -> None:
        """Missing binaries raise a feature error.

        Args:
            tmp_path: The pytest temporary path fixture.
        """
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        with pytest.raises(FeatureUnavailableError):
            synthesize_audio(score, tmp_path / "out.m4a", Features())

    def test_synthesize_m4a(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pipeline invokes FluidSynth then ffmpeg for M4A.

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        ScoreEditor(score).write_line("soprano", 1, [ThemeNote("C5", 1.0)])
        features = Features(
            ffmpeg=tmp_path / "ffmpeg.exe",
            fluidsynth=tmp_path / "fluidsynth.exe",
            soundfont=tmp_path / "s.sf2",
        )
        out = synthesize_audio(score, tmp_path / "out.m4a", features, "m4a")
        assert out == tmp_path / "out.m4a"
        assert len(calls) == _EXPECTED_CALLS
        assert calls[0][calls[0].index("-g") + 1] == "0.85"
        assert calls[1][calls[1].index("-af") + 1].startswith("loudnorm")
        assert calls[1][-1] == str(out)
        assert "aac" in calls[1]
        assert not (tmp_path / "out.mid").exists()
        assert not (tmp_path / "out.wav").exists()

    def test_synthesize_mp3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pipeline encodes MP3 with libmp3lame.

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        ScoreEditor(score).write_line("soprano", 1, [ThemeNote("C5", 1.0)])
        features = Features(
            ffmpeg=tmp_path / "ffmpeg.exe",
            fluidsynth=tmp_path / "fluidsynth.exe",
            soundfont=tmp_path / "s.sf2",
        )
        out = synthesize_audio(score, tmp_path / "out.mp3", features, "mp3")
        assert out == tmp_path / "out.mp3"
        assert "libmp3lame" in calls[1]
        assert not (tmp_path / "out.mid").exists()
        assert not (tmp_path / "out.wav").exists()

    def test_synthesize_serialises_same_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two exports of the same file both succeed (per-output lock reuse).

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        ScoreEditor(score).write_line("soprano", 1, [ThemeNote("C5", 1.0)])
        features = Features(
            ffmpeg=tmp_path / "ffmpeg.exe",
            fluidsynth=tmp_path / "fluidsynth.exe",
            soundfont=tmp_path / "s.sf2",
        )
        out = tmp_path / "same.m4a"
        synthesize_audio(score, out, features, "m4a")
        synthesize_audio(score, out, features, "m4a")
        assert len(calls) == _EXPECTED_CALLS * 2

    def test_section_sizes_noop(self) -> None:
        """No counts, an empty mapping and unit counts keep the score as is."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["flute"])
        assert _apply_section_sizes(score, None) is score
        assert _apply_section_sizes(score, {}) is score
        assert _apply_section_sizes(score, {"flute": 1}) is score

    def test_section_sizes_expand(self) -> None:
        """A section is expanded into quietened per-player copies."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["flute", "violin"]
        )
        ScoreEditor(score).write_line("flute", 1, [ThemeNote("C5", 1.0)])
        score.metadata = metadata.Metadata()
        score.metadata.title = "Section"
        for item in score.recurse().notes:
            item.volume.velocity = 90
        rendered = _apply_section_sizes(score, {"flute": 3})
        assert rendered.metadata is not None
        assert rendered.metadata.title == "Section"
        names = sorted(str(part.id or part.partName) for part in rendered.parts)
        assert names == ["flute~1", "flute~2", "flute~3", "violin"]
        copies = [part for part in rendered.parts if str(part.id).startswith("flute~")]
        for part in copies:
            velocities = [item.volume.velocity for item in part.recurse().notes]
            assert velocities == [52]
        assert len(score.parts) == 2


class _FakeSinger:
    """A vocal backend returning a fixed sample count."""

    def __init__(self, size: int = 100) -> None:
        """Store the sample count.

        Args:
            size: Number of samples to return.
        """
        self.size = size

    def synthesize(self, _notes: Sequence[SungNote]) -> NDArray[np.float64]:
        """Return fixed samples.

        Args:
            _notes: Notes to sing (ignored).

        Returns:
            The samples.
        """
        return np.ones(self.size, dtype=np.float64)


class _SilentSinger:
    """A vocal backend that returns a full buffer of digital silence."""

    def synthesize(self, notes: Sequence[SungNote]) -> NDArray[np.float64]:
        """Return zeros of the expected duration.

        Args:
            notes: Notes to sing.

        Returns:
            Silent samples.
        """
        length = int(
            max((note.onset + note.duration for note in notes), default=0.0) * 44100
        )
        return np.zeros(max(1, length), dtype=np.float64)


def _vocal_score(*, with_violin: bool = True) -> stream.Score:
    """Build a score with a sung soprano and optionally a violin.

    Args:
        with_violin: Whether to add an instrumental part.

    Returns:
        The score.
    """
    voices = ["soprano", "violin"] if with_violin else ["soprano"]
    score = new_score(key="C", time_signature="4/4", tempo_bpm=120, voices=voices)
    editor = ScoreEditor(score)
    editor.write_line("soprano", 1, [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
    if with_violin:
        editor.write_line("violin", 1, [ThemeNote("G4", 1.0)])
    attach_lyrics(
        score.parts[0],
        1,
        [Syllable("love", "lʌv"), Syllable("you", "juː")],
    )
    return score


def _unlyriced_vocal_score() -> stream.Score:
    """Build a marked vocal soprano score whose notes carry no lyrics.

    Returns:
        The score.
    """
    score = new_score(
        key="C", time_signature="4/4", tempo_bpm=120, voices=["soprano", "violin"]
    )
    editor = ScoreEditor(score)
    editor.write_line("soprano", 1, [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
    editor.write_line("violin", 1, [ThemeNote("G4", 1.0)])
    return score


def _vocal_features(tmp_path: Path) -> Features:
    """Return available audio features rooted in a temp directory.

    Args:
        tmp_path: The pytest temporary path fixture.

    Returns:
        The features.
    """
    return Features(
        ffmpeg=tmp_path / "ffmpeg.exe",
        fluidsynth=tmp_path / "fluidsynth.exe",
        soundfont=tmp_path / "s.sf2",
    )


class TestStemHelpers:
    """WAV writing and vocal-stem preparation."""

    def test_write_wav_roundtrip(self, tmp_path: Path) -> None:
        """In-range samples survive the 16-bit conversion.

        Args:
            tmp_path: The pytest temporary path fixture.
        """
        samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        path = tmp_path / "tone.wav"
        _write_wav(path, samples)
        data = _read_wav(path)
        assert data.size == samples.size
        assert np.allclose(data, samples, atol=1e-3)

    def test_write_wav_sanitises(self, tmp_path: Path) -> None:
        """NaN, infinities and out-of-range values become finite samples.

        Args:
            tmp_path: The pytest temporary path fixture.
        """
        samples = np.array([np.nan, np.inf, -np.inf, 2.0, -2.0])
        path = tmp_path / "noisy.wav"
        _write_wav(path, samples)
        data = _read_wav(path)
        assert np.all(np.isfinite(data))
        assert data[0] == 0.0
        assert data[1] > 0.99
        assert data[2] < -0.99
        assert data[3] > 0.99
        assert data[4] < -0.99

    def test_is_silent(self) -> None:
        """Empty, flat, non-finite and audible buffers are classified right."""
        assert _is_silent(np.zeros(0, dtype=np.float64))
        assert _is_silent(np.zeros(10, dtype=np.float64))
        assert _is_silent(np.array([np.nan, np.inf, -np.inf], dtype=np.float64))
        assert not _is_silent(np.ones(10, dtype=np.float64))

    def test_fit_vocal_level(self) -> None:
        """Quiet and loud stems are normalised, with a gain cap and peak ceiling."""
        target = 0.2
        loud = _fit_vocal_level(np.ones(1000, dtype=np.float64))
        assert abs(float(np.sqrt(np.mean(np.square(loud)))) - target) < 1e-6
        quiet = _fit_vocal_level(np.full(1000, 0.01, dtype=np.float64))
        assert abs(float(np.sqrt(np.mean(np.square(quiet)))) - 0.08) < 1e-6
        assert _fit_vocal_level(np.zeros(0, dtype=np.float64)).size == 0
        assert (
            float(np.max(np.abs(_fit_vocal_level(np.zeros(100, dtype=np.float64)))))
            == 0.0
        )
        spiky = np.full(1000, 0.01, dtype=np.float64)
        spiky[0] = 0.9
        assert float(np.max(np.abs(_fit_vocal_level(spiky)))) <= 0.98 + 1e-9

    def test_instrumental_score(self) -> None:
        """Only voices that really sing are removed from the instrumental."""
        score = _vocal_score()
        stripped = _instrumental_score(score, ["soprano"])
        assert [str(part.id) for part in stripped.parts] == ["violin"]
        assert len(score.parts) == 2
        assert _instrumental_score(score, []) is score
        assert _instrumental_score(score, ["flute"]) is score

    def test_vocal_stems_skip_silence(self, tmp_path: Path) -> None:
        """Empty and all-zero backend output produces no stem.

        Args:
            tmp_path: The pytest temporary path fixture.
        """
        notes = [SungNote(60, 0.0, 0.1, "a")]
        assert _vocal_stems({"soprano": notes}, _FakeSinger(size=0), tmp_path) == {}
        assert _vocal_stems({"soprano": notes}, _SilentSinger(), tmp_path) == {}

    def test_vocal_stems_level(self, tmp_path: Path) -> None:
        """The written stem is attenuated to the mix target.

        Args:
            tmp_path: The pytest temporary path fixture.
        """
        notes = [SungNote(60, 0.0, 0.1, "a")]
        stems = _vocal_stems({"soprano": notes}, _FakeSinger(size=4410), tmp_path)
        data = _read_wav(stems["soprano"])
        assert abs(float(np.sqrt(np.mean(np.square(data)))) - 0.2) < 1e-3


class TestVocalAudio:
    """Vocal stems mixed with the instrumental render."""

    def test_mix_with_instruments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Vocals are synthesised and mixed with the instrumental stem."""
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        out = tmp_path / "out.m4a"
        synthesize_audio(
            _vocal_score(),
            out,
            _vocal_features(tmp_path),
            "m4a",
            vocal_voices={"soprano": "en"},
            vocal_backend=_FakeSinger(),
        )
        assert len(calls) == _EXPECTED_CALLS
        assert any("amix" in part for part in calls[1])
        assert any("voice-0.wav" in part for part in calls[1])
        assert calls[1][-1] == str(out)

    def test_vocals_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A vocals-only score skips FluidSynth and encodes the stem directly."""
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        out = tmp_path / "out.mp3"
        synthesize_audio(
            _vocal_score(with_violin=False),
            out,
            _vocal_features(tmp_path),
            "mp3",
            vocal_voices={"soprano": "en"},
            vocal_backend=_FakeSinger(),
        )
        assert len(calls) == 1
        assert "-af" in calls[0]
        assert calls[0][-1] == str(out)

    def test_marked_vocal_without_lyrics_is_sung(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A marked voice is sung even after its lyrics were lost.

        The built-in singer must produce a stem; otherwise the voice would be
        removed from the instrumental render and vanish from the mix.
        """
        calls, scratch = _keep_intermediates(monkeypatch, tmp_path)
        synthesize_audio(
            _unlyriced_vocal_score(),
            tmp_path / "out.m4a",
            _vocal_features(tmp_path),
            "m4a",
            vocal_voices={"soprano": "en"},
        )
        assert any("voice-0.wav" in part for part in calls[1])
        assert any("amix" in part for part in calls[1])
        midi = converter.parse(str(scratch / "score.mid"))
        assert isinstance(midi, stream.Score)
        assert len(midi.parts) == 1

    def test_silent_backend_keeps_instrumental(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A backend that renders silence leaves the voice to FluidSynth.

        Dropping the part because it is marked vocal would silence it
        completely; keeping it in the instrumental render is the safe fallback.
        """
        calls, scratch = _keep_intermediates(monkeypatch, tmp_path)
        synthesize_audio(
            _vocal_score(),
            tmp_path / "out.m4a",
            _vocal_features(tmp_path),
            "m4a",
            vocal_voices={"soprano": "en"},
            vocal_backend=_SilentSinger(),
        )
        assert len(calls) == _EXPECTED_CALLS
        assert not any("amix" in part for part in calls[1])
        midi = converter.parse(str(scratch / "score.mid"))
        assert isinstance(midi, stream.Score)
        assert len(midi.parts) == 2

    def test_empty_stem_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A backend that returns no samples leaves the instrumental only."""
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr("harmoniatextor.render.audio.subprocess.run", fake_run)
        out = tmp_path / "out.m4a"
        synthesize_audio(
            _vocal_score(),
            out,
            _vocal_features(tmp_path),
            "m4a",
            vocal_voices={"soprano": "en"},
            vocal_backend=_FakeSinger(size=0),
        )
        assert len(calls) == _EXPECTED_CALLS
        assert not any("amix" in part for part in calls[1])

    def test_no_inputs(self, tmp_path: Path) -> None:
        """A score with nothing to render reports a feature error."""
        with pytest.raises(FeatureUnavailableError):
            synthesize_audio(
                stream.Score(),
                tmp_path / "out.m4a",
                _vocal_features(tmp_path),
                "m4a",
                vocal_voices={"soprano": "en"},
                vocal_backend=_FakeSinger(),
            )
