# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the domain layer."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from harmoniatextor.domain.enums import Severity, VoiceSlot
from harmoniatextor.domain.interval import parse_interval
from harmoniatextor.domain.key import parse_key
from harmoniatextor.domain.models import (
    CheckReport,
    CheckViolation,
    ThemeNote,
    theme_fingerprint,
)
from harmoniatextor.domain.params import (
    MeasurePosition,
    MeasureRange,
    VoiceExchangeParams,
    parse_measure_position,
)

_FIFTH_SEMITONES = 7
_FIFTH_STEPS = 4
_FOURTH_SEMITONES = -5
_OCTAVE_SEMITONES = 12


class TestInterval:
    """Interval parsing."""

    def test_perfect_fifth(self) -> None:
        """A perfect fifth is seven semitones and four steps."""
        parsed = parse_interval(5)
        assert parsed.semitones == _FIFTH_SEMITONES
        assert parsed.steps == _FIFTH_STEPS
        assert parsed.name == "P5"

    def test_negative_fourth(self) -> None:
        """A descending fourth is negative."""
        parsed = parse_interval(-4)
        assert parsed.semitones == _FOURTH_SEMITONES
        assert parsed.name == "-P4"

    def test_major_third(self) -> None:
        """A third uses major quality."""
        assert parse_interval(3).name == "M3"

    def test_octave_semitones(self) -> None:
        """Values above seven are treated as semitones."""
        parsed = parse_interval(12)
        assert parsed.semitones == _OCTAVE_SEMITONES
        assert parsed.name == "P8"

    def test_zero_rejected(self) -> None:
        """A zero interval is invalid."""
        with pytest.raises(ValueError, match="non-zero"):
            parse_interval(0)


class TestKey:
    """Key parsing."""

    def test_major(self) -> None:
        """Upper-case keys are major."""
        key = parse_key("C")
        assert key.is_major
        assert key.music21_name == "C"

    def test_minor(self) -> None:
        """Lower-case keys are minor."""
        key = parse_key("a")
        assert not key.is_major
        assert key.music21_name == "a"

    def test_flat(self) -> None:
        """Flats become dashes for music21."""
        assert parse_key("Eb").music21_name == "E-"

    def test_sharp(self) -> None:
        """Sharps are preserved."""
        assert parse_key("f#").music21_name == "f#"

    def test_invalid(self) -> None:
        """Invalid keys are rejected."""
        with pytest.raises(ValueError, match="invalid key"):
            parse_key("H")


class TestModels:
    """Entity behaviour."""

    def test_theme_fingerprint_stable(self) -> None:
        """Fingerprints are deterministic."""
        notes = [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)]
        assert theme_fingerprint(notes) == theme_fingerprint(notes)

    def test_report_ok(self) -> None:
        """A report without errors is ok."""
        report = CheckReport()
        assert report.ok
        assert report.to_dict() == {"ok": True, "violations": []}

    def test_report_with_error(self) -> None:
        """An error makes the report fail and exposes warnings separately."""
        error = CheckViolation("r", Severity.ERROR, 1, "s", None, "k", "m", "s")
        warning = CheckViolation("r2", Severity.WARNING, 2, "a", None, "k", "m", "s")
        report = CheckReport(violations=[error, warning])
        assert not report.ok
        assert report.errors == [error]
        assert report.warnings == [warning]
        assert report.to_dict()["ok"] is False

    def test_voice_slot_values(self) -> None:
        """Voice slots have the expected canonical values."""
        assert VoiceSlot.SOPRANO.value == "soprano"


class TestParams:
    """Parameter validation."""

    def test_measure_range_order(self) -> None:
        """A reversed measure range is rejected."""
        with pytest.raises(ValidationError):
            MeasureRange(start=4, end=2)

    def test_parse_measure_position(self) -> None:
        """Positions parse into measure and beat."""
        assert parse_measure_position("12+1.5") == (12, 1.5)
        assert parse_measure_position("3") == (3, 1.0)

    def test_parse_measure_position_invalid(self) -> None:
        """Malformed positions are rejected."""
        with pytest.raises(ValueError, match="invalid measure position"):
            parse_measure_position("abc")

    def test_parse_measure_position_bad_beat(self) -> None:
        """Non-positive beats are rejected."""
        with pytest.raises(ValueError, match="invalid measure position"):
            parse_measure_position("3+0")

    def test_measure_position_model(self) -> None:
        """The model validates the position string."""
        assert MeasurePosition(measure_position="4+2").measure_position == "4+2"
        with pytest.raises(ValidationError):
            MeasurePosition(measure_position="nope")

    def test_voice_exchange_requires_targets(self) -> None:
        """Voice exchange needs themes or voices."""
        with pytest.raises(ValidationError):
            VoiceExchangeParams(measure_range=MeasureRange(start=1, end=2))

    def test_voice_exchange_with_voices(self) -> None:
        """Explicit voices satisfy the requirement."""
        params = VoiceExchangeParams(
            voice_1="soprano",
            voice_2="bass",
            measure_range=MeasureRange(start=1, end=2),
        )
        assert params.voice_1 == "soprano"
