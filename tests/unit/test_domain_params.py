# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for domain.params."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from harmoniatextor.domain.params import (
    ExtendedHarmonyParams,
    FreeVoiceLeadingParams,
    MeasurePosition,
    MeasureRange,
    SetTimeSignatureParams,
    VoiceExchangeParams,
    parse_measure_position,
)


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

    def test_extended_harmony_defaults(self) -> None:
        """Extension degrees default to the seventh and ninth."""
        params = ExtendedHarmonyParams(
            voice="piano", key="C", measure_range=MeasureRange(start=1, end=1)
        )
        assert params.extensions == ["7", "9"]

    def test_free_voice_leading_requires_reason(self) -> None:
        """A blank exemption reason is rejected."""
        with pytest.raises(ValidationError):
            FreeVoiceLeadingParams(
                voice="soprano",
                measure_range=MeasureRange(start=1, end=1),
                reason="no",
            )

    def test_set_time_signature_valid(self) -> None:
        """A well-formed time signature is accepted and stripped."""
        assert SetTimeSignatureParams(time_signature=" 3/4 ").time_signature == "3/4"

    def test_set_time_signature_invalid(self) -> None:
        """Malformed or non-positive time signatures are rejected."""
        with pytest.raises(ValidationError):
            SetTimeSignatureParams(time_signature="bad")
        with pytest.raises(ValidationError):
            SetTimeSignatureParams(time_signature="0/4")
