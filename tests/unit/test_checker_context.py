# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for checker.context."""

from __future__ import annotations

from harmoniatextor.checker.context import CheckerContext, StructuralExpectation

_SPACING_LIMIT = 16
_EXPECTED_MEASURE = 3


class TestCheckerContext:
    """Context and structural expectations."""

    def test_defaults(self) -> None:
        """Defaults match the strict single-movement profile."""
        context = CheckerContext()
        assert context.genre == "plain"
        assert context.tonic == "C"
        assert context.enforce_voice_count is True
        assert context.spacing_limit == _SPACING_LIMIT
        assert context.expectations == []

    def test_expectation(self) -> None:
        """Structural expectations keep their fields."""
        expectation = StructuralExpectation(
            measure=_EXPECTED_MEASURE, key="G", cadence=True, label="end"
        )
        assert expectation.measure == _EXPECTED_MEASURE
        assert expectation.key == "G"
        assert expectation.cadence is True
        assert expectation.label == "end"
