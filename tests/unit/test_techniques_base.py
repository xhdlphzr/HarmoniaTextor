# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for techniques.base."""

from __future__ import annotations

from harmoniatextor.score.io import new_score
from harmoniatextor.techniques.base import (
    TechniqueCategory,
    TechniqueContext,
    TechniqueError,
    TechniqueResult,
)


class TestTechniquesBase:
    """Context, error, result and category primitives."""

    def test_context_defaults(self) -> None:
        """The context defaults to a plain, four-part texture."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        context = TechniqueContext(score=score, themes={})
        assert context.genre == "plain"
        assert context.voice_profile == "four_part"

    def test_error(self) -> None:
        """A technique error carries a code and message."""
        error = TechniqueError("X", "boom")
        assert error.code == "X"
        assert error.message == "boom"
        assert str(error) == "boom"

    def test_result(self) -> None:
        """A result wraps the new score and warnings."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        result = TechniqueResult(score=score)
        assert result.score is score
        assert result.warnings == []

    def test_category(self) -> None:
        """Categories serialise to their names."""
        assert TechniqueCategory.MELODIC.value == "melodic"
