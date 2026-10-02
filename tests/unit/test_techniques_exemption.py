# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for techniques.exemption."""

from __future__ import annotations

from music21 import stream

from harmoniatextor.techniques import TechniqueContext, build_default_registry
from harmoniatextor.techniques.exemption import FREE_VOICE_LEADING_EXEMPT


class TestFreeVoiceLeading:
    """The free-voice-leading rule exemption."""

    def test_exempts(self) -> None:
        """The exemption waives the expected voice-leading rules."""
        technique = build_default_registry().get("free_voice_leading")
        assert technique.exempts == FREE_VOICE_LEADING_EXEMPT
        assert "pf5th" not in technique.exempts

    def test_ordinary_technique_has_no_exemption(self) -> None:
        """Ordinary techniques waive nothing."""
        assert build_default_registry().get("imitation").exempts == frozenset()

    def test_apply(self, score4: stream.Score) -> None:
        """Applying the exemption returns the unchanged score and a note."""
        technique = build_default_registry().get("free_voice_leading")
        context = TechniqueContext(score=score4, themes={})
        params = technique.params_model(
            voice="soprano",
            measure_range={"start": 1, "end": 2},
            reason="表情需要自由进行",
        )
        result = technique.apply(context, params)
        assert result.score is score4
        assert result.warnings
