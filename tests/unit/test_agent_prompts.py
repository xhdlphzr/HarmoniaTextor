# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for agent.prompts."""

from __future__ import annotations

from harmoniatextor.agent.prompts import (
    system_prompt,
)
from harmoniatextor.genres import PlainGenre
from harmoniatextor.techniques import build_default_registry


class TestPrompts:
    """Prompt generation."""

    def test_system_prompt(self) -> None:
        """The system prompt lists techniques and rules."""
        prompt = system_prompt(PlainGenre(), build_default_registry())
        assert "technique_imitation" in prompt
        assert "pf5th" in prompt
        assert "层次感" in prompt
        assert "没有任何声部" in prompt
        assert "add_part" in prompt
        assert "所有 musicxml 参数都是片段" in prompt
        assert "第一个 part" in prompt
        assert "修正:" in prompt
