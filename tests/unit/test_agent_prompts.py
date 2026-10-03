# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for agent.prompts."""

from __future__ import annotations

from harmoniatextor.agent.prompts import (
    architect_instruction,
    architect_system,
    system_prompt,
)
from harmoniatextor.genres import PlainGenre
from harmoniatextor.styles import BUILTIN_KITS
from harmoniatextor.techniques import build_default_registry


class TestPrompts:
    """Prompt generation."""

    def test_system_prompt(self) -> None:
        """The system prompt lists the style techniques and rules."""
        style = BUILTIN_KITS["baroque"]
        prompt = system_prompt(
            PlainGenre(), style, build_default_registry(), style.rules
        )
        assert "technique_imitation" in prompt
        assert "当前风格:巴洛克" in prompt
        assert "pf5th" in prompt
        assert "层次感" in prompt
        assert "没有任何声部" in prompt
        assert "add_part" in prompt
        assert "所有 musicxml 参数都是片段" in prompt
        assert "第一个 part" in prompt
        assert "修正:" in prompt

    def test_system_prompt_is_style_scoped(self) -> None:
        """Only the style's rules and techniques appear."""
        style = BUILTIN_KITS["impressionist"]
        prompt = system_prompt(
            PlainGenre(), style, build_default_registry(), style.rules
        )
        assert "当前风格:印象派" in prompt
        assert "pf5th" not in prompt
        assert "voices" in prompt

    def test_architect_prompts(self) -> None:
        """The architect prompts carry the style."""
        style = BUILTIN_KITS["classical"]
        assert "古典主义" in architect_system(style)
        assert "古典主义" in architect_instruction(style)
