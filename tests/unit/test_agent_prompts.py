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
        assert "Current style: Baroque" in prompt
        assert "pf5th" in prompt
        assert "layered" in prompt
        assert "no voices at all" in prompt
        assert "add_part" in prompt
        assert "every musicxml parameter is a fragment" in prompt
        assert "first part" in prompt
        assert "braced" in prompt
        assert "bracketed as a section" in prompt
        assert "Fix:" in prompt

    def test_system_prompt_teaches_annotate(self) -> None:
        """The system prompt explains how to write performable marks."""
        style = BUILTIN_KITS["baroque"]
        prompt = system_prompt(
            PlainGenre(), style, build_default_registry(), style.rules
        )
        assert "How to write expression and marks" in prompt
        assert "annotate" in prompt
        assert "dolce" in prompt
        assert "at least 2 notes" in prompt
        assert "staccato" in prompt
        assert "pedal" in prompt

    def test_system_prompt_is_style_scoped(self) -> None:
        """Only the style's rules and techniques appear."""
        style = BUILTIN_KITS["impressionist"]
        prompt = system_prompt(
            PlainGenre(), style, build_default_registry(), style.rules
        )
        assert "Current style: Impressionist" in prompt
        assert "pf5th" not in prompt
        assert "voices" in prompt

    def test_architect_prompts(self) -> None:
        """The architect prompts carry the style."""
        style = BUILTIN_KITS["classical"]
        assert "Current style: Classical" in architect_system(style)
        assert "Current style: Classical" in architect_instruction(style)
