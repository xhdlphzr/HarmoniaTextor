# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Step 1 planning tools: movements and their prompts."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, StructuredTool

from harmoniatextor.domain.params import (
    AddMovementParams,
    SetMovementPromptParams,
    SetTitleParams,
)
from harmoniatextor.service.service import CompositionService, ToolResult

__all__ = ["build_planning_tools"]


def _payload(result: ToolResult) -> str:
    """Render a planning result as JSON for the model.

    Args:
        result: The service result.

    Returns:
        A JSON string.
    """
    return json.dumps(
        {
            "ok": result.ok,
            "message": result.message,
            "movement_id": result.movement_id,
            "error_code": result.error_code,
        },
        ensure_ascii=False,
    )


def _movement_id(service: CompositionService, work_id: str, number: int) -> str | None:
    """Resolve a one-based movement number to its identifier.

    Args:
        service: Composition service.
        work_id: Active work.
        number: One-based movement number.

    Returns:
        The movement identifier, or ``None`` when out of range.
    """
    work = service.get_work(work_id)
    if 1 <= number <= len(work.movements):
        return work.movements[number - 1].id
    return None


def build_planning_tools(service: CompositionService, work_id: str) -> list[BaseTool]:
    """Build the Step 1 planning tool set.

    Args:
        service: Composition service to call.
        work_id: Active work.

    Returns:
        The set-title, add-movement and set-movement-prompt tools.
    """

    def set_title(title: str) -> str:
        """Set the title of the work."""
        return _payload(service.set_title(work_id, title))

    def add_movement(name: str = "") -> str:
        """Add a movement and receive its number."""
        return _payload(service.add_movement(work_id, name or None))

    def set_movement_prompt(movement: int, prompt: str) -> str:
        """Record the concrete creation requirement of one movement."""
        movement_id = _movement_id(service, work_id, movement)
        if movement_id is None:
            return _payload(
                ToolResult(
                    False, error_code="BAD_PARAM", message=f"乐章不存在:{movement}"
                )
            )
        return _payload(service.set_movement_prompt(work_id, movement_id, prompt))

    return [
        StructuredTool.from_function(
            func=set_title,
            name="set_title",
            description=(
                "Set a concise, fitting title for the work before planning the "
                "movements. If it is never called the genre name is used instead."
            ),
            args_schema=SetTitleParams,
        ),
        StructuredTool.from_function(
            func=add_movement,
            name="add_movement",
            description=(
                "Add a movement to the work and return its number. There is no "
                "template to pick; describe the structure in the movement prompt."
            ),
            args_schema=AddMovementParams,
        ),
        StructuredTool.from_function(
            func=set_movement_prompt,
            name="set_movement_prompt",
            description=(
                "Write the concrete creation requirement (prompt) for one movement. "
                "movement is the movement number (1-based). Every movement must get one."
            ),
            args_schema=SetMovementPromptParams,
        ),
    ]
