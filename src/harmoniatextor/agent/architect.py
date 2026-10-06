# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Step 1 architect: plan movements and their per-movement prompts.

The architect drives the planning model through the planning tools until every
movement carries a concrete prompt, then records a readable plan for the work
page and the generation phase.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from harmoniatextor.agent.loop import run_tool_calls
from harmoniatextor.agent.planning_tools import build_planning_tools
from harmoniatextor.agent.prompts import architect_instruction, architect_system
from harmoniatextor.service.service import CompositionService

__all__ = ["Architect", "PlanResult", "plan_tree", "render_plan"]

ProgressCallback = Callable[[dict[str, Any]], None]

_MISSING_MESSAGE = (
    "These movements still have no requirement; fill each one in with "
    "set_movement_prompt(movement, prompt):\n{listing}"
)
_NO_MOVEMENTS_MESSAGE = (
    "There are no movements yet; first plan movements with add_movement and write "
    "a prompt for each."
)


@dataclass(slots=True)
class PlanResult:
    """Outcome of a Step 1 planning session.

    Attributes:
        completed: Whether every movement received a prompt.
        plan: The rendered plan text.
        messages: The full planner conversation.
    """

    completed: bool
    plan: str
    messages: list[BaseMessage] = field(default_factory=list)
    tree: list[dict[str, Any]] = field(default_factory=list)


def plan_tree(service: CompositionService, work_id: str) -> list[dict[str, Any]]:
    """Return the plan as a movement/prompt tree.

    Args:
        service: Composition service.
        work_id: Active work.

    Returns:
        One dictionary per movement.
    """
    work = service.get_work(work_id)
    return [
        {
            "index": index,
            "id": movement.id,
            "name": movement.name,
            "prompt": movement.prompt,
        }
        for index, movement in enumerate(work.movements, start=1)
    ]


def render_plan(service: CompositionService, work_id: str) -> str:
    """Render a work's movements and prompts as readable text.

    Args:
        service: Composition service.
        work_id: Active work.

    Returns:
        A multi-line plan description.
    """
    work = service.get_work(work_id)
    lines: list[str] = []
    for index, movement in enumerate(work.movements, start=1):
        lines.append(f"Movement {index}: {movement.name}")
        lines.append(f"  Requirement: {movement.prompt}")
    return "\n".join(lines)


def _missing_listing(service: CompositionService, work_id: str) -> list[str]:
    """Render the movements still missing a prompt.

    Args:
        service: Composition service.
        work_id: Active work.

    Returns:
        One bullet per movement without a prompt.
    """
    work = service.get_work(work_id)
    numbers = {
        movement.id: index for index, movement in enumerate(work.movements, start=1)
    }
    return [
        f"- Movement {numbers.get(movement.id, '?')} ({movement.name})"
        for movement in service.missing_movement_prompts(work_id)
    ]


class Architect:
    """Drive Step 1 until every movement has a concrete prompt.

    Attributes:
        service: Composition service.
        chat_model: Chat model driving the planner.
        max_steps: Maximum number of planner turns.
        max_rounds: Maximum number of "still missing" nudges.
    """

    def __init__(
        self,
        service: CompositionService,
        chat_model: BaseChatModel,
        *,
        max_steps: int = 60,
        max_rounds: int = 8,
    ) -> None:
        """Initialise the architect.

        Args:
            service: Composition service.
            chat_model: Chat model.
            max_steps: Maximum number of planner turns.
            max_rounds: Maximum number of coverage nudges.
        """
        self.service = service
        self.chat_model = chat_model
        self.max_steps = max_steps
        self.max_rounds = max_rounds

    def plan(
        self,
        work_id: str,
        goal: str,
        *,
        on_event: ProgressCallback | None = None,
    ) -> PlanResult:
        """Run the Step 1 planning session.

        Args:
            work_id: Active work.
            goal: The composition goal.
            on_event: Optional progress callback.

        Returns:
            The planning outcome.
        """
        tools = build_planning_tools(self.service, work_id)
        mapping = {tool.name: tool for tool in tools}
        bound = self.chat_model.bind_tools(tools)
        style = self.service.style_for(self.service.get_work(work_id))
        messages: list[BaseMessage] = [
            SystemMessage(architect_system(style)),
            HumanMessage(f"{goal}\n\n{architect_instruction(style)}"),
        ]
        _emit(on_event, {"kind": "plan_start"})
        steps = 0
        rounds = 0
        completed = False
        while steps < self.max_steps:
            response = bound.invoke(messages)
            messages.append(response)
            steps += 1
            if run_tool_calls(messages, response, mapping, steps, on_event):
                continue
            movements = self.service.get_work(work_id).movements
            if not movements:
                messages.append(HumanMessage(_NO_MOVEMENTS_MESSAGE))
                continue
            listing = _missing_listing(self.service, work_id)
            if listing:
                rounds += 1
                if rounds > self.max_rounds:
                    break
                messages.append(
                    HumanMessage(_MISSING_MESSAGE.format(listing="\n".join(listing)))
                )
                continue
            completed = True
            break
        plan = render_plan(self.service, work_id)
        tree = plan_tree(self.service, work_id)
        work = self.service.get_work(work_id)
        movement_id = work.movements[0].id if work.movements else ""
        self.service.record_plan(work_id, movement_id, plan)
        _emit(on_event, {"kind": "plan", "text": plan, "tree": tree})
        return PlanResult(completed, plan, messages, tree)


def _emit(on_event: ProgressCallback | None, event: dict[str, Any]) -> None:
    """Forward an event to the progress callback when one is set.

    Args:
        on_event: Progress callback or ``None``.
        event: Event payload.
    """
    if on_event is not None:
        on_event(event)
