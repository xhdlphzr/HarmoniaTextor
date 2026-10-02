# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""The built-in LangChain composer loop.

The loop drives an OpenAI-compatible chat model with bound tools.  Every tool
call goes through the composition service, which runs the symbolic checker, so
violations are returned to the model as tool output.

When the symbolic layer passes, an optional independent reviewer AI inspects
the score in a fresh session.  If it rejects the score, its suggestions are
appended to the *same* creator session and the loop continues.  Long sessions
are compressed automatically at 90% of the context window.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool

from harmoniatextor.agent.compression import (
    DEFAULT_CONTEXT_WINDOW,
    compress_messages,
    ensure_tool_responses,
    message_text,
)
from harmoniatextor.agent.prompts import STEP1_INSTRUCTION, STEP2_INSTRUCTION, system_prompt
from harmoniatextor.agent.reviewer import ReviewerAI, ReviewResult
from harmoniatextor.agent.tools import build_tools
from harmoniatextor.checker.engine import format_feedback
from harmoniatextor.service.service import CompositionService
from harmoniatextor.techniques.registry import TechniqueRegistry

__all__ = [
    "AgentLoop",
    "AgentRunResult",
    "ProgressCallback",
    "execute_call",
    "run_tool_calls",
]

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of an agent run.

    Attributes:
        completed: Whether the run finished successfully.
        steps: Number of model turns executed.
        final_text: The final assistant text.
        messages: The full message history.
        review_passed: Reviewer verdict, or ``None`` when no reviewer ran.
        review_suggestions: The reviewer's latest suggestions.
        plan: The Step 1 creation plan, when one was produced.
    """

    completed: bool
    steps: int
    final_text: str
    messages: list[BaseMessage] = field(default_factory=list)
    review_passed: bool | None = None
    review_suggestions: str = ""
    plan: str = ""


class AgentLoop:
    """A bounded tool-calling loop over a composition service.

    Attributes:
        service: Composition service.
        chat_model: Chat model driving the loop.
        reviewer: Optional independent reviewer AI.
        max_steps: Maximum number of model turns.
        max_reviews: Maximum number of reviewer rejections before giving up.
        context_window: Context window used for compression.
        techniques: Technique registry used to build tools.
    """

    def __init__(  # noqa: PLR0913
        self,
        service: CompositionService,
        chat_model: BaseChatModel,
        *,
        reviewer: ReviewerAI | None = None,
        max_steps: int | None = None,
        max_reviews: int = 5,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        techniques: TechniqueRegistry | None = None,
    ) -> None:
        """Initialise the loop.

        Args:
            service: Composition service.
            chat_model: Chat model.
            reviewer: Optional independent reviewer AI.
            max_steps: Maximum number of model turns, or ``None`` for no limit.
            max_reviews: Maximum number of reviewer rejections.
            context_window: Context window used for compression.
            techniques: Technique registry; defaults to the service registry.
        """
        self.service = service
        self.chat_model = chat_model
        self.reviewer = reviewer
        self.max_steps = max_steps
        self.max_reviews = max_reviews
        self.context_window = context_window
        self.techniques = techniques

    def run(  # noqa: PLR0913
        self,
        work_id: str,
        movement_id: str,
        goal: str,
        *,
        feedback: str | None = None,
        on_event: ProgressCallback | None = None,
        auto_continue: bool = True,
        plan: bool = False,
    ) -> AgentRunResult:
        """Run the composer loop.

        The loop is fully automatic: it keeps calling tools, feeds symbolic
        violations back to the model, and — when the model stops while the
        checker still reports errors — continues automatically with the checker
        feedback until the score passes.  By default there is no step limit.
        Once the symbolic layer passes, the reviewer AI is consulted; a
        rejection is appended to the creator session and the loop continues.

        When ``plan`` is true, Step 1 runs first in the same session: the model
        plans the instruments and their emotional arcs before Step 2 starts
        composing.  A plan stored by an earlier run is injected otherwise.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            goal: The composition goal or instruction.
            feedback: Optional human audition feedback to incorporate.
            on_event: Optional progress callback receiving event dictionaries.
            auto_continue: Whether to re-prompt on a failing check.
            plan: Whether to run the Step 1 planning phase first.

        Returns:
            The run result.
        """
        work = self.service.get_work(work_id)
        genre = self.service.genres.get(work.genre)
        style = self.service.style_for(work)
        rules = self.service.effective_rules(work)
        techniques = (
            self.techniques if self.techniques is not None else (self.service.techniques_for(work))
        )
        tools = build_tools(self.service, work_id, movement_id, techniques)
        mapping = {tool.name: tool for tool in tools}
        bound = self.chat_model.bind_tools(tools)
        prompt = system_prompt(genre, style, techniques, rules)
        messages: list[BaseMessage] = [SystemMessage(content=prompt)]
        instruction = goal if not feedback else f"{goal}\n\n人工品鉴意见:{feedback}"
        messages.append(HumanMessage(content=instruction))
        _emit(on_event, {"kind": "start", "goal": goal, "tools": sorted(mapping)})
        plan_text = ""
        if plan:
            plan_text = self._plan(messages, work_id, movement_id, on_event)
            messages.append(HumanMessage(content=STEP2_INSTRUCTION))
        else:
            stored_plan = self.service.latest_plan(work_id)
            if stored_plan:
                messages.append(SystemMessage(content="已有的创作规划:\n" + stored_plan))
        steps = 0
        reviews = 0
        while self.max_steps is None or steps < self.max_steps:
            self._compress(messages, work_id, movement_id, on_event)
            ensure_tool_responses(messages)
            steps += 1
            _emit(on_event, {"kind": "thinking", "step": steps})
            response: AIMessage = bound.invoke(messages)
            messages.append(response)
            if run_tool_calls(messages, response, mapping, steps, on_event):
                continue
            text = message_text(response)
            report = self.service.check(work_id, movement_id)
            _emit(
                on_event,
                {
                    "kind": "assistant",
                    "step": steps,
                    "text": text,
                    "ok": report.ok,
                    "violations": report.to_dict()["violations"],
                },
            )
            if not report.ok:
                if not auto_continue:
                    return AgentRunResult(False, steps, text, messages, plan=plan_text)
                feedback = "符号层仍未通过,请继续修正:\n" + format_feedback(report)
                _emit(
                    on_event,
                    {"kind": "feedback", "layer": "symbolic", "step": steps, "text": feedback},
                )
                messages.append(HumanMessage(content=feedback))
                continue
            if self.reviewer is None:
                return AgentRunResult(True, steps, text, messages, plan=plan_text)
            review = self._review(work_id, movement_id, goal, genre.display_name, on_event)
            if review.passed:
                return AgentRunResult(
                    True,
                    steps,
                    text,
                    messages,
                    review_passed=True,
                    review_suggestions="",
                    plan=plan_text,
                )
            reviews += 1
            if reviews >= self.max_reviews:
                return AgentRunResult(
                    False,
                    steps,
                    text,
                    messages,
                    review_passed=False,
                    review_suggestions=review.suggestions,
                    plan=plan_text,
                )
            feedback = "检查AI未通过,请根据以下意见继续修改:\n" + review.suggestions
            _emit(
                on_event,
                {"kind": "feedback", "layer": "reviewer", "step": steps, "text": feedback},
            )
            messages.append(HumanMessage(content=feedback))
        return AgentRunResult(False, steps, "", messages, plan=plan_text)

    def _plan(
        self,
        messages: list[BaseMessage],
        work_id: str,
        movement_id: str,
        on_event: ProgressCallback | None,
    ) -> str:
        """Run Step 1: plan instruments and per-voice emotional arcs.

        The plan is produced in the same session as Step 2, so the composer
        continues from it, and it is recorded for the work page.

        Args:
            messages: Conversation messages, modified in place.
            work_id: Active work.
            movement_id: Active movement.
            on_event: Optional progress callback.

        Returns:
            The plan text.
        """
        messages.append(HumanMessage(content=STEP1_INSTRUCTION))
        _emit(on_event, {"kind": "plan_start"})
        response: AIMessage = self.chat_model.invoke(messages)
        text = message_text(response)
        messages.append(AIMessage(content=text))
        _emit(on_event, {"kind": "plan", "text": text})
        self.service.record_plan(work_id, movement_id, text)
        return text

    def _compress(
        self,
        messages: list[BaseMessage],
        work_id: str,
        movement_id: str,
        on_event: ProgressCallback | None,
    ) -> None:
        """Compress the session when it exceeds 90% of the context window.

        Args:
            messages: Conversation messages, modified in place.
            work_id: Active work.
            movement_id: Active movement.
            on_event: Optional progress callback.
        """
        compressed = compress_messages(
            self.chat_model,
            messages,
            context_window=self.context_window,
            artifact_label="当前完整 MusicXML",
            artifact_provider=lambda: self.service.current_musicxml(work_id, movement_id),
            pinned_provider=lambda: self.service.latest_plan(work_id) or "",
        )
        if compressed:
            _emit(on_event, {"kind": "compress", "messages": len(messages)})

    def _review(
        self,
        work_id: str,
        movement_id: str,
        goal: str,
        genre_name: str,
        on_event: ProgressCallback | None,
    ) -> ReviewResult:
        """Consult the reviewer AI in a fresh session.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            goal: The composition goal.
            genre_name: Display name of the active genre.
            on_event: Optional progress callback.

        Returns:
            The reviewer verdict.
        """
        assert self.reviewer is not None
        _emit(on_event, {"kind": "review_start"})
        work = self.service.get_work(work_id)
        style = self.service.style_for(work)
        rules = self.service.effective_rules(work)
        report = self.service.check(work_id, movement_id)
        result = self.reviewer.review(
            goal=goal,
            genre_name=genre_name,
            style_name=style.name,
            rules=rules,
            score_xml=self.service.current_musicxml(work_id, movement_id),
            check_summary=format_feedback(report),
        )
        _emit(
            on_event,
            {"kind": "review", "passed": result.passed, "suggestions": result.suggestions},
        )
        return result


def execute_call(mapping: dict[str, BaseTool], call: ToolCall) -> ToolMessage:
    """Execute a single tool call.

    Args:
        mapping: Tool name to tool.
        call: The tool call.

    Returns:
        A tool message carrying the output.
    """
    name = str(call.get("name", ""))
    call_id = str(call.get("id", ""))
    tool = mapping.get(name)
    if tool is None:
        return ToolMessage(content=f"unknown tool: {name}", tool_call_id=call_id)
    args = call.get("args", {})
    try:
        output = tool.invoke(args)
    except Exception as exc:
        output = f"tool error: {exc}"
    return ToolMessage(content=str(output), tool_call_id=call_id)


def run_tool_calls(
    messages: list[BaseMessage],
    response: AIMessage,
    mapping: dict[str, BaseTool],
    steps: int,
    on_event: ProgressCallback | None,
) -> bool:
    """Answer every tool call in a response, valid or not.

    ``langchain_openai`` serialises both ``tool_calls`` and
    ``invalid_tool_calls`` into the request, so an unparseable call must still
    receive a tool response or the provider rejects the next turn.

    Args:
        messages: Conversation messages, modified in place.
        response: The assistant response.
        mapping: Tool name to tool.
        steps: Current step number.
        on_event: Optional progress callback.

    Returns:
        ``True`` when the response contained any tool call.
    """
    invalid = list(getattr(response, "invalid_tool_calls", None) or [])
    if not response.tool_calls and not invalid:
        return False
    for call in response.tool_calls:
        name = str(call.get("name", ""))
        _emit(
            on_event,
            {"kind": "tool_call", "step": steps, "tool": name, "args": call.get("args", {})},
        )
        message = execute_call(mapping, call)
        _emit(on_event, {"kind": "tool_result", "step": steps, **_summarize(name, message)})
        messages.append(message)
    for call in invalid:
        name = str(call.get("name", ""))
        _emit(on_event, {"kind": "tool_call", "step": steps, "tool": name, "args": {}})
        message = ToolMessage(
            content=(
                "工具参数解析失败(通常是 JSON 转义错误),请重新调用并给出合法 JSON:"
                f"{call.get('error', '')}。"
                "请避免一次性手工书写大段 MusicXML,改用 "
                "submit_theme / add_part / technique_* / edit 分步构建。"
            ),
            tool_call_id=str(call.get("id", "")),
        )
        _emit(on_event, {"kind": "tool_result", "step": steps, **_summarize(name, message)})
        messages.append(message)
    return True


def _emit(on_event: ProgressCallback | None, event: dict[str, Any]) -> None:
    """Forward an event to the progress callback when one is set.

    Args:
        on_event: Progress callback or ``None``.
        event: Event payload.
    """
    if on_event is not None:
        on_event(event)


def _summarize(name: str, message: ToolMessage) -> dict[str, Any]:
    """Summarise a tool message for progress reporting.

    Args:
        name: Tool name.
        message: The tool message.

    Returns:
        A dictionary with ``tool``, ``ok``, ``message`` and ``violations``.
    """
    payload: dict[str, Any] = {"tool": name, "ok": None, "message": str(message.content)[:400]}
    try:
        data = json.loads(str(message.content))
    except json.JSONDecodeError, TypeError:
        return payload
    if isinstance(data, dict):
        payload["ok"] = data.get("ok")
        payload["message"] = str(data.get("message", ""))
        payload["violations"] = data.get("violations", [])
        payload["theme_id"] = data.get("theme_id")
    return payload
