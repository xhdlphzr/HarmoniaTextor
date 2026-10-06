# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the LangChain composer agent."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.tool import invalid_tool_call

from harmoniatextor.agent.architect import Architect, plan_tree, render_plan
from harmoniatextor.agent.compression import (
    content_text,
)
from harmoniatextor.agent.loop import AgentLoop
from harmoniatextor.agent.movements import MovementComposer
from harmoniatextor.agent.planning_tools import build_planning_tools
from harmoniatextor.agent.prompts import (
    ARCHITECT_INSTRUCTION,
    ARCHITECT_SYSTEM,
)
from harmoniatextor.agent.reviewer import (
    ReviewerAI,
    ReviewResult,
)
from harmoniatextor.agent.tools import build_tools, result_payload
from harmoniatextor.domain.enums import Severity
from harmoniatextor.domain.models import CheckReport, CheckViolation, ThemeNote
from harmoniatextor.score.io import from_musicxml, new_score, to_musicxml
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.service.service import CompositionService, ToolResult
from harmoniatextor.techniques import build_default_registry

_EXPECTED_TOOL_COUNT = 46
_AUTO_CONTINUE_CALLS = 2
_EXPECTED_MESSAGES = 2
_COMPRESSED_MESSAGES = 1
_REVIEW_ROUNDS = 2
_REPAIRED_MESSAGES = 4
_COMPLETE_MESSAGES = 3


class ScriptedChatModel:
    """A chat-model double that returns a fixed script of messages.

    Attributes:
        responses: Messages returned in order.
        cursor: Index of the next message to return.
    """

    def __init__(self, *, responses: list[AIMessage]) -> None:
        """Initialise the script.

        Args:
            responses: Messages to return in order.
        """
        self.responses = responses
        self.cursor = 0

    def bind_tools(self, _tools: object, **_kwargs: object) -> ScriptedChatModel:
        """Ignore tool binding and return self.

        Args:
            _tools: Bound tools (ignored).
            _kwargs: Extra keyword arguments (ignored).

        Returns:
            This model.
        """
        return self

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return the next scripted message.

        Args:
            _messages: Conversation messages (ignored).
            _kwargs: Extra keyword arguments (ignored).

        Returns:
            The next scripted assistant message.
        """
        message = self.responses[self.cursor]
        self.cursor += 1
        return message


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    """Build an assistant message requesting a tool.

    Args:
        name: The name.
        args: The args.
        call_id: The call id.

    Returns:
        The tool call result.
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def invalid_call(name: str, call_id: str = "x1", error: str = "bad json") -> AIMessage:
    """Build an assistant message with an unparseable tool call.

    Args:
        name: The name.
        call_id: The call id.
        error: The error.

    Returns:
        The invalid call result.
    """
    return AIMessage(
        content="",
        invalid_tool_calls=[
            invalid_tool_call(name=name, args="{bad", id=call_id, error=error)
        ],
    )


def assert_no_dangling_tool_calls(messages: list[BaseMessage]) -> None:
    """Assert every assistant tool call (valid or invalid) has a response.

        ``langchain_openai`` serialises both ``tool_calls`` and
        ``invalid_tool_calls`` into the request's ``tool_calls`` field, so a
        matching tool response is required for either kind.

    Args:
        messages: Conversation messages.
    """
    index = 0
    while index < len(messages):
        valid = getattr(messages[index], "tool_calls", None) or []
        invalid = getattr(messages[index], "invalid_tool_calls", None) or []
        calls = [*valid, *invalid]
        if calls:
            needed = {str(call.get("id", "")) for call in calls}
            answered: set[str] = set()
            scan = index + 1
            while scan < len(messages) and isinstance(messages[scan], ToolMessage):
                answered.add(str(getattr(messages[scan], "tool_call_id", "")))
                scan += 1
            assert needed <= answered, (
                f"dangling tool calls at {index}: {needed - answered}"
            )
        index += 1


class ValidatingChatModel:
    """A scripted model that asserts every request is structurally valid."""

    def __init__(self, *, responses: list[AIMessage]) -> None:
        """Initialise the script.

        Args:
            responses: Messages to return in order.
        """
        self.responses = responses
        self.cursor = 0

    def bind_tools(self, _tools: object, **_kwargs: object) -> ValidatingChatModel:
        """Ignore tool binding and return self.

        Args:
            _tools: The tools.
            _kwargs: The kwargs.

        Returns:
            The bind tools result.
        """
        return self

    def invoke(self, messages: list[BaseMessage], **_kwargs: object) -> AIMessage:
        """Validate the request then return the next scripted message.

        Args:
            messages: Conversation messages.
            _kwargs: The kwargs.

        Returns:
            The invoke result.
        """
        assert_no_dangling_tool_calls(messages)
        response = self.responses[min(self.cursor, len(self.responses) - 1)]
        self.cursor += 1
        return response


class RecordingChatModel(ValidatingChatModel):
    """A validating model that records every request it receives."""

    def __init__(self, *, responses: list[AIMessage]) -> None:
        """Initialise the script and the recording buffer.

        Args:
            responses: The responses.
        """
        super().__init__(responses=responses)
        self.seen: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage], **_kwargs: object) -> AIMessage:
        """Record the request then return the next scripted message.

        Args:
            messages: Conversation messages.
            _kwargs: The kwargs.

        Returns:
            The invoke result.
        """
        self.seen.append(list(messages))
        return super().invoke(messages, **_kwargs)


def theme_xml() -> str:
    """Build a simple theme melody.

    Returns:
        The resulting text.
    """
    score = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["soprano"])
    ScoreEditor(score).write_line(
        "soprano", 1, [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)]
    )
    return to_musicxml(score)


def note_xml(pitch: str) -> str:
    """Build a single-note soprano score.

    Args:
        pitch: Scientific pitch name.

    Returns:
        The resulting text.
    """
    score = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["soprano"])
    ScoreEditor(score).write_line("soprano", 1, [ThemeNote(pitch, 1.0)])
    return to_musicxml(score)


@pytest.fixture
def prepared(service: CompositionService) -> tuple[CompositionService, str, str, int]:
    """Create a work with one submitted theme.

    Args:
        service: The composition service.

    Returns:
        The prepared result.
    """
    work = service.create_work("Demo", "plain", "C")
    movement_id = work.movements[0].id
    result = service.submit_theme(work.id, movement_id, theme_xml())
    return service, work.id, movement_id, result.theme_id or 1


class TestTools:
    """Tool construction and payloads."""

    def test_build_tools(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """Control tools plus 25 technique tools are built.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        tools = build_tools(service, work_id, movement_id)
        names = {tool.name for tool in tools}
        assert "set_title" not in names
        assert "read" in names
        assert "submit_theme" in names
        assert "add_part" in names
        assert "remove_part" in names
        assert "set_tempo" in names
        assert "set_time_signature" in names
        assert "annotate" in names
        assert "edit" in names
        assert "insert" in names
        assert "delete" in names
        assert "refine_score" not in names
        assert len(tools) == _EXPECTED_TOOL_COUNT

    def test_payload_never_includes_score(self) -> None:
        """Mutation payloads never include the full score."""
        payload = result_payload(ToolResult(True, full_musicxml="<x/>"))
        assert "full_musicxml" not in payload

    def test_movement_tools_keep_composition_set(
        self, service: CompositionService
    ) -> None:
        """A movement session keeps themes, edits and every technique.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        names = {tool.name for tool in build_tools(service, work.id, "m01")}
        assert {
            "read",
            "submit_theme",
            "add_part",
            "remove_part",
            "set_tempo",
            "set_time_signature",
            "annotate",
            "edit",
            "insert",
            "delete",
        } <= names
        assert "set_title" not in names
        assert sum(1 for name in names if name.startswith("technique_")) == len(
            build_default_registry().ids()
        )

    def test_invoke_control_tools(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """The submission and edit tools execute through the service.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        tools = {tool.name: tool for tool in build_tools(service, work_id, movement_id)}
        submitted = tools["submit_theme"].invoke(
            {"musicxml": theme_xml(), "instrument": "Soprano"}
        )
        assert "theme_id" in submitted
        added = tools["add_part"].invoke({"voice": "flute", "instrument": "Flute"})
        assert "ok" in added
        removed = tools["remove_part"].invoke({"voice": "flute"})
        assert "ok" in removed
        tempo = tools["set_tempo"].invoke({"bpm": 100})
        assert "ok" in tempo
        meter = tools["set_time_signature"].invoke({"time_signature": "3/4"})
        assert "ok" in meter
        edited = tools["edit"].invoke(
            {"measure": 1, "voice": "soprano", "musicxml": theme_xml()}
        )
        assert "full_musicxml" not in edited
        marked = tools["annotate"].invoke(
            {"measure": 1, "voice": "soprano", "mark": "dynamic", "value": "f"}
        )
        assert "ok" in marked
        inserted = tools["insert"].invoke({"measure": 1})
        assert "full_musicxml" in inserted
        deleted = tools["delete"].invoke({"measure": 1})
        assert "full_musicxml" in deleted
        full = tools["read"].invoke({})
        assert "score-partwise" in full

    def test_payload_with_report(self) -> None:
        """Payloads include violations from the report."""
        report = CheckReport(
            [CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")]
        )
        payload = result_payload(ToolResult(False, report=report))
        assert "violations" in payload

    def test_tools_defer_checking(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """Agent tools defer the symbolic check to the review step.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        tools = {tool.name: tool for tool in build_tools(service, work_id, movement_id)}
        payload = json.loads(
            tools["submit_theme"].invoke(
                {"musicxml": theme_xml(), "instrument": "Soprano"}
            )
        )
        assert payload["ok"] is True
        assert "violations" not in payload
        assert payload["theme_id"] is not None


class TestAgentLoop:
    """The bounded tool-calling loop."""

    def test_completes_after_tool(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """A tool call followed by text completes the run.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, theme_id = prepared
        model = ScriptedChatModel(
            responses=[
                tool_call(
                    "technique_imitation",
                    {
                        "theme_id": theme_id,
                        "target_voice": "bass",
                        "delay_measures": 1,
                        "interval": 5,
                    },
                ),
                AIMessage(content="完成"),
            ]
        )
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work_id, movement_id, "写一个模仿"
        )
        assert outcome.completed
        assert outcome.final_text == "完成"

    def test_unknown_tool(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """Unknown tools are reported back.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(
            responses=[tool_call("does_not_exist", {}), AIMessage(content="done")]
        )
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work_id, movement_id, "goal"
        )
        assert outcome.completed

    def test_tool_error(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """Tool errors are reported back.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(
            responses=[tool_call("technique_imitation", {}), AIMessage(content="done")]
        )
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work_id, movement_id, "goal"
        )
        assert outcome.completed

    def test_max_steps(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """The loop stops at the step budget.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, theme_id = prepared
        call = tool_call(
            "technique_imitation",
            {"theme_id": theme_id, "target_voice": "bass"},
        )
        model = ScriptedChatModel(responses=[call, call, call])
        outcome = AgentLoop(service, cast("BaseChatModel", model), max_steps=2).run(
            work_id, movement_id, "goal"
        )
        assert not outcome.completed

    def test_list_content(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """List content is flattened to text.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(
            responses=[AIMessage(content=[{"type": "text", "text": "hi"}])]
        )
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work_id, movement_id, "goal"
        )
        assert outcome.final_text == "hi"

    def test_content_text_scalar(self) -> None:
        """Non string/list content is stringified."""
        assert content_text(123) == "123"

    def test_feedback(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """Human feedback is appended to the instruction.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(responses=[AIMessage(content="ok")])
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work_id, movement_id, "goal", feedback="more"
        )
        assert outcome.completed

    def test_auto_continue_until_pass(self, service: CompositionService) -> None:
        """A failing check re-prompts the model automatically.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        failing = CheckReport(
            [CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")]
        )
        reports = [failing, CheckReport([])]
        calls: list[int] = []

        def fake_check(*_args: object, **_kwargs: object) -> CheckReport:
            calls.append(1)
            return reports[min(len(calls) - 1, len(reports) - 1)]

        service.check = fake_check  # type: ignore[method-assign]
        events: list[dict[str, Any]] = []
        model = ScriptedChatModel(
            responses=[AIMessage(content="a"), AIMessage(content="b")]
        )
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work.id, movement_id, "goal", on_event=events.append
        )
        assert outcome.completed
        assert outcome.final_text == "b"
        assert len(calls) == _AUTO_CONTINUE_CALLS
        assert any(event["kind"] == "assistant" for event in events)

    def test_auto_continue_disabled(self, service: CompositionService) -> None:
        """A failing check stops immediately when auto-continue is disabled.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        failing = CheckReport(
            [CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")]
        )

        def fake_check(*_args: object, **_kwargs: object) -> CheckReport:
            return failing

        service.check = fake_check  # type: ignore[method-assign]
        model = ScriptedChatModel(responses=[AIMessage(content="x")])
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work.id, movement_id, "goal", auto_continue=False
        )
        assert not outcome.completed

    def test_plan_phase(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """Step 1 plans instruments and arcs before composing.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="规划:钢琴情感由平静到激昂"),
                AIMessage(content="完成"),
            ]
        )
        events: list[dict[str, Any]] = []
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work_id, movement_id, "goal", on_event=events.append, plan=True
        )
        assert outcome.plan == "规划:钢琴情感由平静到激昂"
        assert any(event["kind"] == "plan" for event in events)
        assert service.latest_plan(work_id) == "规划:钢琴情感由平静到激昂"

    def test_stored_plan_injected(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """An existing plan is injected when not planning again.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        service.record_plan(work_id, movement_id, "旧规划")
        model = ScriptedChatModel(responses=[AIMessage(content="完成")])
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work_id, movement_id, "goal"
        )
        assert outcome.completed


class _ReviewModel:
    """A model double returning scripted review responses."""

    def __init__(self, responses: list[AIMessage]) -> None:
        """Initialise the script.

        Args:
            responses: Responses returned in order.
        """
        self.responses = responses
        self.cursor = 0
        self.bound: list[object] = []

    def bind_tools(self, tools: list[object], **_kwargs: object) -> _ReviewModel:
        """Remember the bound tools.

        Args:
            tools: Bound tools.
            _kwargs: Extra keyword arguments (ignored).

        Returns:
            This model.
        """
        self.bound = tools
        return self

    def invoke(self, messages: list[BaseMessage], **_kwargs: object) -> AIMessage:
        """Validate the request then return the next scripted response.

        Args:
            messages: Conversation messages.
            _kwargs: The kwargs.

        Returns:
            The invoke result.
        """
        assert_no_dangling_tool_calls(messages)
        response = self.responses[min(self.cursor, len(self.responses) - 1)]
        self.cursor += 1
        return response


class _ReviewerStub:
    """A reviewer double returning scripted verdicts."""

    def __init__(self, verdicts: list[ReviewResult]) -> None:
        """Initialise the verdict script.

        Args:
            verdicts: Verdicts returned in order.
        """
        self.verdicts = verdicts
        self.calls = 0

    def review(self, **_kwargs: object) -> ReviewResult:
        """Return the next scripted verdict.

        Args:
            _kwargs: The kwargs.

        Returns:
            The review result.
        """
        verdict = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return verdict


class TestAgentLoopReview:
    """The reviewer cycle inside the composer loop."""

    def test_review_pass(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """A passing reviewer completes the run.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(responses=[AIMessage(content="完成")])
        stub = _ReviewerStub([ReviewResult(True)])
        events: list[dict[str, Any]] = []
        outcome = AgentLoop(
            service, cast("BaseChatModel", model), reviewer=cast("ReviewerAI", stub)
        ).run(work_id, movement_id, "goal", on_event=events.append)
        assert outcome.completed
        assert outcome.review_passed is True
        assert any(event["kind"] == "review" and event["passed"] for event in events)

    def test_review_reject_then_pass(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """A rejection re-prompts the same session and can pass later.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(
            responses=[AIMessage(content="a"), AIMessage(content="b")]
        )
        stub = _ReviewerStub([ReviewResult(False, "问题"), ReviewResult(True)])
        outcome = AgentLoop(
            service, cast("BaseChatModel", model), reviewer=cast("ReviewerAI", stub)
        ).run(work_id, movement_id, "goal")
        assert outcome.completed
        assert stub.calls == _REVIEW_ROUNDS

    def test_review_gives_up(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """Repeated rejections stop at the review budget.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(responses=[AIMessage(content="x")] * 4)
        stub = _ReviewerStub([ReviewResult(False, "no")])
        outcome = AgentLoop(
            service,
            cast("BaseChatModel", model),
            reviewer=cast("ReviewerAI", stub),
            max_reviews=2,
        ).run(work_id, movement_id, "goal")
        assert not outcome.completed
        assert outcome.review_passed is False

    def test_compression_event(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """A tiny context window triggers compression mid-run.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ValidatingChatModel(
            responses=[AIMessage(content="摘要"), AIMessage(content="完成")]
        )
        events: list[dict[str, Any]] = []
        outcome = AgentLoop(
            service, cast("BaseChatModel", model), context_window=0
        ).run(work_id, movement_id, "goal", on_event=events.append)
        assert outcome.completed
        assert any(event["kind"] == "compress" for event in events)

    def test_requests_are_structurally_valid(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """Every request keeps tool calls paired with tool responses.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, theme_id = prepared
        model = ValidatingChatModel(
            responses=[
                tool_call(
                    "technique_imitation",
                    {
                        "theme_id": theme_id,
                        "target_voice": "bass",
                        "delay_measures": 1,
                        "interval": 5,
                    },
                ),
                AIMessage(content="a"),
                AIMessage(content="b"),
            ]
        )
        stub = _ReviewerStub([ReviewResult(False, "改改"), ReviewResult(True)])
        outcome = AgentLoop(
            service, cast("BaseChatModel", model), reviewer=cast("ReviewerAI", stub)
        ).run(work_id, movement_id, "goal")
        assert outcome.completed

    def test_invalid_tool_call_is_answered(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """An unparseable tool call is answered, not left dangling.

        Args:
            prepared: A prepared work and movement.
        """
        service, work_id, movement_id, _ = prepared
        model = ValidatingChatModel(
            responses=[
                invalid_call("technique_imitation", call_id="x1"),
                AIMessage(content="a"),
                AIMessage(content="b"),
            ]
        )
        stub = _ReviewerStub([ReviewResult(False, "改改"), ReviewResult(True)])
        outcome = AgentLoop(
            service, cast("BaseChatModel", model), reviewer=cast("ReviewerAI", stub)
        ).run(work_id, movement_id, "goal")
        assert outcome.completed


class TestPlanningTools:
    """Step 1 planning tools."""

    def test_build_and_invoke(self, service: CompositionService) -> None:
        """Title, movement and prompt tools drive the plan.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        tools = {tool.name: tool for tool in build_planning_tools(service, work.id)}
        assert set(tools) == {"set_title", "add_movement", "set_movement_prompt"}
        titled = json.loads(tools["set_title"].invoke({"title": "新标题"}))
        assert titled["ok"] is True
        assert service.get_work(work.id).title == "新标题"
        moved = json.loads(tools["add_movement"].invoke({}))
        assert moved["ok"] is True
        prompt = json.loads(
            tools["set_movement_prompt"].invoke({"movement": 1, "prompt": "写一乐章"})
        )
        assert prompt["ok"] is True
        assert service.get_work(work.id).movements[0].prompt == "写一乐章"

    def test_unknown_movement(self, service: CompositionService) -> None:
        """An out-of-range movement number is reported.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        tools = {tool.name: tool for tool in build_planning_tools(service, work.id)}
        assert (
            json.loads(
                tools["set_movement_prompt"].invoke({"movement": 9, "prompt": "x"})
            )["ok"]
            is False
        )

    def test_prompting_is_documented(self, service: CompositionService) -> None:
        """The planner and tools explain the per-movement prompting.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        tools = {tool.name: tool for tool in build_planning_tools(service, work.id)}
        assert "set_movement_prompt" in ARCHITECT_INSTRUCTION
        assert "set_movement_prompt" in ARCHITECT_SYSTEM
        assert "极其详细" in ARCHITECT_SYSTEM
        assert "没有乐章模板" in ARCHITECT_SYSTEM
        assert "自成一体" in ARCHITECT_SYSTEM
        assert "跨乐章指代" in ARCHITECT_SYSTEM
        assert "独立、自足" in ARCHITECT_INSTRUCTION
        assert "跨乐章指代" in ARCHITECT_INSTRUCTION
        assert "Every movement must get one" in tools["set_movement_prompt"].description


class TestArchitect:
    """The Step 1 planning loop."""

    def test_fills_prompts(self, service: CompositionService) -> None:
        """A plan completes once every movement has a prompt.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        model = ScriptedChatModel(
            responses=[
                tool_call("set_title", {"title": "新标题"}),
                tool_call("add_movement", {}),
                tool_call("set_movement_prompt", {"movement": 1, "prompt": "p1"}),
                tool_call("add_movement", {}),
                tool_call("set_movement_prompt", {"movement": 2, "prompt": "p2"}),
                AIMessage(content="done"),
            ]
        )
        events: list[dict[str, Any]] = []
        result = Architect(service, cast("BaseChatModel", model)).plan(
            work.id, "goal", on_event=events.append
        )
        assert result.completed
        assert service.get_work(work.id).title == "新标题"
        assert "p1" in result.plan
        assert service.latest_plan(work.id) == result.plan
        assert render_plan(service, work.id) == result.plan
        assert result.tree[0]["prompt"] == "p1"
        assert result.tree[1]["prompt"] == "p2"
        assert any(event["kind"] == "plan" for event in events)

    def test_plan_tree(self, service: CompositionService) -> None:
        """The plan tree carries movements and prompts.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        tree = plan_tree(service, work.id)
        assert tree[0]["id"] == "m01"
        assert tree[0]["prompt"] == "p1"

    def test_missing_prompts_nudged(self, service: CompositionService) -> None:
        """An incomplete plan is nudged until covered.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        model = ScriptedChatModel(
            responses=[
                tool_call("add_movement", {}),
                AIMessage(content="done"),
                tool_call("set_movement_prompt", {"movement": 1, "prompt": "p1"}),
                AIMessage(content="done"),
            ]
        )
        result = Architect(service, cast("BaseChatModel", model)).plan(work.id, "goal")
        assert result.completed

    def test_no_movements_nudged(self, service: CompositionService) -> None:
        """A plan without movements is nudged.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="done"),
                tool_call("add_movement", {}),
                tool_call("set_movement_prompt", {"movement": 1, "prompt": "p"}),
                AIMessage(content="done"),
            ]
        )
        result = Architect(service, cast("BaseChatModel", model)).plan(work.id, "goal")
        assert result.completed

    def test_max_rounds(self, service: CompositionService) -> None:
        """Giving up after too many nudges leaves the plan incomplete.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        model = ScriptedChatModel(
            responses=[
                tool_call("add_movement", {}),
                AIMessage(content="done"),
                AIMessage(content="done"),
            ]
        )
        result = Architect(service, cast("BaseChatModel", model), max_rounds=1).plan(
            work.id, "goal"
        )
        assert not result.completed

    def test_max_steps(self, service: CompositionService) -> None:
        """The planner stops at its step budget.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        model = ScriptedChatModel(responses=[AIMessage(content="done")])
        result = Architect(service, cast("BaseChatModel", model), max_steps=1).plan(
            work.id, "goal"
        )
        assert not result.completed


def two_movement_work(service: CompositionService) -> str:
    """Create a work with two prompted free movements.

    Args:
        service: The composition service.

    Returns:
        The resulting text.
    """
    work = service.create_work("Demo", "plain", "C", with_movements=False)
    service.add_movement(work.id)
    service.add_movement(work.id)
    service.set_movement_prompt(work.id, "m01", "p1")
    service.set_movement_prompt(work.id, "m02", "p2")
    return work.id


class TestMovementComposer:
    """Compose and review movements."""

    def test_compose_two_movements(self, service: CompositionService) -> None:
        """Each movement composes in a fresh session, then merges.

        Args:
            service: The composition service.
        """
        work_id = two_movement_work(service)
        model = ValidatingChatModel(
            responses=[
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
            ]
        )
        events: list[dict[str, Any]] = []
        result = MovementComposer(service, cast("BaseChatModel", model)).compose(
            work_id, "goal", on_event=events.append
        )
        assert result.completed
        assert result.merged_musicxml
        assert from_musicxml(result.merged_musicxml).parts
        assert result.review_passed is None
        assert any(event["kind"] == "movement_done" for event in events)
        assert any(event["kind"] == "assistant" for event in events)
        assert any(event["kind"] == "instruction" and event["text"] for event in events)

    def test_movement_check_retries(self, service: CompositionService) -> None:
        """A movement that fails the symbolic check is retried.

        Args:
            service: The composition service.
        """
        work_id = two_movement_work(service)
        failing = CheckReport(
            [CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")]
        )
        reports = [failing, CheckReport([]), CheckReport([])]
        calls: list[int] = []

        def fake_check(*_args: object, **_kwargs: object) -> CheckReport:
            calls.append(1)
            return reports[min(len(calls) - 1, len(reports) - 1)]

        service.check_score = fake_check  # type: ignore[method-assign]
        model = ValidatingChatModel(
            responses=[
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
            ]
        )
        events: list[dict[str, Any]] = []
        result = MovementComposer(service, cast("BaseChatModel", model)).compose(
            work_id, "goal", on_event=events.append
        )
        assert result.completed
        assert len(calls) > 1
        assert any(
            event["kind"] == "feedback"
            and event["layer"] == "symbolic"
            and event["text"]
            for event in events
        )

    def test_review_reject_then_pass(self, service: CompositionService) -> None:
        """A rejected movement is fixed and re-reviewed.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        model = ValidatingChatModel(
            responses=[
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
            ]
        )
        stub = _ReviewerStub([ReviewResult(False, "改改"), ReviewResult(True)])
        events: list[dict[str, Any]] = []
        result = MovementComposer(
            service, cast("BaseChatModel", model), reviewer=cast("ReviewerAI", stub)
        ).compose(work.id, "goal", on_event=events.append)
        assert result.completed
        assert result.review_passed is True
        assert any(
            event["kind"] == "feedback"
            and event["layer"] == "reviewer"
            and "改改" in event["text"]
            for event in events
        )

    def test_review_reuses_movement_session(self, service: CompositionService) -> None:
        """A rejected movement is fixed in its own existing session.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        model = RecordingChatModel(
            responses=[
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
            ]
        )
        stub = _ReviewerStub([ReviewResult(False, "改改"), ReviewResult(True)])
        result = MovementComposer(
            service, cast("BaseChatModel", model), reviewer=cast("ReviewerAI", stub)
        ).compose(work.id, "goal")
        assert result.completed
        fix_request = model.seen[2]
        assert any(isinstance(message, ToolMessage) for message in fix_request)
        assert any("改改" in str(message.content) for message in fix_request)

    def test_review_gives_up(self, service: CompositionService) -> None:
        """Too many rejections end the run.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        model = ValidatingChatModel(
            responses=[
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="done"),
            ]
        )
        stub = _ReviewerStub([ReviewResult(False, "no")])
        result = MovementComposer(
            service,
            cast("BaseChatModel", model),
            reviewer=cast("ReviewerAI", stub),
            max_reviews=1,
        ).compose(work.id, "goal")
        assert not result.completed
        assert result.review_passed is False
        assert result.review_suggestions == "no"

    def test_instruction_has_earlier_themes(self, service: CompositionService) -> None:
        """The instruction carries the earlier movements' themes.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        service.set_movement_prompt(work.id, "m02", "p2")
        service.submit_theme(work.id, "m01", note_xml("A5"), check=False)
        movement = service.get_work(work.id).movements[1]
        composer = MovementComposer(
            service, cast("BaseChatModel", ScriptedChatModel(responses=[]))
        )
        text = composer._instruction(work.id, movement, "goal")
        assert "此前已出现的主题" in text
        assert "A5" in text
        assert "submit_theme" in text

    def test_movement_report_uses_own_score(self, service: CompositionService) -> None:
        """A movement is checked on its own score.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.submit_theme(work.id, "m01", theme_xml(), check=False)
        composer = MovementComposer(
            service, cast("BaseChatModel", ScriptedChatModel(responses=[]))
        )
        movement = service.get_work(work.id).movements[0]
        assert composer._movement_report(work.id, movement).ok

    def test_empty_movement_is_nudged(self, service: CompositionService) -> None:
        """A movement without notes is not accepted as complete.

        Args:
            service: The composition service.
        """
        work_id = two_movement_work(service)
        model = ValidatingChatModel(
            responses=[
                AIMessage(content="我完成了"),
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="好了"),
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="好了"),
            ]
        )
        result = MovementComposer(service, cast("BaseChatModel", model)).compose(
            work_id, "goal"
        )
        assert result.completed
        assert all(
            service._has_notes(work_id, movement.id)
            for movement in service.get_work(work_id).movements
        )

    def test_review_runs_before_next_movement(
        self, service: CompositionService
    ) -> None:
        """Each movement is reviewed before the next one is composed.

        Args:
            service: The composition service.
        """
        work_id = two_movement_work(service)
        model = ValidatingChatModel(
            responses=[
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="a"),
                tool_call(
                    "submit_theme",
                    {
                        "musicxml": theme_xml(),
                        "voice": "soprano",
                        "instrument": "Soprano",
                    },
                ),
                AIMessage(content="b"),
            ]
        )
        stub = _ReviewerStub([ReviewResult(True), ReviewResult(True)])
        events: list[dict[str, Any]] = []
        result = MovementComposer(
            service, cast("BaseChatModel", model), reviewer=cast("ReviewerAI", stub)
        ).compose(work_id, "goal", on_event=events.append)
        assert result.completed
        assert result.review_passed is True
        assert stub.calls == _REVIEW_ROUNDS
        kinds = [event["kind"] for event in events]
        starts = [index for index, kind in enumerate(kinds) if kind == "movement_start"]
        assert kinds.index("review_start") < starts[1]

    def test_compress_reinjects_movement_instruction(
        self, service: CompositionService
    ) -> None:
        """Compressing a movement session re-injects the movement instruction.

        Args:
            service: The composition service.
        """
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        service.submit_theme(work.id, "m01", theme_xml(), check=False)
        movement = service.get_work(work.id).movements[0]
        model = ScriptedChatModel(responses=[AIMessage(content="摘要")])
        composer = MovementComposer(
            service, cast("BaseChatModel", model), context_window=0
        )
        messages: list[BaseMessage] = [HumanMessage(content="旧对话")]
        events: list[dict[str, Any]] = []
        composer._compress(work.id, movement, "goal", events.append, messages)
        assert any(event["kind"] == "compress" for event in events)
        assert any("p1" in str(message.content) for message in messages)
