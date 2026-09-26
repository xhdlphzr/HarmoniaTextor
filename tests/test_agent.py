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
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import invalid_tool_call

from harmoniatextor.agent.architect import Architect, plan_tree, render_plan
from harmoniatextor.agent.compression import (
    _bounded_transcript,
    compress_messages,
    content_text,
    ensure_tool_responses,
    message_text,
    token_count,
)
from harmoniatextor.agent.llm_factory import create_chat_model, resolve_setting
from harmoniatextor.agent.loop import AgentLoop
from harmoniatextor.agent.movements import MovementComposer
from harmoniatextor.agent.planning_tools import build_planning_tools
from harmoniatextor.agent.prompts import (
    ARCHITECT_INSTRUCTION,
    ARCHITECT_SYSTEM,
    system_prompt,
)
from harmoniatextor.agent.reviewer import (
    _SYSTEM,
    ReviewerAI,
    ReviewResult,
    _extract,
    _submit_review,
)
from harmoniatextor.agent.tools import build_tools, result_payload
from harmoniatextor.domain.enums import Severity
from harmoniatextor.domain.models import CheckReport, CheckViolation, ThemeNote
from harmoniatextor.genres import PlainGenre
from harmoniatextor.score.io import from_musicxml, new_score, to_musicxml
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.service.service import CompositionService, ToolResult
from harmoniatextor.techniques import build_default_registry

_EXPECTED_TOOL_COUNT = 32
_AUTO_CONTINUE_CALLS = 2
_COUNTED_TOKENS = 7
_HEURISTIC_MIN = 10
_EXPECTED_MESSAGES = 2
_COMPRESSED_MESSAGES = 1
_REVIEW_ROUNDS = 2
_REPAIRED_MESSAGES = 4
_COMPLETE_MESSAGES = 3
_LONG_TEXT_CHARS = 7000
_TOKENIZER_PER_TEXT = 3


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
    """Build an assistant message requesting a tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def invalid_call(name: str, call_id: str = "x1", error: str = "bad json") -> AIMessage:
    """Build an assistant message with an unparseable tool call."""
    return AIMessage(
        content="",
        invalid_tool_calls=[invalid_tool_call(name=name, args="{bad", id=call_id, error=error)],
    )


def assert_no_dangling_tool_calls(messages: list[BaseMessage]) -> None:
    """Assert every assistant tool call (valid or invalid) has a response.

    ``langchain_openai`` serialises both ``tool_calls`` and
    ``invalid_tool_calls`` into the request's ``tool_calls`` field, so a
    matching tool response is required for either kind.
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
            assert needed <= answered, f"dangling tool calls at {index}: {needed - answered}"
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
        """Ignore tool binding and return self."""
        return self

    def invoke(self, messages: list[BaseMessage], **_kwargs: object) -> AIMessage:
        """Validate the request then return the next scripted message."""
        assert_no_dangling_tool_calls(messages)
        response = self.responses[min(self.cursor, len(self.responses) - 1)]
        self.cursor += 1
        return response


class RecordingChatModel(ValidatingChatModel):
    """A validating model that records every request it receives."""

    def __init__(self, *, responses: list[AIMessage]) -> None:
        """Initialise the script and the recording buffer."""
        super().__init__(responses=responses)
        self.seen: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage], **_kwargs: object) -> AIMessage:
        """Record the request then return the next scripted message."""
        self.seen.append(list(messages))
        return super().invoke(messages, **_kwargs)


def theme_xml() -> str:
    """Build a simple theme melody."""
    score = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["soprano"])
    ScoreEditor(score).write_line("soprano", 1, [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
    return to_musicxml(score)


def note_xml(pitch: str) -> str:
    """Build a single-note soprano score."""
    score = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["soprano"])
    ScoreEditor(score).write_line("soprano", 1, [ThemeNote(pitch, 1.0)])
    return to_musicxml(score)


@pytest.fixture
def prepared(service: CompositionService) -> tuple[CompositionService, str, str, int]:
    """Create a work with one submitted theme."""
    work = service.create_work("Demo", "plain", "C")
    movement_id = work.movements[0].id
    result = service.submit_theme(work.id, movement_id, theme_xml())
    return service, work.id, movement_id, result.theme_id or 1


class TestLLMFactory:
    """Chat model factory."""

    @pytest.fixture(autouse=True)
    def _no_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Isolate the factory from the user's real config file."""
        monkeypatch.setattr("harmoniatextor.agent.llm_factory.load_config", lambda: {})

    def test_explicit_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit arguments are forwarded."""
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        created = cast("object", create_chat_model(model="m", base_url="u", api_key="k"))
        assert created == "model"
        assert captured["model"] == "m"

    def test_environment_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables are used as defaults."""
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_BASE_URL", "env-url")
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured.update({key: kwargs[key] for key in ("model", "base_url", "api_key")})
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        create_chat_model()
        assert captured == {"model": "env-model", "base_url": "env-url", "api_key": "env-key"}

    def test_openai_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OPENAI_API_KEY is used when LLM_API_KEY is absent."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured["api_key"] = kwargs["api_key"]
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        create_chat_model()
        assert captured["api_key"] == "openai-key"

    def test_config_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The config file overrides environment and defaults."""
        monkeypatch.setattr(
            "harmoniatextor.agent.llm_factory.load_config",
            lambda: {
                "model": "cfg-model",
                "base_url": "cfg-url",
                "api_key": "cfg-key",
            },
        )
        captured: dict[str, Any] = {}

        def fake(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "model"

        monkeypatch.setattr("harmoniatextor.agent.llm_factory.ChatOpenAI", fake)
        create_chat_model()
        assert captured["model"] == "cfg-model"
        assert captured["base_url"] == "cfg-url"
        assert captured["api_key"] == "cfg-key"
        assert "temperature" not in captured


class TestResolveSetting:
    """Setting precedence resolution."""

    def test_explicit_wins(self) -> None:
        """An explicit value wins over everything."""
        assert resolve_setting("x", {"key": "y"}, "key", "ENV", "d") == "x"

    def test_config_wins(self) -> None:
        """The config file wins over environment and defaults."""
        assert resolve_setting(None, {"key": "y"}, "key", "ENV", "d") == "y"

    def test_environment_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The environment wins over the default."""
        monkeypatch.setenv("ENV", "e")
        assert resolve_setting(None, {}, "key", "ENV", "d") == "e"

    def test_default(self) -> None:
        """The default is used when nothing else is set."""
        assert resolve_setting(None, {}, "key", "ENV", "d") == "d"

    def test_without_env_name(self) -> None:
        """A missing environment name falls back to the default."""
        assert resolve_setting(None, {}, "key", None, "d") == "d"


class TestTools:
    """Tool construction and payloads."""

    def test_build_tools(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """Control tools plus 25 technique tools are built."""
        service, work_id, movement_id, _ = prepared
        tools = build_tools(service, work_id, movement_id)
        names = {tool.name for tool in tools}
        assert "set_title" not in names
        assert "read" in names
        assert "submit_theme" in names
        assert "add_part" in names
        assert "edit" in names
        assert "insert" in names
        assert "delete" in names
        assert "overwrite" in names
        assert "refine_score" not in names
        assert len(tools) == _EXPECTED_TOOL_COUNT

    def test_payload_never_includes_score(self) -> None:
        """Mutation payloads never include the full score."""
        payload = result_payload(ToolResult(True, full_musicxml="<x/>"))
        assert "full_musicxml" not in payload

    def test_movement_tools_keep_composition_set(self, service: CompositionService) -> None:
        """A movement session keeps themes, edits and every technique."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        names = {tool.name for tool in build_tools(service, work.id, "m01")}
        assert {
            "read",
            "submit_theme",
            "add_part",
            "edit",
            "insert",
            "delete",
            "overwrite",
        } <= names
        assert "set_title" not in names
        assert sum(1 for name in names if name.startswith("technique_")) == len(
            build_default_registry().ids()
        )

    def test_invoke_control_tools(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """The submission and edit tools execute through the service."""
        service, work_id, movement_id, _ = prepared
        tools = {tool.name: tool for tool in build_tools(service, work_id, movement_id)}
        submitted = tools["submit_theme"].invoke({"musicxml": theme_xml()})
        assert "theme_id" in submitted
        added = tools["add_part"].invoke({"voice": "flute", "instrument": "Flute"})
        assert "ok" in added
        edited = tools["edit"].invoke({"measure": 1, "voice": "soprano", "musicxml": theme_xml()})
        assert "full_musicxml" not in edited
        inserted = tools["insert"].invoke({"measure": 1})
        assert "full_musicxml" in inserted
        deleted = tools["delete"].invoke({"measure": 1})
        assert "full_musicxml" in deleted
        overwritten = tools["overwrite"].invoke({"musicxml": theme_xml()})
        assert "full_musicxml" not in overwritten
        full = tools["read"].invoke({})
        assert "score-partwise" in full

    def test_payload_with_report(self) -> None:
        """Payloads include violations from the report."""
        report = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])
        payload = result_payload(ToolResult(False, report=report))
        assert "violations" in payload

    def test_tools_defer_checking(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """Agent tools defer the symbolic check to the review step."""
        service, work_id, movement_id, _ = prepared
        tools = {tool.name: tool for tool in build_tools(service, work_id, movement_id)}
        payload = json.loads(tools["submit_theme"].invoke({"musicxml": theme_xml()}))
        assert payload["ok"] is True
        assert "violations" not in payload
        assert payload["theme_id"] is not None


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


class TestAgentLoop:
    """The bounded tool-calling loop."""

    def test_completes_after_tool(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """A tool call followed by text completes the run."""
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

    def test_unknown_tool(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """Unknown tools are reported back."""
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(
            responses=[tool_call("does_not_exist", {}), AIMessage(content="done")]
        )
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(work_id, movement_id, "goal")
        assert outcome.completed

    def test_tool_error(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """Tool errors are reported back."""
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(
            responses=[tool_call("technique_imitation", {}), AIMessage(content="done")]
        )
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(work_id, movement_id, "goal")
        assert outcome.completed

    def test_max_steps(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """The loop stops at the step budget."""
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

    def test_list_content(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """List content is flattened to text."""
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(responses=[AIMessage(content=[{"type": "text", "text": "hi"}])])
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(work_id, movement_id, "goal")
        assert outcome.final_text == "hi"

    def test_content_text_scalar(self) -> None:
        """Non string/list content is stringified."""
        assert content_text(123) == "123"

    def test_feedback(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """Human feedback is appended to the instruction."""
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(responses=[AIMessage(content="ok")])
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work_id, movement_id, "goal", feedback="more"
        )
        assert outcome.completed

    def test_auto_continue_until_pass(self, service: CompositionService) -> None:
        """A failing check re-prompts the model automatically."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        failing = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])
        reports = [failing, CheckReport([])]
        calls: list[int] = []

        def fake_check(*_args: object, **_kwargs: object) -> CheckReport:
            calls.append(1)
            return reports[min(len(calls) - 1, len(reports) - 1)]

        service.check = fake_check  # type: ignore[method-assign]
        events: list[dict[str, Any]] = []
        model = ScriptedChatModel(responses=[AIMessage(content="a"), AIMessage(content="b")])
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work.id, movement_id, "goal", on_event=events.append
        )
        assert outcome.completed
        assert outcome.final_text == "b"
        assert len(calls) == _AUTO_CONTINUE_CALLS
        assert any(event["kind"] == "assistant" for event in events)

    def test_auto_continue_disabled(self, service: CompositionService) -> None:
        """A failing check stops immediately when auto-continue is disabled."""
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        failing = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])

        def fake_check(*_args: object, **_kwargs: object) -> CheckReport:
            return failing

        service.check = fake_check  # type: ignore[method-assign]
        model = ScriptedChatModel(responses=[AIMessage(content="x")])
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(
            work.id, movement_id, "goal", auto_continue=False
        )
        assert not outcome.completed

    def test_plan_phase(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """Step 1 plans instruments and arcs before composing."""
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

    def test_stored_plan_injected(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """An existing plan is injected when not planning again."""
        service, work_id, movement_id, _ = prepared
        service.record_plan(work_id, movement_id, "旧规划")
        model = ScriptedChatModel(responses=[AIMessage(content="完成")])
        outcome = AgentLoop(service, cast("BaseChatModel", model)).run(work_id, movement_id, "goal")
        assert outcome.completed


class _SummarizerModel:
    """A model double that counts tokens and returns a summary."""

    def __init__(self, *, tokens: int = 0, content: str = "摘要") -> None:
        """Initialise the double.

        Args:
            tokens: Token count reported to the caller.
            content: Summary returned by ``invoke``.
        """
        self.tokens = tokens
        self.content = content
        self.invocations = 0

    def get_num_tokens_from_messages(self, _messages: object) -> int:
        """Return the configured token count."""
        return self.tokens

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return the configured summary."""
        self.invocations += 1
        return AIMessage(content=self.content)


class _NoCountModel:
    """A model double without a token counter."""

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return a summary."""
        return AIMessage(content="摘要")


class _BrokenCountModel:
    """A model double whose token counter fails."""

    def get_num_tokens_from_messages(self, _messages: object) -> int:
        """Raise to exercise the fallback path."""
        raise RuntimeError("no counter")

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return a summary."""
        return AIMessage(content="摘要")


class _TokenizerModel:
    """A model double exposing only a per-text tokenizer."""

    def get_num_tokens(self, _text: str) -> int:
        """Return a fixed per-text token count."""
        return _TOKENIZER_PER_TEXT

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return a summary."""
        return AIMessage(content="摘要")


class _BrokenTokenizerModel:
    """A model double whose every tokenizer fails."""

    def get_num_tokens_from_messages(self, _messages: object) -> int:
        """Raise to skip the full-message counter."""
        raise RuntimeError("no counter")

    def get_num_tokens(self, _text: str) -> int:
        """Raise to force the heuristic."""
        raise RuntimeError("no tokenizer")

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return a summary."""
        return AIMessage(content="摘要")


class TestCompression:
    """Conversation compression."""

    def test_content_text_list(self) -> None:
        """List content is flattened to text."""
        assert content_text([{"text": "hi"}, "yo"]) == "hiyo"

    def test_bounded_transcript(self) -> None:
        """A long history is truncated and bounded for the summariser."""
        messages: list[BaseMessage] = [
            HumanMessage(content="x" * _LONG_TEXT_CHARS) for _ in range(5)
        ]
        text = _bounded_transcript(messages, budget_chars=50)
        assert "截断" in text
        assert len(text) < _LONG_TEXT_CHARS

    def test_message_text_content(self) -> None:
        """Visible content is preferred over reasoning."""
        message = AIMessage(content="可见文字", additional_kwargs={"reasoning_content": "思考"})
        assert message_text(message) == "可见文字"

    def test_message_text_reasoning_fallback(self) -> None:
        """Reasoning content is surfaced when the content is empty."""
        message = AIMessage(content="", additional_kwargs={"reasoning_content": "思考"})
        assert message_text(message) == "思考"

    def test_message_text_reasoning_alias(self) -> None:
        """The ``reasoning`` alias is also accepted."""
        message = AIMessage(content="", additional_kwargs={"reasoning": "推理"})
        assert message_text(message) == "推理"

    def test_message_text_empty(self) -> None:
        """A message without any text yields an empty string."""
        assert message_text(AIMessage(content="")) == ""

    def test_token_count_counter(self) -> None:
        """A model counter is preferred."""
        model = _SummarizerModel(tokens=_COUNTED_TOKENS)
        counted = token_count(cast("BaseChatModel", model), [HumanMessage(content="x")])
        assert counted == _COUNTED_TOKENS

    def test_token_count_fallback(self) -> None:
        """The character heuristic is used without a counter."""
        model = _NoCountModel()
        heuristic = token_count(cast("BaseChatModel", model), [HumanMessage(content="a" * 40)])
        assert heuristic >= _HEURISTIC_MIN

    def test_token_count_broken(self) -> None:
        """A failing counter falls back to the heuristic."""
        model = _BrokenCountModel()
        heuristic = token_count(cast("BaseChatModel", model), [HumanMessage(content="a" * 40)])
        assert heuristic >= _HEURISTIC_MIN

    def test_token_count_tokenizer(self) -> None:
        """A per-text tokenizer is used when the full-message counter is absent."""
        model = _TokenizerModel()
        counted = token_count(cast("BaseChatModel", model), [HumanMessage(content="你好")])
        assert counted == _TOKENIZER_PER_TEXT + 1

    def test_token_count_broken_tokenizer(self) -> None:
        """A failing tokenizer falls back to the heuristic."""
        model = _BrokenTokenizerModel()
        heuristic = token_count(cast("BaseChatModel", model), [HumanMessage(content="a" * 40)])
        assert heuristic >= _HEURISTIC_MIN

    def test_token_count_tokenizer_with_calls(self) -> None:
        """Tool-call arguments are counted by the tokenizer too."""
        model = _TokenizerModel()
        counted = token_count(cast("BaseChatModel", model), [tool_call("read", {})])
        assert counted == _TOKENIZER_PER_TEXT * 2 + 1

    def test_below_threshold(self) -> None:
        """Short conversations are left untouched."""
        model = _SummarizerModel(tokens=1)
        messages = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert not compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
        )
        assert model.invocations == 0

    def test_compress_with_system(self) -> None:
        """A long conversation is folded into the system message."""
        model = _SummarizerModel(tokens=10_000)
        messages: list[BaseMessage] = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
        )
        assert model.invocations == 1
        assert len(messages) == _COMPRESSED_MESSAGES
        text = content_text(messages[0].content)
        assert "sys" in text
        assert "摘要" in text
        assert "<xml/>" in text

    def test_compress_pins_plan(self) -> None:
        """Pinned text is kept verbatim in the system message."""
        model = _SummarizerModel(tokens=10_000)
        messages: list[BaseMessage] = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
            pinned_provider=lambda: "Step 1 规划内容",
        )
        assert "Step 1 规划内容" in content_text(messages[0].content)

    def test_compress_without_system(self) -> None:
        """A conversation without a system prompt keeps the summary and artifact."""
        model = _SummarizerModel(tokens=10_000)
        messages: list[BaseMessage] = [HumanMessage(content="hi")]
        assert compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
        )
        assert len(messages) == _COMPRESSED_MESSAGES
        assert "<xml/>" in content_text(messages[0].content)

    def test_compress_renders_tool_calls(self) -> None:
        """Tool calls are included in the summary transcript."""
        model = _SummarizerModel(tokens=10_000)
        messages: list[BaseMessage] = [
            SystemMessage(content="sys"),
            tool_call("technique_imitation", {"theme_id": 1}),
        ]
        assert compress_messages(
            cast("BaseChatModel", model),
            messages,
            context_window=1000,
            artifact_label="乐谱",
            artifact_provider=lambda: "<xml/>",
        )

    def test_ensure_tool_responses(self) -> None:
        """A dangling tool call gets a placeholder tool response."""
        messages: list[BaseMessage] = [
            SystemMessage(content="sys"),
            tool_call("technique_imitation", {"theme_id": 1}, call_id="c1"),
            HumanMessage(content="continue"),
        ]
        ensure_tool_responses(messages)
        assert len(messages) == _REPAIRED_MESSAGES
        assert isinstance(messages[2], ToolMessage)
        assert messages[2].tool_call_id == "c1"

    def test_ensure_tool_responses_complete(self) -> None:
        """A complete history is left untouched."""
        messages: list[BaseMessage] = [
            SystemMessage(content="sys"),
            tool_call("technique_imitation", {"theme_id": 1}, call_id="c1"),
            ToolMessage(content="ok", tool_call_id="c1"),
        ]
        ensure_tool_responses(messages)
        assert len(messages) == _COMPLETE_MESSAGES

    def test_ensure_tool_responses_invalid(self) -> None:
        """An unparseable tool call also gets a placeholder response."""
        messages: list[BaseMessage] = [
            SystemMessage(content="sys"),
            invalid_call("technique_imitation", call_id="x1"),
            HumanMessage(content="continue"),
        ]
        ensure_tool_responses(messages)
        assert len(messages) == _REPAIRED_MESSAGES
        assert isinstance(messages[2], ToolMessage)
        assert messages[2].tool_call_id == "x1"


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
        """Validate the request then return the next scripted response."""
        assert_no_dangling_tool_calls(messages)
        response = self.responses[min(self.cursor, len(self.responses) - 1)]
        self.cursor += 1
        return response


class TestReviewer:
    """The independent reviewer AI."""

    def test_pass(self) -> None:
        """A passing verdict is returned."""
        model = _ReviewModel([tool_call("submit_review", {"passed": True, "suggestions": ""})])
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert result.passed

    def test_reject(self) -> None:
        """A rejection carries its suggestions."""
        model = _ReviewModel([tool_call("submit_review", {"passed": False, "suggestions": "问题"})])
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert not result.passed
        assert result.suggestions == "问题"

    def test_no_verdict(self) -> None:
        """A model that never submits yields a rejection."""
        model = _ReviewModel([AIMessage(content="再看看")])
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert not result.passed
        assert result.suggestions

    def test_extract_ignores_other_tools(self) -> None:
        """Non-submit tool calls are ignored."""
        assert _extract(tool_call("other", {})) is None

    def test_other_tool_call(self) -> None:
        """A non-submit tool call is answered and retried."""
        model = _ReviewModel(
            [
                tool_call("other", {}, call_id="x1"),
                tool_call("submit_review", {"passed": True, "suggestions": ""}),
            ]
        )
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert result.passed

    def test_invalid_submit_call(self) -> None:
        """An unparseable submit call is answered and retried."""
        model = _ReviewModel(
            [
                invalid_call("submit_review", call_id="x1"),
                tool_call("submit_review", {"passed": True, "suggestions": ""}),
            ]
        )
        result = ReviewerAI(cast("BaseChatModel", model)).review(
            goal="g", genre_name="赋格", score_xml="<xml/>", check_summary="通过"
        )
        assert result.passed

    def test_submit_review_tool(self) -> None:
        """The submit tool acknowledges the verdict."""
        assert _submit_review(True) == "已收到评审结论。"

    def test_system_checks_movement_division(self) -> None:
        """The reviewer is told to check movement division."""
        assert "乐章划分" in _SYSTEM

    def test_system_excludes_symbolic_rules(self) -> None:
        """The reviewer is told not to re-check symbolic-layer rules."""
        assert "平行五度" in _SYSTEM
        assert "以这些规则为由打回" in _SYSTEM
        assert "符号层" in _SYSTEM


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
        """Return the next scripted verdict."""
        verdict = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return verdict


class TestAgentLoopReview:
    """The reviewer cycle inside the composer loop."""

    def test_review_pass(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """A passing reviewer completes the run."""
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
        """A rejection re-prompts the same session and can pass later."""
        service, work_id, movement_id, _ = prepared
        model = ScriptedChatModel(responses=[AIMessage(content="a"), AIMessage(content="b")])
        stub = _ReviewerStub([ReviewResult(False, "问题"), ReviewResult(True)])
        outcome = AgentLoop(
            service, cast("BaseChatModel", model), reviewer=cast("ReviewerAI", stub)
        ).run(work_id, movement_id, "goal")
        assert outcome.completed
        assert stub.calls == _REVIEW_ROUNDS

    def test_review_gives_up(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """Repeated rejections stop at the review budget."""
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

    def test_compression_event(self, prepared: tuple[CompositionService, str, str, int]) -> None:
        """A tiny context window triggers compression mid-run."""
        service, work_id, movement_id, _ = prepared
        model = ValidatingChatModel(
            responses=[AIMessage(content="摘要"), AIMessage(content="完成")]
        )
        events: list[dict[str, Any]] = []
        outcome = AgentLoop(service, cast("BaseChatModel", model), context_window=0).run(
            work_id, movement_id, "goal", on_event=events.append
        )
        assert outcome.completed
        assert any(event["kind"] == "compress" for event in events)

    def test_requests_are_structurally_valid(
        self, prepared: tuple[CompositionService, str, str, int]
    ) -> None:
        """Every request keeps tool calls paired with tool responses."""
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
        """An unparseable tool call is answered, not left dangling."""
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
        """Title, movement and prompt tools drive the plan."""
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
        """An out-of-range movement number is reported."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        tools = {tool.name: tool for tool in build_planning_tools(service, work.id)}
        assert (
            json.loads(tools["set_movement_prompt"].invoke({"movement": 9, "prompt": "x"}))["ok"]
            is False
        )

    def test_prompting_is_documented(self, service: CompositionService) -> None:
        """The planner and tools explain the per-movement prompting."""
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
        """A plan completes once every movement has a prompt."""
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
        """The plan tree carries movements and prompts."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        tree = plan_tree(service, work.id)
        assert tree[0]["id"] == "m01"
        assert tree[0]["prompt"] == "p1"

    def test_missing_prompts_nudged(self, service: CompositionService) -> None:
        """An incomplete plan is nudged until covered."""
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
        """A plan without movements is nudged."""
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
        """Giving up after too many nudges leaves the plan incomplete."""
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
        """The planner stops at its step budget."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        model = ScriptedChatModel(responses=[AIMessage(content="done")])
        result = Architect(service, cast("BaseChatModel", model), max_steps=1).plan(work.id, "goal")
        assert not result.completed


def two_movement_work(service: CompositionService) -> str:
    """Create a work with two prompted free movements."""
    work = service.create_work("Demo", "plain", "C", with_movements=False)
    service.add_movement(work.id)
    service.add_movement(work.id)
    service.set_movement_prompt(work.id, "m01", "p1")
    service.set_movement_prompt(work.id, "m02", "p2")
    return work.id


class TestMovementComposer:
    """Compose and review movements."""

    def test_compose_two_movements(self, service: CompositionService) -> None:
        """Each movement composes in a fresh session, then merges."""
        work_id = two_movement_work(service)
        model = ValidatingChatModel(
            responses=[
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="done"),
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
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
        """A movement that fails the symbolic check is retried."""
        work_id = two_movement_work(service)
        failing = CheckReport([CheckViolation("r", Severity.ERROR, 1, None, None, "k", "m", "s")])
        reports = [failing, CheckReport([]), CheckReport([])]
        calls: list[int] = []

        def fake_check(*_args: object, **_kwargs: object) -> CheckReport:
            calls.append(1)
            return reports[min(len(calls) - 1, len(reports) - 1)]

        service.check_score = fake_check  # type: ignore[method-assign]
        model = ValidatingChatModel(
            responses=[
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="done"),
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="done"),
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
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
            event["kind"] == "feedback" and event["layer"] == "symbolic" and event["text"]
            for event in events
        )

    def test_review_reject_then_pass(self, service: CompositionService) -> None:
        """A rejected movement is fixed and re-reviewed."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        model = ValidatingChatModel(
            responses=[
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="done"),
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
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
            event["kind"] == "feedback" and event["layer"] == "reviewer" and "改改" in event["text"]
            for event in events
        )

    def test_review_reuses_movement_session(self, service: CompositionService) -> None:
        """A rejected movement is fixed in its own existing session."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        model = RecordingChatModel(
            responses=[
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="done"),
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
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
        """Too many rejections end the run."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        model = ValidatingChatModel(
            responses=[
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="done"),
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="done"),
            ]
        )
        stub = _ReviewerStub([ReviewResult(False, "no")])
        result = MovementComposer(
            service, cast("BaseChatModel", model), reviewer=cast("ReviewerAI", stub), max_reviews=1
        ).compose(work.id, "goal")
        assert not result.completed
        assert result.review_passed is False
        assert result.review_suggestions == "no"

    def test_instruction_has_earlier_themes(self, service: CompositionService) -> None:
        """The instruction carries the earlier movements' themes."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        service.set_movement_prompt(work.id, "m02", "p2")
        service.submit_theme(work.id, "m01", note_xml("A5"), check=False)
        movement = service.get_work(work.id).movements[1]
        composer = MovementComposer(service, cast("BaseChatModel", ScriptedChatModel(responses=[])))
        text = composer._instruction(work.id, movement, "goal")
        assert "此前已出现的主题" in text
        assert "A5" in text
        assert "submit_theme" in text

    def test_movement_report_uses_own_score(self, service: CompositionService) -> None:
        """A movement is checked on its own score."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.submit_theme(work.id, "m01", theme_xml(), check=False)
        composer = MovementComposer(service, cast("BaseChatModel", ScriptedChatModel(responses=[])))
        movement = service.get_work(work.id).movements[0]
        assert composer._movement_report(work.id, movement).ok

    def test_empty_movement_is_nudged(self, service: CompositionService) -> None:
        """A movement without notes is not accepted as complete."""
        work_id = two_movement_work(service)
        model = ValidatingChatModel(
            responses=[
                AIMessage(content="我完成了"),
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="好了"),
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="好了"),
            ]
        )
        result = MovementComposer(service, cast("BaseChatModel", model)).compose(work_id, "goal")
        assert result.completed
        assert all(
            service._has_notes(work_id, movement.id)
            for movement in service.get_work(work_id).movements
        )

    def test_review_runs_before_next_movement(self, service: CompositionService) -> None:
        """Each movement is reviewed before the next one is composed."""
        work_id = two_movement_work(service)
        model = ValidatingChatModel(
            responses=[
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
                AIMessage(content="a"),
                tool_call("submit_theme", {"musicxml": theme_xml(), "voice": "soprano"}),
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

    def test_compress_reinjects_movement_instruction(self, service: CompositionService) -> None:
        """Compressing a movement session re-injects the movement instruction."""
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.set_movement_prompt(work.id, "m01", "p1")
        service.submit_theme(work.id, "m01", theme_xml(), check=False)
        movement = service.get_work(work.id).movements[0]
        model = ScriptedChatModel(responses=[AIMessage(content="摘要")])
        composer = MovementComposer(service, cast("BaseChatModel", model), context_window=0)
        messages: list[BaseMessage] = [HumanMessage(content="旧对话")]
        events: list[dict[str, Any]] = []
        composer._compress(work.id, movement, "goal", events.append, messages)
        assert any(event["kind"] == "compress" for event in events)
        assert any("p1" in str(message.content) for message in messages)
