# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Independent reviewer AI ("check AI") for composer sessions.

Every call to :meth:`ReviewerAI.review` opens a brand-new conversation, so the
reviewer never carries state between verdicts.  When it rejects a score the
creator session receives the suggestions and keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from harmoniatextor.agent.compression import (
    DEFAULT_CONTEXT_WINDOW,
    compress_messages,
    content_text,
    ensure_tool_responses,
)
from harmoniatextor.checker.rules import BUILTIN_RULES
from harmoniatextor.i18n import translate

__all__ = ["ReviewResult", "ReviewerAI"]

_REVIEW_TURNS = 3

_SYSTEM_TEMPLATE = (
    "You are a strict, independent reviewer of {style} music. You only review; "
    "you never edit the score.\n"
    "Important: mechanical theory rules such as {rules} have already been "
    "strictly checked and guaranteed by the programmatic symbolic layer - do "
    "**not** re-check them and do **not** reject on their account; judge only "
    "artistic, stylistic and expressive matters.\n"
    "Review the work along these dimensions:\n"
    "1. Requirements: check the [Movement requirement (Step 1 composition plan, "
    "incl. the user goal)] item by item; if any is unmet you must reject, saying "
    "which one is missing and how to add it.\n"
    "2. {style} style: whether it matches the style's thematic development, "
    "texture, harmony and tone, rather than mechanical patchwork or an "
    "incongruous style.\n"
    "3. Structural completeness: whether the themes are fully developed and the "
    "whole is shaped rather than patchwork.\n"
    "4. Mood and emotion: whether the work expresses a unified, sincere mood and "
    "emotion rather than mechanical patchwork.\n"
    "5. Rhythm and melody: the hands'/voices' rhythms should not be too "
    "identical and the overall rhythm should not be too even, yet changes must "
    "not be abrupt; the melodic writing should flow naturally and not be "
    "disjointed.\n"
    "6. Playability: whether the rhythm and writing suit the actual instruments.\n"
    "The work may be single- or multi-movement: say so if multi-movement fits "
    "better, but do not force it; if it is multi-movement, make sure there are "
    "clear movement divisions (each movement's start/end, tempo and role); if "
    "they are missing you must reject and explain how to divide them.\n"
    "After reviewing you must call the submit_review tool: passed says whether "
    "it passes; when rejecting, suggestions must give, item by item, the measure "
    "number, the voices involved (who and who) and the concrete fix - do not "
    "give vague conclusions."
)


def _system_text(style_name: str, rules: frozenset[str]) -> str:
    """Render the reviewer system prompt for a style.

    Args:
        style_name: Active style display name.
        rules: Rule identifiers enforced for the work.

    Returns:
        The system prompt text.
    """
    names = ", ".join(
        translate("rule." + rule.rule_id, "en")
        for rule in BUILTIN_RULES
        if rule.rule_id in rules
    )
    return _SYSTEM_TEMPLATE.format(style=style_name, rules=names)


class _ReviewParams(BaseModel):
    """Arguments of the reviewer's submit tool.

    Attributes:
        passed: Whether the score passes the review.
        suggestions: Concrete problems and fixes when the score is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(description="Whether the score passes the review.")
    suggestions: str = Field(
        default="",
        description="Concrete problems and fixes; required when passed is false.",
    )


def _submit_review(passed: bool, suggestions: str = "") -> str:
    """Record the reviewer verdict.

    Args:
        passed: Whether the score passes.
        suggestions: Concrete problems and fixes.

    Returns:
        A confirmation string.
    """
    return "Review received."


_REVIEW_TOOL = StructuredTool.from_function(
    func=_submit_review,
    name="submit_review",
    description=(
        "Submit the review verdict: passed says whether it passes, suggestions "
        "give the problems and fixes."
    ),
    args_schema=_ReviewParams,
)


@dataclass(slots=True)
class ReviewResult:
    """Outcome of one reviewer session.

    Attributes:
        passed: Whether the score passes the review.
        suggestions: Concrete problems and fixes when rejected.
    """

    passed: bool
    suggestions: str = ""


class ReviewerAI:
    """An independent reviewer that opens a fresh session for every verdict.

    Attributes:
        chat_model: Model driving the review.
        context_window: Context window used for compression.
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        """Initialise the reviewer.

        Args:
            chat_model: Model driving the review.
            context_window: Context window used for compression.
        """
        self.chat_model = chat_model
        self.context_window = context_window

    def review(
        self,
        *,
        goal: str,
        genre_name: str,
        style_name: str,
        rules: frozenset[str],
        score_xml: str,
        check_summary: str,
    ) -> ReviewResult:
        """Review a score in a fresh session.

        Args:
            goal: The requirement the movement must satisfy (the Step 1 plan
                prompt, which itself embeds the user's goal).
            genre_name: Display name of the active genre.
            style_name: Display name of the active style.
            rules: Rule identifiers enforced for the work.
            score_xml: The complete current MusicXML.
            check_summary: Human-readable symbolic check summary.

        Returns:
            The review verdict.
        """
        messages: list[BaseMessage] = [
            SystemMessage(
                content=f"{_system_text(style_name, rules)}\nCurrent genre: {genre_name}."
            ),
            HumanMessage(
                content=(
                    f"[Movement requirement (Step 1 composition plan, incl. the "
                    f"user goal)]\n{goal}\n\n"
                    f"[Symbolic-layer result (already checked programmatically; "
                    f"you need not re-check)]\n{check_summary}\n\n"
                    f"Full score MusicXML:\n{score_xml}"
                )
            ),
        ]
        bound = self.chat_model.bind_tools([_REVIEW_TOOL])
        for _ in range(_REVIEW_TURNS):
            compress_messages(
                self.chat_model,
                messages,
                context_window=self.context_window,
                artifact_label="Current full MusicXML",
                artifact_provider=lambda: score_xml,
            )
            ensure_tool_responses(messages)
            response = bound.invoke(messages)
            verdict = _extract(response)
            if verdict is not None:
                return verdict
            messages.append(response)
            calls = getattr(response, "tool_calls", None) or []
            invalid = getattr(response, "invalid_tool_calls", None) or []
            if calls or invalid:
                messages.extend(
                    ToolMessage(
                        content="Call the submit_review tool with your verdict.",
                        tool_call_id=str(call.get("id", "")),
                    )
                    for call in calls
                )
                messages.extend(
                    ToolMessage(
                        content=(
                            "Could not parse the submit_review arguments; call it "
                            f"again with valid JSON: {call.get('error', '')}"
                        ),
                        tool_call_id=str(call.get("id", "")),
                    )
                    for call in invalid
                )
            else:
                messages.append(
                    HumanMessage(
                        content="Call the submit_review tool (passed and suggestions)."
                    )
                )
        return ReviewResult(False, "The reviewer could not reach a verdict.")


def _extract(response: BaseMessage) -> ReviewResult | None:
    """Extract a review verdict from a model response.

    Args:
        response: The model response.

    Returns:
        The verdict, or ``None`` when the model did not call the submit tool.
    """
    calls = getattr(response, "tool_calls", None) or []
    for call in calls:
        if str(call.get("name", "")) == "submit_review":
            args = call.get("args", {})
            return ReviewResult(
                passed=bool(args.get("passed", False)),
                suggestions=content_text(args.get("suggestions", "")),
            )
    return None
