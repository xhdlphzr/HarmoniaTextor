# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Per-movement composition and review.

Each movement is composed in a *fresh* session that receives its own Step 1
prompt and the themes of every earlier movement.  A movement is only accepted
once it contains notes and passes the symbolic checker; there is no cross
movement score injection.  As soon as a movement is composed it is reviewed by
the checker AI; a rejection is sent back to the same session and re-reviewed
before the next movement starts.  The final work is the merged concatenation of
all movements.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from harmoniatextor.agent.compression import (
    DEFAULT_CONTEXT_WINDOW,
    compress_messages,
    message_text,
)
from harmoniatextor.agent.loop import run_tool_calls
from harmoniatextor.agent.prompts import style_display_name, system_prompt
from harmoniatextor.agent.reviewer import ReviewerAI
from harmoniatextor.agent.tools import build_tools
from harmoniatextor.checker.engine import format_feedback
from harmoniatextor.domain.models import CheckReport, Movement
from harmoniatextor.i18n import translate
from harmoniatextor.score.io import from_musicxml
from harmoniatextor.service.service import CompositionService

__all__ = ["ComposeResult", "MovementComposer"]

ProgressCallback = Callable[[dict[str, Any]], None]

_MOVEMENT_TAIL = (
    "Compose this movement only. It starts from an empty score with no default "
    "voices: first create the voices you need with add_part (the instrumentation "
    "is entirely your choice), then submit themes with submit_theme and develop "
    "the melody with technique_*; once broadly shaped, fine-tune measure by "
    "measure and voice with edit(measure, voice, musicxml) (an empty fragment "
    "clears that measure). Use insert / delete to add or remove measures. Do not "
    "rewrite any other movement."
)


@dataclass(slots=True)
class ComposeResult:
    """Outcome of a full multi-movement composition.

    Attributes:
        completed: Whether every movement passed its review.
        merged_musicxml: The merged full-work MusicXML.
        review_passed: Reviewer verdict, or ``None`` when no reviewer ran.
        review_suggestions: The reviewer's combined suggestions on failure.
    """

    completed: bool
    merged_musicxml: str = ""
    review_passed: bool | None = None
    review_suggestions: str = ""


class MovementComposer:
    """Compose and review a work movement by movement.

    Attributes:
        service: Composition service.
        chat_model: Chat model driving each movement session.
        reviewer: Optional independent checker AI.
        max_steps: Maximum turns per movement session.
        max_reviews: Maximum review/fix rounds.
        context_window: Context window used to trigger compression.
    """

    def __init__(
        self,
        service: CompositionService,
        chat_model: BaseChatModel,
        *,
        reviewer: ReviewerAI | None = None,
        max_steps: int = 40,
        max_reviews: int = 5,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        """Initialise the composer.

        Args:
            service: Composition service.
            chat_model: Chat model.
            reviewer: Optional checker AI.
            max_steps: Maximum turns per movement.
            max_reviews: Maximum review rounds.
            context_window: Context window used to trigger compression.
        """
        self.service = service
        self.chat_model = chat_model
        self.reviewer = reviewer
        self.max_steps = max_steps
        self.max_reviews = max_reviews
        self.context_window = context_window

    def compose(
        self, work_id: str, goal: str, *, on_event: ProgressCallback | None = None
    ) -> ComposeResult:
        """Compose and review every movement, then merge the result.

        Each movement is composed and immediately reviewed before the next one
        starts, so a rejected movement is fixed while its session is still warm.

        Args:
            work_id: Active work.
            goal: The composition goal.
            on_event: Optional progress callback.

        Returns:
            The composition outcome.
        """
        movements = self.service.get_work(work_id).movements
        sessions: dict[str, list[BaseMessage]] = {}
        for movement in movements:
            _emit(
                on_event,
                {
                    "kind": "movement_start",
                    "movement": movement.id,
                    "name": movement.name,
                },
            )
            self._compose_movement(work_id, movement, goal, "", on_event, sessions)
            if self.reviewer is not None:
                passed, suggestions = self._review_movement(
                    work_id, movement, goal, on_event, sessions
                )
                if not passed:
                    return ComposeResult(
                        False, self._finalize(work_id), False, suggestions
                    )
            _emit(on_event, {"kind": "movement_done", "movement": movement.id})
        reviewed: bool | None = None if self.reviewer is None else True
        return ComposeResult(True, self._finalize(work_id), reviewed, "")

    def _compose_movement(
        self,
        work_id: str,
        movement: Movement,
        goal: str,
        extra: str,
        on_event: ProgressCallback | None,
        sessions: dict[str, list[BaseMessage]],
    ) -> None:
        """Compose one movement until it checks out.

        A movement starts in its own fresh session.  When a later review rejects
        it, the fix continues the *same* movement session rather than opening a
        new one.

        Args:
            work_id: Active work.
            movement: Movement to compose.
            goal: The composition goal.
            extra: Extra requirement (reviewer suggestions when fixing).
            on_event: Optional progress callback.
            sessions: Per-movement conversations, keyed by movement identifier.
        """
        work = self.service.get_work(work_id)
        style = self.service.style_for(work)
        rules = self.service.effective_rules(work)
        techniques = self.service.techniques_for(work)
        tools = build_tools(self.service, work_id, movement.id, techniques)
        mapping = {tool.name: tool for tool in tools}
        bound = self.chat_model.bind_tools(tools)
        messages = sessions.get(movement.id)
        if messages is None:
            genre = self.service.genres.get(work.genre)
            instruction = self._instruction(work_id, movement, goal)
            messages = [SystemMessage(system_prompt(genre, style, techniques, rules))]
            messages.append(HumanMessage(instruction))
            _emit(
                on_event,
                {
                    "kind": "instruction",
                    "movement": movement.id,
                    "step": 0,
                    "text": instruction,
                },
            )
        else:
            feedback = (
                "The reviewer rejected this movement; keep revising in this "
                "movement's session:\n" + extra
            )
            messages.append(HumanMessage(feedback))
            _emit(
                on_event,
                {
                    "kind": "feedback",
                    "layer": "reviewer",
                    "movement": movement.id,
                    "text": feedback,
                },
            )
        sessions[movement.id] = messages
        steps = 0
        while steps < self.max_steps:
            self._compress(work_id, movement, goal, on_event, messages)
            response = bound.invoke(messages)
            messages.append(response)
            steps += 1
            text = message_text(response)
            if text:
                _emit(
                    on_event,
                    {
                        "kind": "assistant",
                        "step": steps,
                        "text": text,
                        "ok": None,
                        "violations": [],
                    },
                )
            if run_tool_calls(messages, response, mapping, steps, on_event):
                continue
            report = self._movement_report(work_id, movement)
            if not report.ok:
                feedback = self._fix_message(report)
                _emit(
                    on_event,
                    {
                        "kind": "feedback",
                        "layer": "symbolic",
                        "movement": movement.id,
                        "text": feedback,
                    },
                )
                messages.append(HumanMessage(feedback))
                continue
            break

    def _instruction(self, work_id: str, movement: Movement, goal: str) -> str:
        """Build the fresh-session instruction for one movement.

        Args:
            work_id: Active work.
            movement: Movement to compose.
            goal: The composition goal.

        Returns:
            The instruction text.
        """
        parts = [
            f"[Overall goal]\n{goal}",
            f"[This movement's requirement ({movement.name})]\n{movement.prompt}",
        ]
        themes = self._previous_themes(work_id, movement.id)
        if themes:
            parts.append(
                "[Themes already stated (you may quote/develop/vary them, but do "
                "not change the original themes)]\n" + themes
            )
        parts.append(_MOVEMENT_TAIL)
        return "\n\n".join(parts)

    def _previous_themes(self, work_id: str, movement_id: str) -> str:
        """Render the themes of every movement before the given one.

        Args:
            work_id: Active work.
            movement_id: The movement to stop at.

        Returns:
            One line per previous theme.
        """
        lines: list[str] = []
        for movement in self.service.get_work(work_id).movements:
            if movement.id == movement_id:
                break
            themes = self.service.store.load_themes(work_id, movement.id)
            for theme_id, theme in sorted(themes.items()):
                notes = " ".join(
                    f"{note.pitch}/{note.quarter_length}" for note in theme.notes
                )
                lines.append(f"Movement {movement.name} theme {theme_id}: {notes}")
        return "\n".join(lines)

    def _movement_report(self, work_id: str, movement: Movement) -> CheckReport:
        """Check a movement's own score.

        Args:
            work_id: Active work.
            movement: Movement to check.

        Returns:
            The check report.
        """
        return self.service.check_score(
            work_id, movement.id, self.service.current_score(work_id, movement.id)
        )

    def _fix_message(self, report: CheckReport) -> str:
        """Render the follow-up message sent after a failed movement check.

        An empty movement gets the dedicated composition nudge; any other
        failure gets the full symbolic-layer feedback.

        Args:
            report: The failing check report.

        Returns:
            The human message text.
        """
        if any(item.rule_id == "empty" for item in report.violations):
            return (
                "This movement has no notes yet, so it is not complete. Submit a "
                "theme with submit_theme and develop it with technique_*."
            )
        return (
            "This movement still fails the symbolic layer; keep fixing:\n"
            + format_feedback(report)
        )

    def _compress(
        self,
        work_id: str,
        movement: Movement,
        goal: str,
        on_event: ProgressCallback | None,
        messages: list[BaseMessage],
    ) -> None:
        """Compress a movement session when it nears the context window.

        Args:
            work_id: Active work.
            movement: Movement being composed.
            goal: The composition goal.
            on_event: Optional progress callback.
            messages: Conversation messages, modified in place.
        """
        compressed = compress_messages(
            self.chat_model,
            messages,
            context_window=self.context_window,
            artifact_label="Full MusicXML of this movement",
            artifact_provider=lambda: self.service.current_musicxml(
                work_id, movement.id
            ),
            pinned_provider=lambda: self.service.latest_plan(work_id) or "",
        )
        if compressed:
            _emit(on_event, {"kind": "compress", "movement": movement.id})
            messages.append(HumanMessage(self._instruction(work_id, movement, goal)))

    def _review_movement(
        self,
        work_id: str,
        movement: Movement,
        goal: str,
        on_event: ProgressCallback | None,
        sessions: dict[str, list[BaseMessage]],
    ) -> tuple[bool, str]:
        """Review one freshly composed movement, fixing it in place.

        Args:
            work_id: Active work.
            movement: Movement to review.
            goal: The composition goal.
            on_event: Optional progress callback.
            sessions: Per-movement conversations, keyed by movement identifier.

        Returns:
            ``(passed, suggestions)``; suggestions hold the latest rejection.
        """
        assert self.reviewer is not None
        work = self.service.get_work(work_id)
        genre = self.service.genres.get(work.genre)
        suggestions = ""
        reviews = 0
        while True:
            score = self.service.current_musicxml(work_id, movement.id)
            report = self.service.check_score(
                work_id, movement.id, from_musicxml(score)
            )
            _emit(on_event, {"kind": "review_start", "movement": movement.id})
            style = self.service.style_for(work)
            rules = self.service.effective_rules(work)
            result = self.reviewer.review(
                goal=movement.prompt,
                genre_name=translate("genre." + genre.id, "en"),
                style_name=style_display_name(style),
                rules=rules,
                score_xml=score,
                check_summary=format_feedback(report),
            )
            _emit(
                on_event,
                {
                    "kind": "review",
                    "movement": movement.id,
                    "passed": result.passed,
                    "suggestions": result.suggestions,
                },
            )
            if result.passed:
                return True, ""
            suggestions = result.suggestions
            reviews += 1
            if reviews >= self.max_reviews:
                return False, suggestions
            _emit(on_event, {"kind": "movement_fix", "movement": movement.id})
            self._compose_movement(
                work_id, movement, goal, suggestions, on_event, sessions
            )

    def _finalize(self, work_id: str) -> str:
        """Merge every movement.

        Args:
            work_id: Active work.

        Returns:
            The merged MusicXML.
        """
        return self.service.merged_musicxml(work_id)


def _emit(on_event: ProgressCallback | None, event: dict[str, Any]) -> None:
    """Forward an event to the progress callback when one is set.

    Args:
        on_event: Progress callback or ``None``.
        event: Event payload.
    """
    if on_event is not None:
        on_event(event)
