# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Application service tying together storage, techniques, checker and genres.

Both the LangChain agent tools and the Flask web application call into this
service, which keeps the HTTP and LLM layers free of music logic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from music21 import stream
from pydantic import ValidationError

from harmoniatextor.checker.context import CheckerContext
from harmoniatextor.checker.engine import CheckEngine, format_feedback
from harmoniatextor.checker.profile import profile_for_rules
from harmoniatextor.domain.enums import ToolKind, WorkStatus
from harmoniatextor.domain.key import parse_key
from harmoniatextor.domain.models import (
    CheckReport,
    CheckViolation,
    Movement,
    Revision,
    StyleSelection,
    Theme,
    ThemeNote,
    Work,
)
from harmoniatextor.genres.registry import GenreRegistry
from harmoniatextor.genres.registry import build_default_registry as build_genres
from harmoniatextor.score.analysis import first_melody, measure_count
from harmoniatextor.score.io import from_musicxml, new_score, to_musicxml
from harmoniatextor.score.merge import merge_scores
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.storage.project_store import ProjectStore
from harmoniatextor.styles import StyleKit, StyleRegistry
from harmoniatextor.techniques.base import TechniqueContext, TechniqueError
from harmoniatextor.techniques.exemption import FREE_VOICE_LEADING_EXEMPT
from harmoniatextor.techniques.registry import (
    TechniqueRegistry,
)
from harmoniatextor.techniques.registry import (
    build_default_registry as build_techniques,
)

__all__ = [
    "DEFAULT_TITLE",
    "CompositionService",
    "RevisionOrigin",
    "ToolResult",
]

DEFAULT_TITLE = "未命名作品"

#: The only event that marks a generation as *completed*.  Anything else
#: (started but never tagged, or failed with an error) counts as interrupted.
_GENERATION_EVENTS = frozenset(
    {
        "generation_started",
        "generation_finished",
        "generation_failed",
        "generation_interrupted",
    }
)
_GENERATION_COMPLETED = "generation_finished"

#: Technique id whose revision marks a scoped voice-leading exemption.
_FREE_VOICE_LEADING = "free_voice_leading"


def _in_exempt_scope(
    violation: CheckViolation, scopes: list[tuple[str, int, int]]
) -> bool:
    """Return whether a violation falls inside an exemption scope.

    Args:
        violation: The violation to test.
        scopes: ``(voice, start, end)`` scopes.

    Returns:
        ``True`` when the violation is waived.
    """
    for voice, start, end in scopes:
        if start <= violation.measure <= end and voice in {
            violation.voice_a,
            violation.voice_b,
        }:
            return True
    return False


def _filter_exemptions(
    report: CheckReport, scopes: list[tuple[str, int, int]]
) -> CheckReport:
    """Drop violations waived by free-voice-leading exemptions.

    Args:
        report: The raw check report.
        scopes: ``(voice, start, end)`` scopes.

    Returns:
        The report without waived violations.
    """
    if not scopes:
        return report
    kept = [
        violation
        for violation in report.violations
        if not (
            violation.rule_id in FREE_VOICE_LEADING_EXEMPT
            and _in_exempt_scope(violation, scopes)
        )
    ]
    if len(kept) == len(report.violations):
        return report
    return CheckReport(violations=kept)


def _revision_seq(revision_id: str) -> int | None:
    """Return the sequence number encoded in a revision id.

    Args:
        revision_id: Revision identifier such as ``"r-m01-3"``.

    Returns:
        The sequence number, or ``None`` when the id is empty or malformed.
    """
    if not revision_id:
        return None
    _, _, tail = revision_id.rpartition("-")
    try:
        return int(tail)
    except ValueError:
        return None


@dataclass(slots=True)
class ToolResult:
    """Result of a composition tool invocation.

    Attributes:
        ok: Whether the operation passed the symbolic checker.
        full_musicxml: The full score after the operation, when applicable.
        theme_id: Newly assigned theme number for submissions.
        error_code: Machine-readable error code on failure.
        message: Human/LLM readable message.
        report: The check report.
        warnings: Non-fatal technique warnings.
        movement_id: Movement touched by a planning operation.
    """

    ok: bool
    full_musicxml: str | None = None
    theme_id: int | None = None
    error_code: str | None = None
    message: str = ""
    report: CheckReport | None = None
    warnings: list[str] = field(default_factory=list)
    movement_id: str | None = None


@dataclass(slots=True)
class RevisionOrigin:
    """Describes what produced a revision.

    Attributes:
        source: What produced the revision.
        technique: Technique identifier when applicable.
        params: Normalised parameters when applicable.
    """

    source: ToolKind
    technique: str | None = None
    params: dict[str, Any] | None = None


def _now() -> str:
    """Return the current UTC timestamp in ISO format.

    Returns:
        The resulting text.
    """
    return datetime.now(UTC).isoformat()


class CompositionService:
    """High-level operations over works and movements.

    Attributes:
        store: Project storage.
        techniques: Technique registry.
        genres: Genre registry.
        engine: Checker engine.
    """

    def __init__(
        self,
        store: ProjectStore,
        techniques: TechniqueRegistry | None = None,
        genres: GenreRegistry | None = None,
        engine: CheckEngine | None = None,
        styles: StyleRegistry | None = None,
    ) -> None:
        """Initialise the service.

        Args:
            store: Project storage.
            techniques: Technique registry; defaults to the 36 built-ins.
            genres: Genre registry; defaults to the four built-ins.
            engine: Checker engine; defaults to a fresh engine.
            styles: Style registry; defaults to the built-in plus stored kits.
        """
        self.store = store
        self.techniques = techniques if techniques is not None else build_techniques()
        self.genres = genres if genres is not None else build_genres()
        self.engine = engine if engine is not None else CheckEngine()
        self.styles = styles if styles is not None else StyleRegistry()

    def style_for(self, work: Work) -> StyleKit:
        """Return the effective style kit of a work.

        Args:
            work: The work.

        Returns:
            The work's frozen snapshot, or the default built-in kit for legacy
            works created before styles existed.
        """
        snapshot = work.style
        if snapshot is None:
            return self.styles.resolve(None)
        return StyleKit(
            id=snapshot.id,
            name=snapshot.name,
            brief=snapshot.brief,
            rules=snapshot.rules,
            techniques=snapshot.techniques,
        )

    def effective_rules(self, work: Work) -> frozenset[str]:
        """Return the rule identifiers enabled for a work.

        Args:
            work: The work.

        Returns:
            The style's enabled rule identifiers.  Free-voice-leading
            exemptions do not disable rules; they filter violations at check
            time (see :meth:`_check`).
        """
        return self.style_for(work).rules

    def techniques_for(self, work: Work) -> TechniqueRegistry:
        """Return a technique registry limited to the work's style.

        Args:
            work: The work.

        Returns:
            A registry holding only the style's techniques, in canonical order.
        """
        allowed = self.style_for(work).techniques
        registry = TechniqueRegistry()
        for technique in self.techniques.all():
            if technique.id in allowed:
                registry.register(technique)
        return registry

    def create_work(
        self,
        title: str,
        genre: str,
        tonic: str = "C",
        *,
        style: str | None = None,
        with_movements: bool = True,
    ) -> Work:
        """Create a new work with a movement skeleton.

        Args:
            title: Work title.
            genre: Genre identifier.
            tonic: Initial tonic key; the composer agent replaces it when it
                submits the first theme with a chosen key.
            style: Style kit identifier; defaults to the built-in default kit.
            with_movements: Whether to pre-populate the genre's movement
                skeleton.  The architect adds movements itself.

        Returns:
            The created work.
        """
        genre_obj = self.genres.get(genre)
        kit = self.styles.resolve(style)
        work_id = f"w-{uuid.uuid4().hex[:8]}"
        timestamp = _now()
        work = Work(
            id=work_id,
            title=title,
            genre=genre,
            tonic=parse_key(tonic),
            movements=genre_obj.initialize_work(work_id, tonic)
            if with_movements
            else [],
            status=WorkStatus.DRAFT,
            style=StyleSelection(
                id=kit.id,
                name=kit.name,
                brief=kit.brief,
                rules=kit.rules,
                techniques=kit.techniques,
            ),
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.store.save_work(work)
        self.store.append_journal(
            work_id,
            {"event": "genre_init", "genre": genre, "tonic": tonic, "style": kit.id},
        )
        return work

    def get_work(self, work_id: str) -> Work:
        """Load a work.

        Args:
            work_id: Work identifier.

        Returns:
            The work.
        """
        return self.store.load_work(work_id)

    def set_title(self, work_id: str, title: str) -> ToolResult:
        """Set the title of a work.

        Args:
            work_id: Work identifier.
            title: New title.

        Returns:
            A tool result confirming the change.
        """
        work = self.store.load_work(work_id)
        work.title = title
        work.updated_at = _now()
        self.store.save_work(work)
        self.store.append_journal(work_id, {"event": "title", "title": title})
        return ToolResult(True, message=f"标题已设为:{title}")

    def ensure_title(self, work_id: str) -> str:
        """Fall back to the genre name when no title was chosen.

        Args:
            work_id: Work identifier.

        Returns:
            The resulting title.
        """
        work = self.store.load_work(work_id)
        if work.title.strip() and work.title != DEFAULT_TITLE:
            return work.title
        genre = self.genres.get(work.genre)
        self.set_title(work_id, genre.display_name)
        return genre.display_name

    def list_works(self) -> list[str]:
        """List stored work identifiers.

        Returns:
            Sorted identifiers.
        """
        return self.store.list_works()

    def get_movement(self, work: Work, movement_id: str) -> Movement:
        """Find a movement within a work.

        Args:
            work: Owning work.
            movement_id: Movement identifier.

        Returns:
            The movement.

        Raises:
            KeyError: When the movement does not exist.
        """
        for movement in work.movements:
            if movement.id == movement_id:
                return movement
        raise KeyError(movement_id)

    def add_movement(self, work_id: str, name: str | None = None) -> ToolResult:
        """Append a planned movement to a work.

        Args:
            work_id: Active work.
            name: Optional display name.

        Returns:
            A tool result carrying the new movement identifier and number.
        """
        work = self.store.load_work(work_id)
        index = len(work.movements) + 1
        movement_id = f"m{index:02d}"
        movement = Movement(
            id=movement_id,
            work_id=work_id,
            name=name or f"第 {index} 乐章",
            time_signature="4/4",
            key=work.tonic,
            tempo=96,
            voice_profile="four_part",
        )
        work.movements.append(movement)
        work.updated_at = _now()
        self.store.save_work(work)
        self.store.append_journal(
            work_id, {"event": "movement_added", "movement": movement_id}
        )
        return ToolResult(
            True,
            movement_id=movement_id,
            message=f"已新增乐章,编号 {index}。",
        )

    def set_movement_prompt(
        self, work_id: str, movement_id: str, prompt: str
    ) -> ToolResult:
        """Record the Step 1 prompt of a movement.

        Args:
            work_id: Active work.
            movement_id: Owning movement.
            prompt: The movement's creation requirement.

        Returns:
            A tool result confirming the update.
        """
        work = self.store.load_work(work_id)
        try:
            movement = self.get_movement(work, movement_id)
        except KeyError:
            return ToolResult(
                False, error_code="BAD_PARAM", message=f"乐章不存在:{movement_id}"
            )
        movement.prompt = prompt
        work.updated_at = _now()
        self.store.save_work(work)
        self.store.append_journal(
            work_id, {"event": "movement_prompt", "movement": movement_id}
        )
        return ToolResult(
            True, movement_id=movement_id, message="已记录该乐章的创作要求。"
        )

    def missing_movement_prompts(self, work_id: str) -> list[Movement]:
        """Return the movements that still lack a Step 1 prompt.

        Args:
            work_id: Active work.

        Returns:
            Movements whose prompt is empty.
        """
        return [
            item
            for item in self.store.load_work(work_id).movements
            if not item.prompt.strip()
        ]

    def merged_musicxml(self, work_id: str) -> str:
        """Merge every composed movement into one MusicXML document.

        Args:
            work_id: Active work.

        Returns:
            The merged MusicXML, or an empty string when nothing is composed.
        """
        work = self.store.load_work(work_id)
        xmls = [
            self.current_musicxml(work_id, movement.id)
            for movement in work.movements
            if self._has_notes(work_id, movement.id)
        ]
        if not xmls:
            return ""
        return merge_scores(xmls)

    def _has_notes(self, work_id: str, movement_id: str) -> bool:
        """Return whether a movement's current score contains any note.

        Args:
            work_id: Active work.
            movement_id: Movement identifier.

        Returns:
            ``True`` when the score has at least one note.
        """
        return bool(list(self.current_score(work_id, movement_id).recurse().notes))

    def _owning_movement(self, work: Work, target_id: str) -> Movement:
        """Resolve a composition target to its movement.

        Args:
            work: Owning work.
            target_id: Movement identifier.

        Returns:
            The movement.

        Raises:
            KeyError: When the target is unknown.
        """
        for movement in work.movements:
            if movement.id == target_id:
                return movement
        raise KeyError(target_id)

    def _themes_for(self, work: Work, target_id: str) -> dict[int, Theme]:
        """Build the theme registry visible to a movement.

        A movement may reference the themes of every earlier movement; those are
        re-anchored to the movement's own first measure so techniques write at
        its start.

        Args:
            work: Owning work.
            target_id: Movement identifier.

        Returns:
            A theme registry keyed by theme number.
        """
        order = [movement.id for movement in work.movements]
        registry: dict[int, Theme] = {}
        for movement_id in order[: order.index(target_id) + 1]:
            for theme_id, theme in self.store.load_themes(work.id, movement_id).items():
                if movement_id == target_id:
                    registry[theme_id] = theme
                else:
                    registry[theme_id] = replace(theme, start_measure=1)
        return registry

    def _initial_score(self, movement: Movement) -> stream.Score:
        """Build the empty starting score of a composition target.

        The score intentionally has no parts: the composer AI creates every
        voice itself with ``add_part`` (or by naming a voice in ``submit_theme``
        / ``edit``).

        Args:
            movement: Owning movement.

        Returns:
            A fresh, part-less score.
        """
        return new_score(
            key=movement.key.music21_name,
            time_signature=movement.time_signature,
            tempo_bpm=movement.tempo,
            voices=[],
        )

    def _editor(self, score: stream.Score, movement: Movement) -> ScoreEditor:
        """Build a score editor seeded with a movement's musical settings.

        Args:
            score: Score to edit.
            movement: Movement providing key, meter and tempo.

        Returns:
            A configured editor.
        """
        return ScoreEditor(
            score,
            key=movement.key.music21_name,
            time_signature=movement.time_signature,
            tempo_bpm=movement.tempo,
        )

    def current_score(self, work_id: str, target_id: str) -> stream.Score:
        """Load the canonical score of a movement.

        Args:
            work_id: Work identifier.
            target_id: Movement identifier.

        Returns:
            The current score.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, target_id)
        xml = self.current_musicxml(work_id, target_id)
        if not xml:
            return self._initial_score(movement)
        return from_musicxml(xml)

    def current_musicxml(self, work_id: str, target_id: str) -> str:
        """Load the canonical score of a movement as MusicXML.

        Args:
            work_id: Work identifier.
            target_id: Movement identifier.

        Returns:
            The current MusicXML text, or an empty string while the movement has
            no voices yet.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, target_id)
        meta = self.store.load_revision_meta(work_id, target_id)
        if not meta:
            score = self._initial_score(movement)
            return to_musicxml(score) if score.parts else ""
        match = next(
            (item for item in meta if str(item["id"]) == movement.canonical_revision),
            None,
        )
        seq = int(match["seq"]) if match is not None else int(meta[-1]["seq"])
        return self.store.load_revision_xml(work_id, target_id, seq)

    def check(
        self, work_id: str, target_id: str, *, complete: bool = False
    ) -> CheckReport:
        """Run the symbolic checker on the current score of a target.

        Args:
            work_id: Work identifier.
            target_id: Movement identifier.
            complete: Whether to apply structural-end expectations.

        Returns:
            The check report.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, target_id)
        return self._check(
            work, movement, self.current_score(work_id, target_id), complete=complete
        )

    def check_score(
        self,
        work_id: str,
        target_id: str,
        score: stream.Score,
        *,
        complete: bool = False,
    ) -> CheckReport:
        """Run the symbolic checker on an arbitrary score of a target.

        Args:
            work_id: Work identifier.
            target_id: Movement identifier providing the context.
            score: Score to check.
            complete: Whether to apply structural-end expectations.

        Returns:
            The check report.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, target_id)
        return self._check(work, movement, score, complete=complete)

    def finalize(self, work_id: str, movement_id: str) -> ToolResult:
        """Finalise a movement, applying structural-end checks.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.

        Returns:
            A tool result indicating whether the movement may be finalised.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        score = self.current_score(work_id, movement_id)
        report = self._check(work, movement, score, complete=True)
        if not report.ok:
            return ToolResult(False, message=format_feedback(report), report=report)
        self.store.append_journal(
            work_id, {"event": "finalize", "movement": movement_id}
        )
        self.store.append_journal(
            work_id, {"event": "movement_finalized", "movement": movement_id}
        )
        finalized = self._finalized_movements(work_id)
        if all(item.id in finalized for item in work.movements):
            work.status = WorkStatus.FINAL
        work.updated_at = _now()
        self.store.save_work(work)
        return ToolResult(True, message="乐章已定稿。", report=report)

    def _finalized_movements(self, work_id: str) -> set[str]:
        """Return the movements of a work that have been finalised.

        Args:
            work_id: Work identifier.

        Returns:
            Movement identifiers with a ``movement_finalized`` journal event.
        """
        return {
            str(event.get("movement"))
            for event in self.store.load_journal(work_id)
            if event.get("event") == "movement_finalized"
        }

    def submit_theme(
        self,
        work_id: str,
        movement_id: str,
        musicxml: str,
        *,
        voice: str | None = None,
        instrument: str | None = None,
        key: str | None = None,
        check: bool = True,
    ) -> ToolResult:
        """Insert a newly submitted theme into the current score.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            musicxml: MusicXML of the theme melody.
            voice: Target voice slot; defaults to the fragment's first part.
            instrument: Instrument to assign to the target voice slot.
            key: Absolute key and mode chosen by the composer; ``None`` keeps the
                current key.
            check: Whether to run the symbolic checker immediately.  The composer
                agent defers checking to the moment it hands the score over.

        Returns:
            A tool result carrying the assigned theme number and full score.
        """
        spec = None
        if key:
            try:
                spec = parse_key(key)
            except ValueError:
                return ToolResult(
                    False, error_code="BAD_PARAM", message=f"无效的调式:{key}"
                )
        try:
            theme_score = from_musicxml(musicxml)
        except Exception:  # noqa: BLE001 - reject any unparseable MusicXML
            return ToolResult(
                False, error_code="BAD_PARAM", message="主题 MusicXML 解析失败。"
            )
        source_voice, notes = first_melody(theme_score)
        if not notes:
            return ToolResult(
                False, error_code="BAD_PARAM", message="主题不包含任何音符。"
            )
        target_voice = voice or source_voice
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        score = self.current_score(work_id, movement_id)
        if spec is not None:
            work.tonic = spec
            movement.key = spec
        has_notes = bool(list(score.recurse().notes))
        start = measure_count(score) + 1 if has_notes else 1
        editor = self._editor(score, movement)
        if spec is not None:
            editor.set_key(spec.music21_name)
        if instrument:
            editor.set_instrument(target_voice, instrument)
        editor.write_line(target_voice, start, notes)
        report = self._check(work, movement, score) if check else None
        params: dict[str, Any] = {"voice": target_voice}
        if instrument:
            params["instrument"] = instrument
        if spec is not None:
            params["key"] = spec.raw
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            score,
            RevisionOrigin(ToolKind.SUBMIT, params=params),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(
                False,
                message=format_feedback(report),
                report=report,
            )
        all_themes = self.store.load_all_themes(work_id)
        theme_id = max((item_id for _, item_id in all_themes), default=0) + 1
        themes = self.store.load_themes(work_id, movement_id)
        themes[theme_id] = Theme(
            id=theme_id,
            movement_id=movement_id,
            voice=target_voice,
            start_measure=start,
            notes=notes,
            created_revision=revision.id,
        )
        self.store.save_themes(work_id, movement_id, themes)
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            theme_id=theme_id,
            message=format_feedback(report) if report is not None else "主题已提交。",
            report=report,
        )

    def add_part(
        self,
        work_id: str,
        movement_id: str,
        voice: str,
        instrument: str | None = None,
        *,
        check: bool = True,
    ) -> ToolResult:
        """Add a new instrumental part to a movement.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            voice: Voice slot name for the new part.
            instrument: Instrument to assign; defaults to the voice mapping.
            check: Whether to run the symbolic checker immediately.

        Returns:
            A tool result carrying the full score when the check passes.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        score = self.current_score(work_id, movement_id)
        editor = self._editor(score, movement)
        if editor.get_part(voice) is not None:
            return ToolResult(
                False, error_code="BAD_PARAM", message=f"声部已存在:{voice}"
            )
        editor.get_part(voice, create=True)
        if instrument:
            editor.set_instrument(voice, instrument)
        report = self._check(work, movement, score) if check else None
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            score,
            RevisionOrigin(
                ToolKind.PART, params={"voice": voice, "instrument": instrument}
            ),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(False, message=format_feedback(report), report=report)
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            message=format_feedback(report) if report is not None else "声部已添加。",
            report=report,
        )

    def remove_part(
        self, work_id: str, movement_id: str, voice: str, *, check: bool = True
    ) -> ToolResult:
        """Remove a voice from a movement.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            voice: Voice slot to remove.
            check: Whether to run the symbolic checker immediately.

        Returns:
            A tool result carrying the full score.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        score = self.current_score(work_id, movement_id)
        editor = self._editor(score, movement)
        if not editor.remove_part(voice):
            return ToolResult(
                False, error_code="BAD_PARAM", message=f"声部不存在:{voice}"
            )
        report = self._check(work, movement, score) if check else None
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            score,
            RevisionOrigin(ToolKind.PART, params={"voice": voice, "action": "remove"}),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(False, message=format_feedback(report), report=report)
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            message=format_feedback(report) if report is not None else "声部已删除。",
            report=report,
        )

    def set_tempo(
        self, work_id: str, movement_id: str, bpm: int, *, check: bool = True
    ) -> ToolResult:
        """Change the tempo of a movement.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            bpm: New tempo in quarter notes per minute.
            check: Whether to run the symbolic checker immediately.

        Returns:
            A tool result carrying the full score.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        movement.tempo = bpm
        score = self.current_score(work_id, movement_id)
        editor = self._editor(score, movement)
        editor.set_tempo(bpm)
        report = self._check(work, movement, score) if check else None
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            score,
            RevisionOrigin(ToolKind.TEMPO, params={"bpm": bpm}),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(False, message=format_feedback(report), report=report)
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            message=format_feedback(report)
            if report is not None
            else f"速度已改为 {bpm}。",
            report=report,
        )

    def set_time_signature(
        self, work_id: str, movement_id: str, time_signature: str, *, check: bool = True
    ) -> ToolResult:
        """Change the time signature of a movement.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            time_signature: New time signature such as ``"3/4"``.
            check: Whether to run the symbolic checker immediately.

        Returns:
            A tool result carrying the full score.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        movement.time_signature = time_signature
        score = self.current_score(work_id, movement_id)
        editor = self._editor(score, movement)
        editor.set_time_signature(time_signature)
        report = self._check(work, movement, score) if check else None
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            score,
            RevisionOrigin(ToolKind.METER, params={"time_signature": time_signature}),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(False, message=format_feedback(report), report=report)
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            message=format_feedback(report)
            if report is not None
            else f"拍号已改为 {time_signature}。",
            report=report,
        )

    def annotate(
        self,
        work_id: str,
        movement_id: str,
        measure: int,
        voice: str,
        mark: str,
        value: str = "",
        *,
        check: bool = True,
    ) -> ToolResult:
        """Add an expressive mark (dynamic, slur, pedal, ...) to a measure.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            measure: One-based measure number.
            voice: Voice slot to annotate.
            mark: Mark kind, e.g. ``"dynamic"`` or ``"slur"``.
            value: Mark-specific value (dynamic name, text, or tempo BPM).
            check: Whether to run the symbolic checker immediately.

        Returns:
            A tool result carrying the full score.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        score = self.current_score(work_id, movement_id)
        editor = self._editor(score, movement)
        if editor.get_part(voice) is None:
            return ToolResult(
                False, error_code="BAD_PARAM", message=f"声部不存在:{voice}"
            )
        if not editor.annotate(voice, measure, mark, value):
            return ToolResult(
                False,
                error_code="BAD_PARAM",
                message=f"无法在小节 {measure} 的 {voice} 上添加记号:{mark}。",
            )
        report = self._check(work, movement, score) if check else None
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            score,
            RevisionOrigin(
                ToolKind.MARK, params={"measure": measure, "voice": voice, "mark": mark}
            ),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(False, message=format_feedback(report), report=report)
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            message=format_feedback(report)
            if report is not None
            else f"已添加记号:{mark}。",
            report=report,
        )

    def apply_technique(
        self,
        work_id: str,
        movement_id: str,
        technique_id: str,
        params: dict[str, Any],
        *,
        check: bool = True,
    ) -> ToolResult:
        """Apply a technique to the current score.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            technique_id: Technique identifier.
            params: Raw technique parameters.
            check: Whether to run the symbolic checker immediately.

        Returns:
            A tool result carrying the full score when the check passes.
        """
        try:
            technique = self.techniques.get(technique_id)
        except KeyError:
            return ToolResult(
                False, error_code="BAD_PARAM", message=f"未知技法:{technique_id}"
            )
        try:
            parsed = technique.params_model.model_validate(params)
        except ValidationError as exc:
            return ToolResult(False, error_code="BAD_PARAM", message=str(exc))
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        score = self.current_score(work_id, movement_id)
        themes = self._themes_for(work, movement_id)
        ctx = TechniqueContext(
            score=score,
            themes=themes,
            genre=work.genre,
            voice_profile=movement.voice_profile,
        )
        try:
            outcome = technique.apply(ctx, parsed)
        except TechniqueError as exc:
            return ToolResult(False, error_code=exc.code, message=exc.message)
        report: CheckReport | None = None
        if check:
            scopes = self._exempt_scopes(work, movement)
            if technique.exempts:
                measure_range = parsed.measure_range
                scopes = [
                    *scopes,
                    (str(parsed.voice), measure_range.start, measure_range.end),
                ]
            report = self._check(work, movement, outcome.score, scopes=scopes)
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            outcome.score,
            RevisionOrigin(ToolKind.TECHNIQUE, technique_id, parsed.model_dump()),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(
                False,
                message=format_feedback(report),
                report=report,
                warnings=outcome.warnings,
            )
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            message=format_feedback(report) if report is not None else "技法已应用。",
            report=report,
            warnings=outcome.warnings,
        )

    def edit_measure(
        self,
        work_id: str,
        movement_id: str,
        measure: int,
        voice: str,
        musicxml: str,
        *,
        check: bool = True,
    ) -> ToolResult:
        """Replace or clear one measure of one voice.

        An empty ``musicxml`` fragment clears the measure's sounding and resting
        material instead of failing, so a measure can be emptied on purpose.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            measure: One-based measure number to replace.
            voice: Voice slot to replace.
            musicxml: A MusicXML fragment whose first melody replaces the
                measure; empty clears the measure.
            check: Whether to run the symbolic checker immediately.

        Returns:
            A tool result carrying the full score.
        """
        notes: list[ThemeNote] = []
        if musicxml.strip():
            try:
                fragment = from_musicxml(musicxml)
            except Exception:  # noqa: BLE001 - reject any unparseable MusicXML
                return ToolResult(
                    False,
                    error_code="BAD_PARAM",
                    message="编辑片段 MusicXML 解析失败。",
                )
            _source, notes = first_melody(fragment)
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        score = self.current_score(work_id, movement_id)
        editor = self._editor(score, movement)
        if not notes and editor.get_part(voice) is None:
            return ToolResult(
                False, error_code="BAD_PARAM", message=f"声部不存在:{voice}"
            )
        editor.clear_measure_range(voice, measure, measure)
        if notes:
            editor.write_line(voice, measure, notes)
        report = self._check(work, movement, score) if check else None
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            score,
            RevisionOrigin(ToolKind.EDIT, params={"measure": measure, "voice": voice}),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(False, message=format_feedback(report), report=report)
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            message=format_feedback(report) if report is not None else "该小节已更新。",
            report=report,
        )

    def insert_measure(
        self,
        work_id: str,
        movement_id: str,
        measure: int,
        *,
        voice: str | None = None,
        musicxml: str = "",
        check: bool = True,
    ) -> ToolResult:
        """Insert a new measure in every voice, optionally filled with notes.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            measure: One-based position of the new measure.
            voice: Voice that receives the fragment; required with ``musicxml``.
            musicxml: Optional MusicXML fragment for the new measure.
            check: Whether to run the symbolic checker immediately.

        Returns:
            A tool result carrying the full score.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        score = self.current_score(work_id, movement_id)
        editor = self._editor(score, movement)
        if not score.parts:
            return ToolResult(
                False,
                error_code="BAD_PARAM",
                message="本乐章还没有任何声部,请先用 add_part。",
            )
        editor.insert_measure(measure)
        if musicxml.strip():
            if not voice:
                return ToolResult(
                    False,
                    error_code="BAD_PARAM",
                    message="插入带音符的小节时必须指定 voice。",
                )
            try:
                fragment = from_musicxml(musicxml)
            except Exception:  # noqa: BLE001 - reject any unparseable MusicXML
                return ToolResult(
                    False, error_code="BAD_PARAM", message="插入片段解析失败。"
                )
            _source, notes = first_melody(fragment)
            if notes:
                editor.clear_measure_range(voice, measure, measure)
                editor.write_line(voice, measure, notes)
        report = self._check(work, movement, score) if check else None
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            score,
            RevisionOrigin(
                ToolKind.EDIT,
                params={"measure": measure, "voice": voice or "", "action": "insert"},
            ),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(False, message=format_feedback(report), report=report)
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            message=format_feedback(report) if report is not None else "已插入新小节。",
            report=report,
        )

    def delete_measure(
        self, work_id: str, movement_id: str, measure: int, *, check: bool = True
    ) -> ToolResult:
        """Delete a measure from every voice, shifting later measures.

        Args:
            work_id: Active work.
            movement_id: Active movement.
            measure: One-based measure number to remove.
            check: Whether to run the symbolic checker immediately.

        Returns:
            A tool result carrying the full score.
        """
        work = self.store.load_work(work_id)
        movement = self._owning_movement(work, movement_id)
        score = self.current_score(work_id, movement_id)
        editor = self._editor(score, movement)
        if not editor.delete_measure(measure):
            return ToolResult(
                False, error_code="BAD_PARAM", message=f"小节不存在:{measure}"
            )
        report = self._check(work, movement, score) if check else None
        revision = self._save_revision(
            work,
            movement,
            movement_id,
            score,
            RevisionOrigin(
                ToolKind.EDIT, params={"measure": measure, "action": "delete"}
            ),
            report,
        )
        if report is not None and not report.ok:
            return ToolResult(False, message=format_feedback(report), report=report)
        return ToolResult(
            True,
            full_musicxml=revision.full_xml,
            message=format_feedback(report) if report is not None else "已删除该小节。",
            report=report,
        )

    def rollback(self, work_id: str, movement_id: str, seq: int) -> ToolResult:
        """Roll the canonical score back to an earlier revision.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            seq: Revision sequence number to restore.

        Returns:
            A tool result carrying the restored full score.
        """
        work = self.store.load_work(work_id)
        movement = self.get_movement(work, movement_id)
        meta = self.store.load_revision_meta(work_id, movement_id)
        match = next((item for item in meta if int(item["seq"]) == seq), None)
        if match is None:
            return ToolResult(
                False, error_code="BAD_PARAM", message=f"版本 {seq} 不存在。"
            )
        movement.canonical_revision = str(match["id"])
        work.updated_at = _now()
        self.store.save_work(work)
        self.store.append_journal(
            work_id,
            {"event": "rollback", "movement": movement_id, "seq": seq},
        )
        return ToolResult(
            True,
            full_musicxml=self.store.load_revision_xml(work_id, movement_id, seq),
            message=f"已回退到版本 {seq}。",
        )

    def record_audit(self, work_id: str, movement_id: str, note: str) -> None:
        """Record a human audition note.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            note: The audition note.
        """
        work = self.store.load_work(work_id)
        work.status = WorkStatus.REVISING
        work.updated_at = _now()
        self.store.save_work(work)
        self.store.append_journal(
            work_id,
            {"event": "audit_note", "movement": movement_id, "note": note},
        )

    def record_review(
        self, work_id: str, movement_id: str, passed: bool, suggestions: str
    ) -> None:
        """Record a reviewer AI verdict in the journal.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            passed: Whether the reviewer passed the score.
            suggestions: The reviewer's suggestions.
        """
        self.store.append_journal(
            work_id,
            {
                "event": "review",
                "movement": movement_id,
                "passed": passed,
                "suggestions": suggestions,
            },
        )

    def latest_review(self, work_id: str) -> dict[str, Any] | None:
        """Return the most recent reviewer verdict for a work.

        Args:
            work_id: Work identifier.

        Returns:
            The latest review event, or ``None`` when there is none.
        """
        for event in reversed(self.store.load_journal(work_id)):
            if event.get("event") == "review":
                return event
        return None

    def record_plan(self, work_id: str, movement_id: str, text: str) -> None:
        """Record a Step 1 creation plan in the journal.

        Args:
            work_id: Work identifier.
            movement_id: Movement identifier.
            text: The plan text.
        """
        self.store.append_journal(
            work_id,
            {"event": "plan", "movement": movement_id, "text": text},
        )

    def latest_plan(self, work_id: str) -> str | None:
        """Return the most recent creation plan for a work.

        Args:
            work_id: Work identifier.

        Returns:
            The latest plan text, or ``None`` when there is none.
        """
        for event in reversed(self.store.load_journal(work_id)):
            if event.get("event") == "plan":
                return str(event.get("text", ""))
        return None

    def start_generation(self, work_id: str, prompt: str) -> None:
        """Record that an automatic generation run started.

        Args:
            work_id: Work identifier.
            prompt: The composition goal the run was started with.
        """
        self.store.append_journal(
            work_id, {"event": "generation_started", "prompt": prompt}
        )

    def finish_generation(self, work_id: str, completed: bool) -> None:
        """Tag a generation run as completed, or record that it failed.

        Only a run that reaches its end without crashing is tagged with
        ``generation_finished``.  A run stopped by an error is recorded as
        ``generation_failed`` so that :meth:`interrupt_stale_generations` treats
        it exactly like a run that was killed (both are un-tagged).

        Args:
            work_id: Work identifier.
            completed: Whether the run reached its end (``False`` when it
                crashed before finishing).
        """
        event = _GENERATION_COMPLETED if completed else "generation_failed"
        self.store.append_journal(work_id, {"event": event, "completed": completed})

    def latest_generation_state(self, work_id: str) -> str | None:
        """Return the latest generation lifecycle event of a work.

        Args:
            work_id: Work identifier.

        Returns:
            ``generation_started``, ``generation_finished`` or
            ``generation_interrupted``, or ``None`` when the work was never
            generated automatically.
        """
        for event in reversed(self.store.load_journal(work_id)):
            name = str(event.get("event", ""))
            if name in _GENERATION_EVENTS:
                return name
        return None

    def interrupt_stale_generations(self) -> int:
        """Mark generation runs that never finished, for example after a crash.

        Partially generated works are already persisted revision by revision;
        this only adds an explicit ``generation_interrupted`` marker so an
        interrupted run stays visible in the work history instead of looking
        like an idle draft.

        Returns:
            The number of works newly marked as interrupted.
        """
        marked = 0
        for work_id in self.list_works():
            state = self.latest_generation_state(work_id)
            if state is not None and state not in (
                _GENERATION_COMPLETED,
                "generation_interrupted",
            ):
                self.store.append_journal(work_id, {"event": "generation_interrupted"})
                marked += 1
        return marked

    def _exempt_scopes(
        self, work: Work, movement: Movement
    ) -> list[tuple[str, int, int]]:
        """Return the free-voice-leading exemption scopes of a movement.

        Args:
            work: Owning work.
            movement: Movement under check.

        Returns:
            ``(voice, start, end)`` tuples collected from the movement's
            revisions.
        """
        canonical = _revision_seq(movement.canonical_revision)
        if canonical is None:
            return []
        scopes: list[tuple[str, int, int]] = []
        for meta in self.store.load_revision_meta(work.id, movement.id):
            if meta.get("technique") != _FREE_VOICE_LEADING:
                continue
            # Only exemptions on the current canonical chain apply, so rolling
            # back past the exemption restores the strict rules.
            if int(meta.get("seq", 0)) > canonical:
                continue
            params = meta.get("params") or {}
            voice = params.get("voice")
            measure_range = params.get("measure_range") or {}
            if voice and "start" in measure_range and "end" in measure_range:
                scopes.append(
                    (str(voice), int(measure_range["start"]), int(measure_range["end"]))
                )
        return scopes

    def _check(
        self,
        work: Work,
        movement: Movement,
        score: stream.Score,
        *,
        complete: bool = False,
        scopes: list[tuple[str, int, int]] | None = None,
    ) -> CheckReport:
        """Run the checker with the work's style rules.

        Args:
            work: Owning work.
            movement: Movement under check.
            score: Score to check.
            complete: Whether to apply structural-end expectations.
            scopes: Free-voice-leading exemption scopes; computed from history
                when omitted.

        Returns:
            The check report.
        """
        genre = self.genres.get(work.genre)
        rules = self.effective_rules(work)
        context: CheckerContext = genre.checker_context(
            movement.key.raw,
            measure_count(score),
            complete=complete,
        )
        context.enforce_voice_count = "voices" in rules
        report = self.engine.run(score, context, profile=profile_for_rules(rules))
        active = scopes if scopes is not None else self._exempt_scopes(work, movement)
        return _filter_exemptions(report, active)

    def _save_revision(
        self,
        work: Work,
        movement: Movement,
        target_id: str,
        score: stream.Score,
        origin: RevisionOrigin,
        report: CheckReport | None,
    ) -> Revision:
        """Persist a new revision and update the canonical pointer on success.

        Args:
            work: Owning work.
            movement: Owning movement.
            target_id: Movement identifier being revised.
            score: Resulting score.
            origin: What produced the revision.
            report: Check report, or ``None`` when the checker was deferred.

        Returns:
            The stored revision.
        """
        meta = self.store.load_revision_meta(work.id, target_id)
        seq = len(meta)
        ok = report.ok if report is not None else True
        parent_seq = _revision_seq(movement.canonical_revision)
        revision = Revision(
            id=f"r-{target_id}-{seq}",
            seq=seq,
            movement_id=target_id,
            full_xml=to_musicxml(score),
            source=origin.source,
            technique=origin.technique,
            params=origin.params,
            check=report,
            ok=ok,
            parent_seq=parent_seq,
        )
        self.store.save_revision_for(work.id, revision)
        work.updated_at = _now()
        if ok:
            movement.canonical_revision = revision.id
            if work.status is WorkStatus.DRAFT:
                work.status = WorkStatus.CHECKED
        self.store.save_work(work)
        self.store.append_journal(
            work.id,
            {
                "event": "check_passed" if ok else "check_failed",
                "movement": movement.id,
                "seq": seq,
                "source": origin.source.value,
                "technique": origin.technique,
                "violations": len(report.violations) if report is not None else 0,
            },
        )
        return revision
