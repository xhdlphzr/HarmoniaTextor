# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""LangChain tool wrappers around the composition service."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from harmoniatextor.domain.params import (
    AddPartParams,
    AnnotateParams,
    DeleteMeasureParams,
    EditParams,
    InsertMeasureParams,
    RemovePartParams,
    SetTempoParams,
    SubmitThemeParams,
)
from harmoniatextor.service.service import CompositionService, ToolResult
from harmoniatextor.techniques.registry import TechniqueRegistry

__all__ = ["build_tools", "result_payload"]


def result_payload(result: ToolResult, *, include_score: bool = False) -> str:
    """Render a tool result as a JSON string for the language model.

    Most mutation tools deliberately omit the score: they return only status,
    theme number and violations, and the model fetches the score with the
    ``read`` tool when it needs it, which keeps a long session inside the model
    context window.  The ``insert`` and ``delete`` tools opt in to the full
    score because a structural change makes the previous layout hard to infer.

    Args:
        result: The tool result.
        include_score: Whether to include the full MusicXML.

    Returns:
        A JSON string.
    """
    payload: dict[str, Any] = {
        "ok": result.ok,
        "theme_id": result.theme_id,
        "error_code": result.error_code,
        "message": result.message,
        "warnings": result.warnings,
    }
    if result.report is not None:
        payload["violations"] = result.report.to_dict()["violations"]
    if include_score and result.full_musicxml is not None:
        payload["full_musicxml"] = result.full_musicxml
    return json.dumps(payload, ensure_ascii=False)


def build_tools(
    service: CompositionService,
    work_id: str,
    movement_id: str,
    techniques: TechniqueRegistry | None = None,
) -> list[BaseTool]:
    """Build the full LangChain tool set for one movement.

    Args:
        service: Composition service to call.
        work_id: Active work identifier.
        movement_id: Active movement identifier.
        techniques: Technique registry; defaults to the service registry.

    Returns:
        The read tool, the submission tool, the measure-edit tool and one tool
        per technique.
    """

    def read() -> str:
        """Return the complete current MusicXML score of the movement.

        Returns:
            The resulting text.
        """
        xml = service.current_musicxml(work_id, movement_id)
        return xml or "空谱:本乐章还没有任何声部。请先用 add_part 建立声部。"

    def submit_theme(
        musicxml: str, instrument: str, key: str = "", voice: str = ""
    ) -> str:
        """Submit a theme melody, choosing the target part's instrument.

        Args:
            musicxml: MusicXML fragment.
            instrument: Instrument name.
            key: Key spelling.
            voice: Voice slot name.

        Returns:
            The resulting text.
        """
        return result_payload(
            service.submit_theme(
                work_id,
                movement_id,
                musicxml,
                voice=voice or None,
                instrument=instrument,
                key=key or None,
                check=False,
            )
        )

    def add_part(voice: str, instrument: str) -> str:
        """Add a new instrumental part and always name its instrument.

        Args:
            voice: Voice slot name.
            instrument: Instrument name.

        Returns:
            The resulting text.
        """
        return result_payload(
            service.add_part(work_id, movement_id, voice, instrument, check=False)
        )

    def remove_part(voice: str) -> str:
        """Remove a voice from the movement.

        Args:
            voice: Voice slot name.

        Returns:
            The resulting text.
        """
        return result_payload(
            service.remove_part(work_id, movement_id, voice, check=False)
        )

    def set_tempo(bpm: int) -> str:
        """Change the tempo of the movement.

        Args:
            bpm: Tempo in quarter notes per minute.

        Returns:
            The resulting text.
        """
        return result_payload(service.set_tempo(work_id, movement_id, bpm, check=False))

    def annotate(measure: int, voice: str, mark: str, value: str = "") -> str:
        """Add an expressive mark to one measure of one voice.

        Args:
            measure: One-based measure number.
            voice: Voice slot name.
            mark: Expressive mark kind.
            value: Raw value.

        Returns:
            The resulting text.
        """
        return result_payload(
            service.annotate(
                work_id, movement_id, measure, voice, mark, value, check=False
            )
        )

    def edit(measure: int, voice: str, musicxml: str = "") -> str:
        """Replace or clear one measure of one voice.

        Args:
            measure: One-based measure number.
            voice: Voice slot name.
            musicxml: MusicXML fragment.

        Returns:
            The resulting text.
        """
        return result_payload(
            service.edit_measure(
                work_id, movement_id, measure, voice, musicxml, check=False
            )
        )

    def insert(measure: int, voice: str = "", musicxml: str = "") -> str:
        """Insert a new measure in every voice and receive the full score.

        Args:
            measure: One-based measure number.
            voice: Voice slot name.
            musicxml: MusicXML fragment.

        Returns:
            The resulting text.
        """
        return result_payload(
            service.insert_measure(
                work_id,
                movement_id,
                measure,
                voice=voice or None,
                musicxml=musicxml,
                check=False,
            ),
            include_score=True,
        )

    def delete(measure: int) -> str:
        """Delete a measure from every voice and receive the full score.

        Args:
            measure: One-based measure number.

        Returns:
            The resulting text.
        """
        return result_payload(
            service.delete_measure(work_id, movement_id, measure, check=False),
            include_score=True,
        )

    tools: list[BaseTool] = [
        StructuredTool.from_function(
            func=read,
            name="read",
            description=(
                "Return the complete current MusicXML score of the movement. "
                "Mutation tools no longer return the score, so call this when you "
                "need to inspect the current notes before editing."
            ),
        ),
        StructuredTool.from_function(
            func=submit_theme,
            name="submit_theme",
            description=(
                "Submit a new theme melody as MusicXML and choose the key and mode of "
                "the movement (e.g. 'C' for C major, 'a' for A minor). Give the target "
                "voice slot and, required, that part's instrument. Returns the assigned "
                "theme number and the check result."
            ),
            args_schema=SubmitThemeParams,
        ),
        StructuredTool.from_function(
            func=add_part,
            name="add_part",
            description=(
                "Add a new part to the movement. Give voice (the slot name) and its "
                "instrument (required, e.g. 'Flute', 'Violin', 'Cello', 'Piano'); the "
                "instrument is never inferred. Use this to write separate parts for the "
                "same instrument (e.g. violin1 and violin2) or to add a new instrument."
            ),
            args_schema=AddPartParams,
        ),
        StructuredTool.from_function(
            func=remove_part,
            name="remove_part",
            description=(
                "Remove a voice (part) from the movement by its slot name. Use this "
                "when the planned texture needs fewer voices."
            ),
            args_schema=RemovePartParams,
        ),
        StructuredTool.from_function(
            func=set_tempo,
            name="set_tempo",
            description=(
                "Change the movement tempo in quarter notes per minute (BPM). "
                "Use this to adjust the speed of the whole movement."
            ),
            args_schema=SetTempoParams,
        ),
        StructuredTool.from_function(
            func=annotate,
            name="annotate",
            description=(
                "Add an expressive mark to one measure of one voice. mark is one of: "
                "dynamic (value like 'pp','mf','f','ff'), text (value like 'dolce'), "
                "crescendo, diminuendo, accent, tenuto, staccato, slur, pedal, or "
                "tempo (value = BPM, for a tempo change inside the movement)."
            ),
            args_schema=AnnotateParams,
        ),
        StructuredTool.from_function(
            func=edit,
            name="edit",
            description=(
                "Replace or clear one measure of one voice. Give the one-based measure "
                "number and the voice slot; optionally give a MusicXML fragment whose "
                "first melody becomes the measure. With no fragment the measure is "
                "cleared. Use this to fix local symbolic-check violations without "
                "rewriting the whole piece; call read to inspect the current score."
            ),
            args_schema=EditParams,
        ),
        StructuredTool.from_function(
            func=insert,
            name="insert",
            description=(
                "Insert a new measure at the given one-based position in every voice; "
                "later measures shift later. Optionally give a MusicXML fragment and the "
                "voice that should receive its melody. Returns the complete updated score. "
                "Use this to extend the piece."
            ),
            args_schema=InsertMeasureParams,
        ),
        StructuredTool.from_function(
            func=delete,
            name="delete",
            description=(
                "Delete the measure at the given one-based number from every voice; "
                "later measures shift earlier. Returns the complete updated score. Use "
                "this to remove a measure entirely; a fragment-free edit only clears its "
                "contents."
            ),
            args_schema=DeleteMeasureParams,
        ),
    ]

    registry = techniques if techniques is not None else service.techniques
    for technique in registry.all():

        def make_func(technique_id: str) -> Any:
            def apply(**kwargs: Any) -> str:
                return result_payload(
                    service.apply_technique(
                        work_id, movement_id, technique_id, kwargs, check=False
                    )
                )

            return apply

        tools.append(
            StructuredTool.from_function(
                func=make_func(technique.id),
                name=f"technique_{technique.id}",
                description=f"{technique.name}: {technique.summary}",
                args_schema=technique.params_model,
            )
        )
    return tools
