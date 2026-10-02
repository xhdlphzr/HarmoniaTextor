# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Enumerations and small shared constants for the domain layer."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "TOOL_KIND_LABELS",
    "VOICE_LABELS",
    "VOICE_SLOTS",
    "WORK_STATUS_LABELS",
    "Severity",
    "ToolKind",
    "VoiceSlot",
    "WorkStatus",
]


class VoiceSlot(StrEnum):
    """Canonical baroque voice slots.

    Genres may define additional instrument slots as plain strings, but these
    four form the default four-part counterpoint core.
    """

    SOPRANO = "soprano"
    ALTO = "alto"
    TENOR = "tenor"
    BASS = "bass"


VOICE_SLOTS: tuple[str, ...] = (
    VoiceSlot.SOPRANO,
    VoiceSlot.ALTO,
    VoiceSlot.TENOR,
    VoiceSlot.BASS,
)


class Severity(StrEnum):
    """Severity of a symbolic-layer violation."""

    ERROR = "error"
    WARNING = "warning"


class ToolKind(StrEnum):
    """Origin of a score revision."""

    SUBMIT = "submit"
    TECHNIQUE = "technique"
    REFINE = "refine"
    EDIT = "edit"
    IMPORT = "import"
    ROLLBACK = "rollback"
    PART = "part"
    TEMPO = "tempo"
    MARK = "mark"
    MERGE = "merge"


class WorkStatus(StrEnum):
    """Lifecycle state of a work."""

    DRAFT = "draft"
    CHECKED = "checked"
    AUDITING = "auditing"
    REVISING = "revising"
    FINAL = "final"


#: Display label message ids (localised by ``harmoniatextor.i18n``).
WORK_STATUS_LABELS: dict[str, str] = {
    WorkStatus.DRAFT.value: "status.draft",
    WorkStatus.CHECKED.value: "status.checked",
    WorkStatus.AUDITING.value: "status.auditing",
    WorkStatus.REVISING.value: "status.revising",
    WorkStatus.FINAL.value: "status.final",
}

#: Display label message ids (localised by ``harmoniatextor.i18n``).
VOICE_LABELS: dict[str, str] = {
    VoiceSlot.SOPRANO.value: "voice.soprano",
    VoiceSlot.ALTO.value: "voice.alto",
    VoiceSlot.TENOR.value: "voice.tenor",
    VoiceSlot.BASS.value: "voice.bass",
    "solo": "voice.solo",
    "violin": "voice.violin",
    "violin1": "voice.violin1",
    "violin2": "voice.violin2",
    "viola": "voice.viola",
    "cello": "voice.cello",
    "contrabass": "voice.contrabass",
    "flute": "voice.flute",
    "oboe": "voice.oboe",
    "clarinet": "voice.clarinet",
    "bassoon": "voice.bassoon",
    "horn": "voice.horn",
    "trumpet": "voice.trumpet",
    "trombone": "voice.trombone",
    "tuba": "voice.tuba",
    "timpani": "voice.timpani",
    "harpsichord": "voice.harpsichord",
    "organ": "voice.organ",
    "piano": "voice.piano",
}

#: Display label message ids (localised by ``harmoniatextor.i18n``).
TOOL_KIND_LABELS: dict[str, str] = {
    ToolKind.SUBMIT.value: "tool.submit",
    ToolKind.TECHNIQUE.value: "tool.technique",
    ToolKind.REFINE.value: "tool.refine",
    ToolKind.EDIT.value: "tool.edit",
    ToolKind.IMPORT.value: "tool.import",
    ToolKind.ROLLBACK.value: "tool.rollback",
    ToolKind.PART.value: "tool.part",
    ToolKind.TEMPO.value: "tool.tempo",
    ToolKind.MARK.value: "tool.mark",
    ToolKind.MERGE.value: "tool.merge",
}
