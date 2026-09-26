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
    MERGE = "merge"


class WorkStatus(StrEnum):
    """Lifecycle state of a work."""

    DRAFT = "draft"
    CHECKED = "checked"
    AUDITING = "auditing"
    REVISING = "revising"
    FINAL = "final"


WORK_STATUS_LABELS: dict[str, str] = {
    WorkStatus.DRAFT.value: "草稿",
    WorkStatus.CHECKED.value: "已校验",
    WorkStatus.AUDITING.value: "送审中",
    WorkStatus.REVISING.value: "修订中",
    WorkStatus.FINAL.value: "已定稿",
}

VOICE_LABELS: dict[str, str] = {
    VoiceSlot.SOPRANO.value: "女高音",
    VoiceSlot.ALTO.value: "女低音",
    VoiceSlot.TENOR.value: "男高音",
    VoiceSlot.BASS.value: "男低音",
    "solo": "独奏",
    "violin": "小提琴",
    "violin1": "第一小提琴",
    "violin2": "第二小提琴",
    "viola": "中提琴",
    "cello": "大提琴",
    "contrabass": "低音提琴",
    "flute": "长笛",
    "oboe": "双簧管",
    "clarinet": "单簧管",
    "bassoon": "大管",
    "horn": "圆号",
    "trumpet": "小号",
    "trombone": "长号",
    "tuba": "大号",
    "timpani": "定音鼓",
    "harpsichord": "羽管键琴",
    "organ": "管风琴",
    "piano": "钢琴",
}

TOOL_KIND_LABELS: dict[str, str] = {
    ToolKind.SUBMIT.value: "提交主题",
    ToolKind.TECHNIQUE.value: "技法",
    ToolKind.REFINE.value: "覆写谱面",
    ToolKind.EDIT.value: "编辑小节",
    ToolKind.IMPORT.value: "初始谱",
    ToolKind.ROLLBACK.value: "回退",
    ToolKind.PART.value: "声部",
    ToolKind.TEMPO.value: "速度",
    ToolKind.MERGE.value: "合并",
}
