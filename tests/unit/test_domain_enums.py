# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for domain.enums."""

from __future__ import annotations

from harmoniatextor.domain.enums import (
    TOOL_KIND_LABELS,
    VOICE_LABELS,
    VOICE_SLOTS,
    WORK_STATUS_LABELS,
    Severity,
    ToolKind,
    VoiceSlot,
    WorkStatus,
)
from harmoniatextor.i18n import catalog_for


class TestEnums:
    """Voice slots, statuses and label mappings."""

    def test_voice_slots(self) -> None:
        """The canonical voice slots are stable."""
        assert VOICE_SLOTS == ("soprano", "alto", "tenor", "bass")
        assert VoiceSlot.SOPRANO.value == "soprano"

    def test_labels_cover_members(self) -> None:
        """Every tool kind and status has a label."""
        for kind in ToolKind:
            assert kind.value in TOOL_KIND_LABELS
        for status in WorkStatus:
            assert status.value in WORK_STATUS_LABELS

    def test_labels_are_message_ids(self) -> None:
        """Every label resolves to a message id in the catalogue."""
        english = catalog_for("en")
        for mapping in (TOOL_KIND_LABELS, WORK_STATUS_LABELS, VOICE_LABELS):
            for message_id in mapping.values():
                assert message_id in english

    def test_severity(self) -> None:
        """Severities serialise to their names."""
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
