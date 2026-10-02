# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for domain.models."""

from __future__ import annotations

from harmoniatextor.domain.enums import Severity, VoiceSlot
from harmoniatextor.domain.models import (
    CheckReport,
    CheckViolation,
    StyleSelection,
    ThemeNote,
    theme_fingerprint,
)


class TestModels:
    """Entity behaviour."""

    def test_theme_fingerprint_stable(self) -> None:
        """Fingerprints are deterministic."""
        notes = [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)]
        assert theme_fingerprint(notes) == theme_fingerprint(notes)

    def test_report_ok(self) -> None:
        """A report without errors is ok."""
        report = CheckReport()
        assert report.ok
        assert report.to_dict() == {"ok": True, "violations": []}

    def test_report_with_error(self) -> None:
        """An error makes the report fail and exposes warnings separately."""
        error = CheckViolation("r", Severity.ERROR, 1, "s", None, "k", "m", "s")
        warning = CheckViolation("r2", Severity.WARNING, 2, "a", None, "k", "m", "s")
        report = CheckReport(violations=[error, warning])
        assert not report.ok
        assert report.errors == [error]
        assert report.warnings == [warning]
        assert report.to_dict()["ok"] is False

    def test_voice_slot_values(self) -> None:
        """Voice slots have the expected canonical values."""
        assert VoiceSlot.SOPRANO.value == "soprano"

    def test_style_selection_roundtrip(self) -> None:
        """A style selection survives a dictionary round trip."""
        selection = StyleSelection(
            id="s-1",
            name="Mine",
            rules=frozenset({"empty", "voices"}),
            techniques=frozenset({"imitation"}),
        )
        data = selection.to_dict()
        assert data["rules"] == ["empty", "voices"]
        assert StyleSelection.from_dict(data) == selection

    def test_style_selection_defaults(self) -> None:
        """Missing lists default to empty sets."""
        selection = StyleSelection.from_dict({"id": "x", "name": "X"})
        assert selection.rules == frozenset()
        assert selection.techniques == frozenset()
