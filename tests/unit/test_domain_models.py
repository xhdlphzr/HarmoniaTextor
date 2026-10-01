# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for domain.models."""

from __future__ import annotations

from harmoniatextor.domain.enums import Severity, VoiceSlot
from harmoniatextor.domain.models import (
    CheckReport,
    CheckViolation,
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
