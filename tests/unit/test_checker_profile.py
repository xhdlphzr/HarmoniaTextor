# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for checker.profile."""

from __future__ import annotations

from harmoniatextor.checker.profile import RuleSetting, ValidationProfile, profile_for_rules
from harmoniatextor.domain.enums import Severity


class TestValidationProfile:
    """Rule enablement and severity overrides."""

    def test_unknown_defaults_enabled(self) -> None:
        """Unknown rules default to enabled."""
        assert ValidationProfile().setting_for("nope").enabled

    def test_explicit_setting(self) -> None:
        """Explicit settings are returned."""
        profile = ValidationProfile(
            {"pf5th": RuleSetting(enabled=False, severity=Severity.WARNING)}
        )
        setting = profile.setting_for("pf5th")
        assert not setting.enabled
        assert setting.severity is Severity.WARNING

    def test_profile_for_rules(self) -> None:
        """Only the requested rules stay enabled."""
        profile = profile_for_rules({"empty", "voices"})
        assert profile.setting_for("empty").enabled
        assert profile.setting_for("voices").enabled
        assert not profile.setting_for("pf5th").enabled
