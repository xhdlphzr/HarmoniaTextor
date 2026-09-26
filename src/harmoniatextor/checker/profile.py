# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Validation profiles: per-genre enable/severity settings for rules."""

from __future__ import annotations

from dataclasses import dataclass, field

from harmoniatextor.domain.enums import Severity

__all__ = ["RuleSetting", "ValidationProfile"]


@dataclass(frozen=True, slots=True)
class RuleSetting:
    """Enablement and severity override for one rule.

    Attributes:
        enabled: Whether the rule runs.
        severity: Optional severity override.
    """

    enabled: bool = True
    severity: Severity | None = None


@dataclass(slots=True)
class ValidationProfile:
    """A mapping of rule identifiers to their settings.

    Attributes:
        settings: Explicit per-rule settings.
    """

    settings: dict[str, RuleSetting] = field(default_factory=dict)

    def setting_for(self, rule_id: str) -> RuleSetting:
        """Return the setting for a rule.

        Args:
            rule_id: Rule identifier.

        Returns:
            The explicit setting, or a default enabled setting.
        """
        return self.settings.get(rule_id, RuleSetting())
