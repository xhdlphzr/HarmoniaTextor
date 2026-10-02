# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Validation profiles: per-genre enable/severity settings for rules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from harmoniatextor.checker.rules import BUILTIN_RULES
from harmoniatextor.domain.enums import Severity

__all__ = ["RuleSetting", "ValidationProfile", "profile_for_rules"]


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


def profile_for_rules(enabled: Iterable[str]) -> ValidationProfile:
    """Build a profile that runs only the given rules.

    Args:
        enabled: Rule identifiers to keep enabled; every other built-in rule is
            explicitly disabled.

    Returns:
        A validation profile with per-rule enablement.
    """
    keep = set(enabled)
    return ValidationProfile(
        settings={rule.rule_id: RuleSetting(enabled=rule.rule_id in keep) for rule in BUILTIN_RULES}
    )
