# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Check rule abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from music21 import stream

from harmoniatextor.checker.context import CheckerContext
from harmoniatextor.domain.enums import Severity
from harmoniatextor.domain.models import CheckViolation

__all__ = ["CheckRule"]


class CheckRule(ABC):
    """Base class for a symbolic-layer check rule.

    Attributes:
        rule_id: Stable rule identifier.
        name: Display name.
        default_severity: Severity emitted when the profile does not override it.
    """

    rule_id: str
    name: str
    default_severity: Severity = Severity.ERROR

    @abstractmethod
    def run(self, score: stream.Score, ctx: CheckerContext) -> list[CheckViolation]:
        """Run the rule over a score.

        Args:
            score: The score to inspect.
            ctx: Checker context.

        Returns:
            Detected violations.
        """
