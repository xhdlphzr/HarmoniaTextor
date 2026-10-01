# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for checker.base."""

from __future__ import annotations

from harmoniatextor.checker.base import CheckRule
from harmoniatextor.checker.rules import EmptyScoreRule
from harmoniatextor.domain.enums import Severity


class TestCheckRule:
    """The rule base class defaults."""

    def test_default_severity(self) -> None:
        """Rules default to error severity."""
        assert CheckRule.default_severity is Severity.ERROR
        assert EmptyScoreRule().default_severity is Severity.ERROR

    def test_identifiers(self) -> None:
        """Concrete rules carry an id and a name."""
        rule = EmptyScoreRule()
        assert rule.rule_id == "empty"
        assert rule.name
