# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Symbolic-layer checker: rules, profiles, engine and feedback."""

from __future__ import annotations

from harmoniatextor.checker.base import CheckRule
from harmoniatextor.checker.context import CheckerContext, StructuralExpectation
from harmoniatextor.checker.engine import CheckEngine, format_feedback
from harmoniatextor.checker.profile import RuleSetting, ValidationProfile
from harmoniatextor.checker.rules import BUILTIN_RULES

__all__ = [
    "BUILTIN_RULES",
    "CheckEngine",
    "CheckRule",
    "CheckerContext",
    "RuleSetting",
    "StructuralExpectation",
    "ValidationProfile",
    "format_feedback",
]
