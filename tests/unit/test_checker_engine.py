# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for checker.engine."""

from __future__ import annotations

from music21 import stream

from harmoniatextor.checker.context import CheckerContext
from harmoniatextor.checker.engine import CheckEngine, format_feedback
from harmoniatextor.checker.profile import RuleSetting, ValidationProfile
from harmoniatextor.domain.enums import Severity
from harmoniatextor.domain.models import CheckReport, CheckViolation
from harmoniatextor.score.streamops import ScoreEditor


def place(
    score: stream.Score, voice: str, offset: float, pitch: str, length: float = 1.0
) -> None:
    """Place a note at a global offset (assumes 4/4).

    Args:
        score: The score to inspect.
        voice: Voice slot name.
        offset: The offset.
        pitch: Scientific pitch name.
        length: The length.
    """
    measure = int(offset // 4) + 1
    inner = offset - (measure - 1) * 4
    ScoreEditor(score).place_note(voice, measure, inner, pitch, length)


class TestEngine:
    """Engine aggregation, profiles and feedback."""

    def test_engine_runs_all(self, score4: stream.Score) -> None:
        """The engine runs the built-in rules.

        Args:
            score4: An empty four-voice score.
        """
        report = CheckEngine().run(score4, CheckerContext())
        assert isinstance(report, CheckReport)

    def test_profile_disables_rule(self, score4: stream.Score) -> None:
        """A disabled rule does not run.

        Args:
            score4: An empty four-voice score.
        """
        place(score4, "soprano", 0.0, "C5")
        place(score4, "alto", 0.0, "C3")
        engine = CheckEngine(
            profile=ValidationProfile({"spacing": RuleSetting(enabled=False)})
        )
        assert not any(
            item.rule_id == "spacing"
            for item in engine.run(score4, CheckerContext()).violations
        )

    def test_profile_overrides_severity(self, score4: stream.Score) -> None:
        """A severity override is applied.

        Args:
            score4: An empty four-voice score.
        """
        place(score4, "soprano", 0.0, "C5")
        place(score4, "alto", 0.0, "C3")
        engine = CheckEngine(
            profile=ValidationProfile(
                {"spacing": RuleSetting(severity=Severity.WARNING)}
            )
        )
        report = engine.run(score4, CheckerContext())
        assert report.ok
        assert report.warnings

    def test_feedback_passed(self) -> None:
        """Passing feedback is concise."""
        assert "检查通过" in format_feedback(CheckReport())

    def test_feedback_warnings(self) -> None:
        """Warnings are reported separately."""
        warning = CheckViolation("r", Severity.WARNING, 1, None, None, "k", "m", "s")
        assert "警告" in format_feedback(CheckReport(violations=[warning]))

    def test_feedback_errors(self) -> None:
        """Errors are rendered with guidance."""
        error = CheckViolation(
            "pf5th",
            Severity.ERROR,
            3,
            "soprano",
            "alto",
            "parallel_fifth",
            "平行五度",
            "x",
        )
        text = format_feedback(CheckReport(violations=[error]))
        assert "检查不通过" in text
        assert "pf5th" in text
        assert "修改建议" in text
        assert "edit(measure, voice, musicxml)" in text

    def test_profile_default(self) -> None:
        """Unknown rules default to enabled."""
        assert ValidationProfile().setting_for("unknown").enabled

    def test_run_with_profile_argument(self, score4: stream.Score) -> None:
        """A profile passed to run() overrides the engine profile.

        Args:
            score4: An empty four-voice score.
        """
        place(score4, "soprano", 0.0, "C5")
        place(score4, "alto", 0.0, "C3")
        engine = CheckEngine()
        report = engine.run(score4, CheckerContext(), profile=ValidationProfile())
        assert any(item.rule_id == "spacing" for item in report.violations)
