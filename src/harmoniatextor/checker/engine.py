# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""The check engine that runs all enabled rules and aggregates a report."""

from __future__ import annotations

from music21 import stream

from harmoniatextor.checker.base import CheckRule
from harmoniatextor.checker.context import CheckerContext
from harmoniatextor.checker.profile import ValidationProfile
from harmoniatextor.checker.rules import BUILTIN_RULES
from harmoniatextor.domain.models import CheckReport, CheckViolation

__all__ = ["CheckEngine", "format_feedback"]


class CheckEngine:
    """Run symbolic check rules and aggregate their violations.

    Attributes:
        rules: Rule instances to run.
        profile: Validation profile controlling enablement and severity.
    """

    def __init__(
        self,
        rules: list[CheckRule] | None = None,
        profile: ValidationProfile | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            rules: Rule instances; defaults to the 14 built-in rules.
            profile: Validation profile; defaults to all rules enabled.
        """
        self.rules: list[CheckRule] = (
            rules if rules is not None else [rule() for rule in BUILTIN_RULES]
        )
        self.profile = profile if profile is not None else ValidationProfile()

    def run(self, score: stream.Score, ctx: CheckerContext) -> CheckReport:
        """Run all enabled rules over a score.

        Args:
            score: The score to validate.
            ctx: Checker context.

        Returns:
            The aggregated report, sorted by measure.
        """
        violations: list[CheckViolation] = []
        for rule in self.rules:
            setting = self.profile.setting_for(rule.rule_id)
            if not setting.enabled:
                continue
            for detected in rule.run(score, ctx):
                if setting.severity is None:
                    violations.append(detected)
                    continue
                violations.append(
                    CheckViolation(
                        rule_id=detected.rule_id,
                        severity=setting.severity,
                        measure=detected.measure,
                        voice_a=detected.voice_a,
                        voice_b=detected.voice_b,
                        kind=detected.kind,
                        message_zh=detected.message_zh,
                        snippet=detected.snippet,
                    )
                )
        violations.sort(key=lambda item: (item.measure, item.rule_id))
        return CheckReport(violations=violations)


def format_feedback(report: CheckReport) -> str:
    """Render a check report as Chinese feedback for the language model.

    Args:
        report: The check report.

    Returns:
        A human/LLM readable feedback block.
    """
    if report.ok and not report.violations:
        return "[检查通过] 符号层未发现任何违规。"
    if report.ok:
        return f"[检查通过,含 {len(report.warnings)} 条警告]"
    lines = [f"[检查不通过] 共 {len(report.errors)} 处违规:"]
    for index, item in enumerate(report.errors, start=1):
        voices = (
            f"{item.voice_a} 与 {item.voice_b}" if item.voice_b else (item.voice_a or "全体声部")
        )
        lines.append(
            f"{index}) 小节 {item.measure} · 声部 {voices} · 规则 {item.rule_id}:"
            f"{item.message_zh} 具体位置:{item.snippet}"
        )
    lines.append("请针对上面每一处,明确指出并修改对应小节与声部的音符。")
    return "\n".join(lines)
