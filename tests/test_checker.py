# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the fifteen symbolic check rules and the engine."""

from __future__ import annotations

from music21 import stream

from harmoniatextor.checker.context import CheckerContext, StructuralExpectation
from harmoniatextor.checker.engine import CheckEngine, format_feedback
from harmoniatextor.checker.profile import RuleSetting, ValidationProfile
from harmoniatextor.checker.rules import (
    BUILTIN_RULES,
    RULE_CONSTRAINTS,
    CadenceTypeRule,
    ChordOmissionRule,
    DiminishedIntervalRule,
    DominantResolutionRule,
    EmptyScoreRule,
    ExcessiveSpacingRule,
    FinalOuterIntervalRule,
    FixedVoiceCountRule,
    HiddenFifthsRule,
    HiddenOctavesRule,
    LeadingToneRule,
    ParallelFifthsRule,
    ParallelOctavesRule,
    TonalUnityRule,
    VoiceCrossingRule,
    _slice_for,
)
from harmoniatextor.checker.slices import build_slices
from harmoniatextor.domain.enums import Severity
from harmoniatextor.domain.models import CheckReport, CheckViolation
from harmoniatextor.score.analysis import VoiceEvent
from harmoniatextor.score.streamops import ScoreEditor


def place(score: stream.Score, voice: str, offset: float, pitch: str, length: float = 1.0) -> None:
    """Place a note at a global offset (assumes 4/4)."""
    measure = int(offset // 4) + 1
    inner = offset - (measure - 1) * 4
    ScoreEditor(score).place_note(voice, measure, inner, pitch, length)


def rule_ids(violations: list[CheckViolation]) -> list[str]:
    """Return the rule ids of violations."""
    return [item.rule_id for item in violations]


class TestVoiceLeading:
    """Rules R1-R7."""

    def test_parallel_fifths(self, score4: stream.Score) -> None:
        """Parallel fifths are detected."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "soprano", 1.0, "D5")
        place(score4, "alto", 0.0, "F4")
        place(score4, "alto", 1.0, "G4")
        assert rule_ids(ParallelFifthsRule().run(score4, CheckerContext())) == ["pf5th"]

    def test_parallel_fifths_clean(self, score4: stream.Score) -> None:
        """Non-parallel motion is accepted."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "soprano", 1.0, "D5")
        place(score4, "alto", 0.0, "F4")
        place(score4, "alto", 1.0, "E4")
        assert ParallelFifthsRule().run(score4, CheckerContext()) == []

    def test_parallel_octaves(self, score4: stream.Score) -> None:
        """Parallel octaves are detected."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "soprano", 1.0, "D5")
        place(score4, "alto", 0.0, "C4")
        place(score4, "alto", 1.0, "D4")
        assert rule_ids(ParallelOctavesRule().run(score4, CheckerContext())) == ["po8ve"]

    def test_hidden_fifths(self, score4: stream.Score) -> None:
        """Hidden fifths are detected."""
        place(score4, "soprano", 0.0, "G4")
        place(score4, "soprano", 1.0, "C5")
        place(score4, "alto", 0.0, "E4")
        place(score4, "alto", 1.0, "F4")
        assert rule_ids(HiddenFifthsRule().run(score4, CheckerContext())) == ["hf5th"]

    def test_hidden_octaves(self, score4: stream.Score) -> None:
        """Hidden octaves are detected."""
        place(score4, "soprano", 0.0, "C#5")
        place(score4, "soprano", 1.0, "D5")
        place(score4, "alto", 0.0, "C4")
        place(score4, "alto", 1.0, "D4")
        assert rule_ids(HiddenOctavesRule().run(score4, CheckerContext())) == ["ho8ve"]

    def test_voice_crossing(self, score4: stream.Score) -> None:
        """Crossing voices are detected."""
        place(score4, "soprano", 0.0, "C4")
        place(score4, "alto", 0.0, "C5")
        assert rule_ids(VoiceCrossingRule().run(score4, CheckerContext())) == ["crossing"]

    def test_excessive_spacing(self, score4: stream.Score) -> None:
        """Over-wide spacing is detected."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "alto", 0.0, "C3")
        assert rule_ids(ExcessiveSpacingRule().run(score4, CheckerContext())) == ["spacing"]

    def test_final_outer_interval(self, score4: stream.Score) -> None:
        """A bad final outer interval is detected."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "bass", 0.0, "D4")
        assert rule_ids(FinalOuterIntervalRule().run(score4, CheckerContext())) == ["final_outer"]

    def test_final_outer_interval_ok(self, score4: stream.Score) -> None:
        """A perfect final interval passes."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "bass", 0.0, "C3")
        assert FinalOuterIntervalRule().run(score4, CheckerContext()) == []

    def test_final_outer_interval_empty(self) -> None:
        """An empty score has no final interval."""
        score = stream.Score()
        assert FinalOuterIntervalRule().run(score, CheckerContext()) == []


class TestHarmony:
    """Rules R8-R11."""

    def test_dominant_resolution(self, score4: stream.Score) -> None:
        """An unresolved dominant seventh is detected."""
        for voice, pitch in zip(
            ["soprano", "alto", "tenor", "bass"], ["F4", "D4", "B3", "G3"], strict=True
        ):
            place(score4, voice, 0.0, pitch)
        for voice, pitch in zip(
            ["soprano", "alto", "tenor", "bass"], ["A4", "F4", "D4", "D3"], strict=True
        ):
            place(score4, voice, 1.0, pitch)
        assert rule_ids(DominantResolutionRule().run(score4, CheckerContext())) == ["dom7res"]

    def test_dominant_resolution_ok(self, score4: stream.Score) -> None:
        """A resolved dominant seventh passes."""
        for voice, pitch in zip(
            ["soprano", "alto", "tenor", "bass"], ["F4", "D4", "B3", "G3"], strict=True
        ):
            place(score4, voice, 0.0, pitch)
        for voice, pitch in zip(
            ["soprano", "alto", "tenor", "bass"], ["E4", "C4", "G3", "C3"], strict=True
        ):
            place(score4, voice, 1.0, pitch)
        assert DominantResolutionRule().run(score4, CheckerContext()) == []

    def test_leading_tone(self, score4: stream.Score) -> None:
        """An unresolved leading tone is detected."""
        place(score4, "soprano", 0.0, "B4")
        place(score4, "soprano", 1.0, "G4")
        assert rule_ids(LeadingToneRule().run(score4, CheckerContext(tonic="C"))) == ["leading"]

    def test_leading_tone_ok(self, score4: stream.Score) -> None:
        """A rising leading tone passes."""
        place(score4, "soprano", 0.0, "B4")
        place(score4, "soprano", 1.0, "C5")
        assert LeadingToneRule().run(score4, CheckerContext(tonic="C")) == []

    def test_doubled_leading_tone(self, score4: stream.Score) -> None:
        """A doubled leading tone is detected."""
        place(score4, "soprano", 0.0, "B4")
        place(score4, "alto", 0.0, "B3")
        assert rule_ids(ChordOmissionRule().run(score4, CheckerContext(tonic="C"))) == ["omission"]

    def test_diminished_interval(self, score4: stream.Score) -> None:
        """An unresolved diminished leap is detected."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "soprano", 1.0, "F#5")
        place(score4, "soprano", 2.0, "A5")
        assert rule_ids(DiminishedIntervalRule().run(score4, CheckerContext())) == ["diminterval"]

    def test_diminished_interval_ok(self, score4: stream.Score) -> None:
        """A resolved diminished leap passes."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "soprano", 1.0, "F#5")
        place(score4, "soprano", 2.0, "F5")
        assert DiminishedIntervalRule().run(score4, CheckerContext()) == []


class TestTonal:
    """Rules R12-R14."""

    def test_tonal_unity(self, score4: stream.Score) -> None:
        """A missing tonic at an expectation is detected."""
        place(score4, "soprano", 0.0, "D5")
        context = CheckerContext(expectations=[StructuralExpectation(measure=1, key="C")])
        assert rule_ids(TonalUnityRule().run(score4, context)) == ["tonality"]

    def test_tonal_unity_ok(self, score4: stream.Score) -> None:
        """A present tonic passes."""
        place(score4, "soprano", 0.0, "C5")
        context = CheckerContext(expectations=[StructuralExpectation(measure=1, key="C")])
        assert TonalUnityRule().run(score4, context) == []

    def test_tonal_unity_no_expectations(self, score4: stream.Score) -> None:
        """Without expectations the rule is skipped."""
        place(score4, "soprano", 0.0, "D5")
        assert TonalUnityRule().run(score4, CheckerContext()) == []

    def test_cadence(self, score4: stream.Score) -> None:
        """A missing authentic cadence is detected."""
        place(score4, "soprano", 0.0, "D5")
        place(score4, "soprano", 4.0, "E5")
        context = CheckerContext(expectations=[StructuralExpectation(measure=2, cadence=True)])
        assert rule_ids(CadenceTypeRule().run(score4, context)) == ["cadence"]

    def test_cadence_ok(self, score4: stream.Score) -> None:
        """An authentic cadence passes."""
        for voice, pitch in zip(["soprano", "bass"], ["G4", "G3"], strict=True):
            place(score4, voice, 0.0, pitch)
        for voice, pitch in zip(["soprano", "bass"], ["C5", "C3"], strict=True):
            place(score4, voice, 4.0, pitch)
        context = CheckerContext(expectations=[StructuralExpectation(measure=2, cadence=True)])
        assert CadenceTypeRule().run(score4, context) == []

    def test_cadence_no_points(self, score4: stream.Score) -> None:
        """Without cadence points the rule is skipped."""
        place(score4, "soprano", 0.0, "C5")
        assert CadenceTypeRule().run(score4, CheckerContext()) == []

    def test_voice_count(self, score4: stream.Score) -> None:
        """A changing voice count is detected."""
        for measure in range(20):
            place(score4, "soprano", measure * 4.0, "C5")
        for measure in range(4):
            place(score4, "alto", measure * 4.0, "E4")
        assert rule_ids(FixedVoiceCountRule().run(score4, CheckerContext())) == ["voices"]

    def test_voice_count_disabled(self, score4: stream.Score) -> None:
        """The rule can be disabled by context."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "soprano", 4.0, "D5")
        assert FixedVoiceCountRule().run(score4, CheckerContext(enforce_voice_count=False)) == []

    def test_tonal_unity_skips_keyless(self, score4: stream.Score) -> None:
        """Expectations without a key are skipped."""
        place(score4, "soprano", 0.0, "D5")
        context = CheckerContext(expectations=[StructuralExpectation(measure=1, key=None)])
        assert TonalUnityRule().run(score4, context) == []

    def test_tonal_unity_far_measure(self, score4: stream.Score) -> None:
        """A far-away expectation falls back to the last slice."""
        place(score4, "soprano", 0.0, "D5")
        context = CheckerContext(expectations=[StructuralExpectation(measure=999, key="C")])
        assert rule_ids(TonalUnityRule().run(score4, context)) == ["tonality"]

    def test_cadence_far_measure(self, score4: stream.Score) -> None:
        """A cadence point before the music falls back to all slices."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "soprano", 4.0, "D5")
        context = CheckerContext(expectations=[StructuralExpectation(measure=0, cadence=True)])
        assert rule_ids(CadenceTypeRule().run(score4, context)) == ["cadence"]

    def test_slice_for_fallback(self, score4: stream.Score) -> None:
        """Locating an unknown event falls back to the first slice."""
        place(score4, "soprano", 0.0, "C5")
        slices = build_slices(score4)
        missing = VoiceEvent(offset=99.0, measure=99, pitch="C5", midi=72, quarter_length=1.0)
        assert _slice_for(slices, missing) is slices[0]


class TestEmpty:
    """The empty-score rule."""

    def test_empty_score(self) -> None:
        """An empty score is rejected."""
        assert rule_ids(EmptyScoreRule().run(stream.Score(), CheckerContext())) == ["empty"]

    def test_non_empty_score(self, score4: stream.Score) -> None:
        """A score with at least one note passes."""
        place(score4, "soprano", 0.0, "C5")
        assert EmptyScoreRule().run(score4, CheckerContext()) == []


class TestEngine:
    """Engine aggregation, profiles and feedback."""

    def test_engine_runs_all(self, score4: stream.Score) -> None:
        """The engine runs the built-in rules."""
        report = CheckEngine().run(score4, CheckerContext())
        assert isinstance(report, CheckReport)

    def test_profile_disables_rule(self, score4: stream.Score) -> None:
        """A disabled rule does not run."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "alto", 0.0, "C3")
        engine = CheckEngine(profile=ValidationProfile({"spacing": RuleSetting(enabled=False)}))
        assert not any(
            item.rule_id == "spacing" for item in engine.run(score4, CheckerContext()).violations
        )

    def test_profile_overrides_severity(self, score4: stream.Score) -> None:
        """A severity override is applied."""
        place(score4, "soprano", 0.0, "C5")
        place(score4, "alto", 0.0, "C3")
        engine = CheckEngine(
            profile=ValidationProfile({"spacing": RuleSetting(severity=Severity.WARNING)})
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
            "pf5th", Severity.ERROR, 3, "soprano", "alto", "parallel_fifth", "平行五度", "x"
        )
        text = format_feedback(CheckReport(violations=[error]))
        assert "检查不通过" in text
        assert "pf5th" in text
        assert "修改建议" in text
        assert "edit(measure, voice, musicxml)" in text

    def test_profile_default(self) -> None:
        """Unknown rules default to enabled."""
        assert ValidationProfile().setting_for("unknown").enabled


def test_rule_constraints_cover_every_rule() -> None:
    """Every built-in rule has an AI-facing constraint description."""
    assert {rule.rule_id for rule in BUILTIN_RULES} == set(RULE_CONSTRAINTS)
    assert all(text.strip() for text in RULE_CONSTRAINTS.values())
