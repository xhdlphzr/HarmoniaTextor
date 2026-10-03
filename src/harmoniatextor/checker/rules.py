# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""The fifteen built-in symbolic check rules."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from music21 import stream

from harmoniatextor.checker.base import CheckRule
from harmoniatextor.checker.context import CheckerContext
from harmoniatextor.checker.slices import (
    Slice,
    build_slices,
    pair_sequence,
    pitch_classes,
    tonic_pc,
)
from harmoniatextor.domain.models import CheckViolation
from harmoniatextor.score.analysis import VoiceEvent, voice_order

__all__ = [
    "BUILTIN_RULES",
    "RULE_CONSTRAINTS",
    "CadenceTypeRule",
    "ChordOmissionRule",
    "DiminishedIntervalRule",
    "DominantResolutionRule",
    "EmptyScoreRule",
    "ExcessiveSpacingRule",
    "FinalOuterIntervalRule",
    "FixedVoiceCountRule",
    "HiddenFifthsRule",
    "HiddenOctavesRule",
    "LeadingToneRule",
    "ParallelFifthsRule",
    "ParallelOctavesRule",
    "TonalUnityRule",
    "VoiceCrossingRule",
]

_TRITONE = 6
_DIMINISHED_SEVENTH = 9
_STEP = 2
_PERFECT_FIFTH = 7
_PERFECT_OCTAVE = 0
_MIN_VOICES = 2
_MIN_CHORDS = 2
_MIN_OPENING_CLOSING = 2
_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class ViolationData:
    """Descriptive payload for a rule violation.

    Attributes:
        kind: Machine-readable category.
        message: English explanation.
        snippet: Short musical description.
        voice_a: First voice.
        voice_b: Second voice.
    """

    kind: str
    message: str
    snippet: str
    voice_a: str | None = None
    voice_b: str | None = None


def _violation(rule: CheckRule, item: Slice, data: ViolationData) -> CheckViolation:
    """Build a violation anchored at a slice.

    Args:
        rule: Emitting rule.
        item: Slice where the violation occurs.
        data: Violation details.

    Returns:
        The violation.
    """
    return CheckViolation(
        rule_id=rule.rule_id,
        severity=rule.default_severity,
        measure=item.measure,
        voice_a=data.voice_a,
        voice_b=data.voice_b,
        kind=data.kind,
        message=data.message,
        snippet=data.snippet,
    )


class EmptyScoreRule(CheckRule):
    """Reject a score that contains no notes at all."""

    rule_id = "empty"
    name = "Empty score"

    def run(self, score: stream.Score, _ctx: CheckerContext) -> list[CheckViolation]:
        """Detect a score without any note.

        Args:
            score: The score to inspect.
            _ctx: The ctx.

        Returns:
            The run result.
        """
        if bool(list(score.recurse().notes)):
            return []
        return [
            CheckViolation(
                rule_id=self.rule_id,
                severity=self.default_severity,
                measure=1,
                voice_a=None,
                voice_b=None,
                kind="empty_score",
                message="The score is empty; it contains no notes.",
                snippet="no notes",
            )
        ]


class ParallelFifthsRule(CheckRule):
    """Detect parallel perfect fifths between adjacent voices."""

    rule_id = "pf5th"
    name = "Parallel fifths"

    def run(self, score: stream.Score, _ctx: CheckerContext) -> list[CheckViolation]:
        """Detect parallel fifths.

        Args:
            score: The score to inspect.
            _ctx: The ctx.

        Returns:
            The run result.
        """
        violations: list[CheckViolation] = []
        order = voice_order(score)
        slices = build_slices(score)
        for voice_a, voice_b in itertools.pairwise(order):
            pairs = pair_sequence(slices, voice_a, voice_b)
            for (a1, b1), (a2, b2) in itertools.pairwise(pairs):
                if (
                    abs(a1.midi - b1.midi) % 12 == _PERFECT_FIFTH
                    and abs(a2.midi - b2.midi) % 12 == _PERFECT_FIFTH
                ):
                    da, db = a2.midi - a1.midi, b2.midi - b1.midi
                    if da != 0 and db != 0 and (da > 0) == (db > 0):
                        violations.append(
                            _violation(
                                self,
                                _slice_for(slices, a2),
                                ViolationData(
                                    kind="parallel_fifth",
                                    message=f"m. {a2.measure}: parallel perfect fifth between {voice_a} and {voice_b}.",
                                    snippet=(
                                        f"{voice_a} {a1.pitch}>{a2.pitch}; "
                                        f"{voice_b} {b1.pitch}>{b2.pitch}"
                                    ),
                                    voice_a=voice_a,
                                    voice_b=voice_b,
                                ),
                            )
                        )
        return violations


class ParallelOctavesRule(CheckRule):
    """Detect parallel perfect octaves or unisons between adjacent voices."""

    rule_id = "po8ve"
    name = "Parallel octaves"

    def run(self, score: stream.Score, _ctx: CheckerContext) -> list[CheckViolation]:
        """Detect parallel octaves.

        Args:
            score: The score to inspect.
            _ctx: The ctx.

        Returns:
            The run result.
        """
        violations: list[CheckViolation] = []
        order = voice_order(score)
        slices = build_slices(score)
        for voice_a, voice_b in itertools.pairwise(order):
            pairs = pair_sequence(slices, voice_a, voice_b)
            for (a1, b1), (a2, b2) in itertools.pairwise(pairs):
                if (
                    abs(a1.midi - b1.midi) % 12 == _PERFECT_OCTAVE
                    and abs(a2.midi - b2.midi) % 12 == _PERFECT_OCTAVE
                ):
                    da, db = a2.midi - a1.midi, b2.midi - b1.midi
                    if da != 0 and db != 0 and (da > 0) == (db > 0):
                        violations.append(
                            _violation(
                                self,
                                _slice_for(slices, a2),
                                ViolationData(
                                    kind="parallel_octave",
                                    message=f"m. {a2.measure}: parallel octave/unison between {voice_a} and {voice_b}.",
                                    snippet=(
                                        f"{voice_a} {a1.pitch}>{a2.pitch}; "
                                        f"{voice_b} {b1.pitch}>{b2.pitch}"
                                    ),
                                    voice_a=voice_a,
                                    voice_b=voice_b,
                                ),
                            )
                        )
        return violations


class HiddenFifthsRule(CheckRule):
    """Detect hidden (exposed) perfect fifths entered by similar motion."""

    rule_id = "hf5th"
    name = "Hidden fifths"

    def run(self, score: stream.Score, _ctx: CheckerContext) -> list[CheckViolation]:
        """Detect hidden fifths.

        Args:
            score: The score to inspect.
            _ctx: The ctx.

        Returns:
            The run result.
        """
        return _hidden_perfect(self, score, _PERFECT_FIFTH, "hidden_fifth", "fifth")


class HiddenOctavesRule(CheckRule):
    """Detect hidden (exposed) perfect octaves entered by similar motion."""

    rule_id = "ho8ve"
    name = "Hidden octaves"

    def run(self, score: stream.Score, _ctx: CheckerContext) -> list[CheckViolation]:
        """Detect hidden octaves.

        Args:
            score: The score to inspect.
            _ctx: The ctx.

        Returns:
            The run result.
        """
        return _hidden_perfect(self, score, _PERFECT_OCTAVE, "hidden_octave", "octave")


def _hidden_perfect(
    rule: CheckRule,
    score: stream.Score,
    target: int,
    kind: str,
    label: str,
) -> list[CheckViolation]:
    """Detect hidden perfect intervals of a target class.

    Args:
        rule: Emitting rule.
        score: Score to inspect.
        target: Perfect interval class (0 for octave, 7 for fifth).
        kind: Machine-readable category.
        label: Chinese label.

    Returns:
        Violations.
    """
    violations: list[CheckViolation] = []
    order = voice_order(score)
    slices = build_slices(score)
    for voice_a, voice_b in itertools.pairwise(order):
        pairs = pair_sequence(slices, voice_a, voice_b)
        for (a1, b1), (a2, b2) in itertools.pairwise(pairs):
            before = abs(a1.midi - b1.midi) % 12
            after = abs(a2.midi - b2.midi) % 12
            da, db = a2.midi - a1.midi, b2.midi - b1.midi
            if (
                after == target
                and before != target
                and da != 0
                and db != 0
                and (da > 0) == (db > 0)
            ):
                violations.append(
                    _violation(
                        rule,
                        _slice_for(slices, a2),
                        ViolationData(
                            kind=kind,
                            message=(
                                f"m. {a2.measure}: {voice_a} and {voice_b} enter a "
                                f"hidden {label} by similar motion."
                            ),
                            snippet=(
                                f"{voice_a} {a1.pitch}->{a2.pitch}, "
                                f"{voice_b} {b1.pitch}->{b2.pitch}"
                            ),
                            voice_a=voice_a,
                            voice_b=voice_b,
                        ),
                    )
                )
    return violations


class VoiceCrossingRule(CheckRule):
    """Detect voice crossing between adjacent voices."""

    rule_id = "crossing"
    name = "Voice crossing"

    def run(self, score: stream.Score, _ctx: CheckerContext) -> list[CheckViolation]:
        """Detect voice crossings.

        Args:
            score: The score to inspect.
            _ctx: The ctx.

        Returns:
            The run result.
        """
        violations: list[CheckViolation] = []
        order = voice_order(score)
        for item in build_slices(score):
            for voice_a, voice_b in itertools.pairwise(order):
                if voice_a in item.events and voice_b in item.events:
                    upper = item.events[voice_a]
                    lower = item.events[voice_b]
                    if upper.midi < lower.midi:
                        violations.append(
                            _violation(
                                self,
                                item,
                                ViolationData(
                                    kind="crossing",
                                    message=(
                                        f"m. {item.measure}: voice crossing between "
                                        f"{voice_a} and {voice_b}."
                                    ),
                                    snippet=f"{voice_a} {upper.pitch}, {voice_b} {lower.pitch}",
                                    voice_a=voice_a,
                                    voice_b=voice_b,
                                ),
                            )
                        )
        return violations


class ExcessiveSpacingRule(CheckRule):
    """Detect adjacent voices spaced wider than the allowed interval."""

    rule_id = "spacing"
    name = "Excessive spacing"

    def run(self, score: stream.Score, ctx: CheckerContext) -> list[CheckViolation]:
        """Detect excessive spacing.

        Args:
            score: The score to inspect.
            ctx: The technique context.

        Returns:
            The run result.
        """
        violations: list[CheckViolation] = []
        order = voice_order(score)
        for item in build_slices(score):
            for voice_a, voice_b in itertools.pairwise(order):
                if voice_a in item.events and voice_b in item.events:
                    upper = item.events[voice_a]
                    lower = item.events[voice_b]
                    if upper.midi - lower.midi > ctx.spacing_limit:
                        violations.append(
                            _violation(
                                self,
                                item,
                                ViolationData(
                                    kind="spacing",
                                    message=(
                                        f"m. {item.measure}: {voice_a} and {voice_b} are "
                                        "more than a tenth apart."
                                    ),
                                    snippet=f"{voice_a} {upper.pitch}, {voice_b} {lower.pitch}",
                                    voice_a=voice_a,
                                    voice_b=voice_b,
                                ),
                            )
                        )
        return violations


class FinalOuterIntervalRule(CheckRule):
    """Check the final outer-voice interval."""

    rule_id = "final_outer"
    name = "Final outer interval"

    def run(self, score: stream.Score, _ctx: CheckerContext) -> list[CheckViolation]:
        """Check the final sonority.

        Args:
            score: The score to inspect.
            _ctx: The ctx.

        Returns:
            The run result.
        """
        order = voice_order(score)
        slices = [
            item for item in build_slices(score) if len(item.events) >= _MIN_VOICES
        ]
        if not slices:
            return []
        last = slices[-1]
        present = [voice for voice in order if voice in last.events]
        top = last.events[present[0]]
        bottom = last.events[present[-1]]
        if abs(top.midi - bottom.midi) % 12 not in {_PERFECT_OCTAVE, _PERFECT_FIFTH}:
            return [
                _violation(
                    self,
                    last,
                    ViolationData(
                        kind="final_interval",
                        message=(
                            f"m. {last.measure}: the final outer-voice interval is not "
                            "a perfect fifth, octave or unison."
                        ),
                        snippet=f"{top.pitch} / {bottom.pitch}",
                        voice_a=present[0],
                        voice_b=present[-1],
                    ),
                )
            ]
        return []


class DominantResolutionRule(CheckRule):
    """Check that dominant seventh chords resolve to their tonic."""

    rule_id = "dom7res"
    name = "Dominant seventh resolution"

    def run(self, score: stream.Score, _ctx: CheckerContext) -> list[CheckViolation]:
        """Check dominant seventh resolutions.

        Args:
            score: The score to inspect.
            _ctx: The ctx.

        Returns:
            The run result.
        """
        violations: list[CheckViolation] = []
        slices = build_slices(score)
        for index, item in enumerate(slices[:-1]):
            pcs = pitch_classes(item.events)
            root = _dominant_root(pcs)
            if root is None:
                continue
            next_pcs = pitch_classes(slices[index + 1].events)
            if (root + 5) % 12 not in next_pcs:
                violations.append(
                    _violation(
                        self,
                        item,
                        ViolationData(
                            kind="dominant_unresolved",
                            message=(
                                f"m. {item.measure}: the dominant seventh does not "
                                "resolve to its tonic."
                            ),
                            snippet="V7 -> ?",
                        ),
                    )
                )
        return violations


def _dominant_root(pcs: set[int]) -> int | None:
    """Return the root of a dominant seventh chord in a pitch-class set.

    Args:
        pcs: Pitch classes present.

    Returns:
        The root pitch class, or ``None`` when the set is not a dominant seventh.
    """
    for root in range(12):
        needed = {root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12}
        if needed <= pcs:
            return root
    return None


class LeadingToneRule(CheckRule):
    """Check that leading tones resolve upward to the tonic."""

    rule_id = "leading"
    name = "Leading-tone resolution"

    def run(self, score: stream.Score, ctx: CheckerContext) -> list[CheckViolation]:
        """Check leading-tone resolutions.

        Args:
            score: The score to inspect.
            ctx: The technique context.

        Returns:
            The run result.
        """
        violations: list[CheckViolation] = []
        tonic = tonic_pc(ctx.tonic)
        leading = (tonic + 11) % 12
        slices = build_slices(score)
        by_voice: dict[str, list[VoiceEvent]] = {}
        for item in slices:
            for voice, event in item.events.items():
                by_voice.setdefault(voice, []).append(event)
        for voice, events in by_voice.items():
            for first, second in itertools.pairwise(events):
                if first.midi % 12 == leading and not (
                    second.midi % 12 == tonic and second.midi > first.midi
                ):
                    violations.append(
                        CheckViolation(
                            rule_id=self.rule_id,
                            severity=self.default_severity,
                            measure=first.measure,
                            voice_a=voice,
                            voice_b=None,
                            kind="leading_unresolved",
                            message=(
                                f"m. {first.measure}: the leading tone in {voice} does "
                                "not resolve up to the tonic."
                            ),
                            snippet=f"{first.pitch}->{second.pitch}",
                        )
                    )
        return violations


class ChordOmissionRule(CheckRule):
    """Check chord handling, notably doubling of the leading tone."""

    rule_id = "omission"
    name = "Leading-tone doubling"

    def run(self, score: stream.Score, ctx: CheckerContext) -> list[CheckViolation]:
        """Detect doubled leading tones.

        Args:
            score: The score to inspect.
            ctx: The technique context.

        Returns:
            The run result.
        """
        violations: list[CheckViolation] = []
        tonic = tonic_pc(ctx.tonic)
        leading = (tonic + 11) % 12
        for item in build_slices(score):
            voices = [
                voice
                for voice, event in item.events.items()
                if event.midi % 12 == leading
            ]
            if len(voices) >= _MIN_VOICES:
                violations.append(
                    _violation(
                        self,
                        item,
                        ViolationData(
                            kind="doubled_leading_tone",
                            message=(
                                f"m. {item.measure}: the leading tone is doubled "
                                f"({', '.join(voices)})."
                            ),
                            snippet=" / ".join(
                                item.events[voice].pitch for voice in voices
                            ),
                            voice_a=voices[0],
                            voice_b=voices[1],
                        ),
                    )
                )
        return violations


class DiminishedIntervalRule(CheckRule):
    """Check that diminished melodic leaps resolve by step."""

    rule_id = "diminterval"
    name = "Diminished leap"

    def run(self, score: stream.Score, _ctx: CheckerContext) -> list[CheckViolation]:
        """Check diminished melodic intervals.

        Args:
            score: The score to inspect.
            _ctx: The ctx.

        Returns:
            The run result.
        """
        violations: list[CheckViolation] = []
        slices = build_slices(score)
        by_voice: dict[str, list[VoiceEvent]] = {}
        for item in slices:
            for voice, event in item.events.items():
                by_voice.setdefault(voice, []).append(event)
        for voice, events in by_voice.items():
            for first, second, third in zip(
                events, events[1:], events[2:], strict=False
            ):
                leap = second.midi - first.midi
                if abs(leap) in {_TRITONE, _DIMINISHED_SEVENTH}:
                    resolution = third.midi - second.midi
                    resolved = abs(resolution) <= _STEP and (resolution > 0) != (
                        leap > 0
                    )
                    if not resolved:
                        violations.append(
                            CheckViolation(
                                rule_id=self.rule_id,
                                severity=self.default_severity,
                                measure=second.measure,
                                voice_a=voice,
                                voice_b=None,
                                kind="diminished_leap",
                                message=(
                                    f"m. {second.measure}: the diminished leap in "
                                    f"{voice} does not resolve."
                                ),
                                snippet=f"{first.pitch}->{second.pitch}->{third.pitch}",
                            )
                        )
        return violations


class TonalUnityRule(CheckRule):
    """Check that structural positions end in their expected key."""

    rule_id = "tonality"
    name = "Tonal unity"

    def run(self, score: stream.Score, ctx: CheckerContext) -> list[CheckViolation]:
        """Check tonal unity.

        Args:
            score: The score to inspect.
            ctx: The technique context.

        Returns:
            The run result.
        """
        violations: list[CheckViolation] = []
        slices = build_slices(score)
        if not slices or not ctx.expectations:
            return []
        for expectation in ctx.expectations:
            if expectation.key is None:
                continue
            expected = tonic_pc(expectation.key)
            window = [
                item
                for item in slices
                if expectation.measure - 1 <= item.measure <= expectation.measure
            ]
            if not window:
                window = [slices[-1]]
            if not any(expected in pitch_classes(item.events) for item in window):
                violations.append(
                    _violation(
                        self,
                        window[-1],
                        ViolationData(
                            kind="tonal_unity",
                            message=(
                                f"m. {window[-1].measure}: structural position "
                                f"{expectation.label or expectation.measure} is not in "
                                f"the expected key {expectation.key}."
                            ),
                            snippet=f"expect tonic pc {expected}",
                        ),
                    )
                )
        return violations


class CadenceTypeRule(CheckRule):
    """Check that structural endpoints carry an authentic cadence."""

    rule_id = "cadence"
    name = "Cadence"

    def run(self, score: stream.Score, ctx: CheckerContext) -> list[CheckViolation]:
        """Check cadence types.

        Args:
            score: The score to inspect.
            ctx: The technique context.

        Returns:
            The run result.
        """
        slices = build_slices(score)
        points = [item for item in ctx.expectations if item.cadence]
        if len(slices) < _MIN_CHORDS or not points:
            return []
        violations: list[CheckViolation] = []
        tonic = tonic_pc(ctx.tonic)
        for point in points:
            relevant = [item for item in slices if item.measure <= point.measure]
            if len(relevant) < _MIN_CHORDS:
                relevant = slices
            last, previous = relevant[-1], relevant[-2]
            last_pcs = pitch_classes(last.events)
            previous_pcs = pitch_classes(previous.events)
            if tonic not in last_pcs or (tonic + 7) % 12 not in previous_pcs:
                violations.append(
                    _violation(
                        self,
                        last,
                        ViolationData(
                            kind="cadence",
                            message=(
                                f"m. {last.measure}: the structural endpoint is not an "
                                "authentic cadence V->I."
                            ),
                            snippet="V->I expected",
                        ),
                    )
                )
        return violations


class FixedVoiceCountRule(CheckRule):
    """Check that the voice count is stable from start to end."""

    rule_id = "voices"
    name = "Fixed voice count"

    def run(self, score: stream.Score, ctx: CheckerContext) -> list[CheckViolation]:
        """Check voice-count stability.

        Args:
            score: The score to inspect.
            ctx: The technique context.

        Returns:
            The run result.
        """
        if not ctx.enforce_voice_count:
            return []
        slices = build_slices(score)
        if not slices:
            return []
        last_measure = slices[-1].measure
        if last_measure < _MIN_OPENING_CLOSING:
            return []
        window = max(1, min(8, last_measure // 2))
        opening = {
            voice for item in slices if item.measure <= window for voice in item.events
        }
        closing = {
            voice
            for item in slices
            if item.measure > last_measure - window
            for voice in item.events
        }
        if opening and opening > closing:
            return [
                _violation(
                    self,
                    slices[-1],
                    ViolationData(
                        kind="voice_count",
                        message=(
                            "Voices sound at the opening but are lost by the end "
                            f"({', '.join(sorted(opening - closing))})."
                        ),
                        snippet=f"open={sorted(opening)} close={sorted(closing)}",
                    ),
                )
            ]
        return []


def _slice_for(slices: list[Slice], event: VoiceEvent) -> Slice:
    """Return the slice containing an event.

    Args:
        slices: All slices.
        event: Event to locate.

    Returns:
        The slice whose offset matches the event.
    """
    for item in slices:
        if abs(item.offset - event.offset) < _EPSILON:
            return item
    return slices[0]


BUILTIN_RULES: tuple[type[CheckRule], ...] = (
    EmptyScoreRule,
    ParallelFifthsRule,
    ParallelOctavesRule,
    HiddenFifthsRule,
    HiddenOctavesRule,
    VoiceCrossingRule,
    ExcessiveSpacingRule,
    FinalOuterIntervalRule,
    DominantResolutionRule,
    LeadingToneRule,
    ChordOmissionRule,
    DiminishedIntervalRule,
    TonalUnityRule,
    CadenceTypeRule,
    FixedVoiceCountRule,
)

#: Precise, actionable statement of each rule's limit, handed verbatim to the
#: composer AI in its first prompt.  Keys must cover every rule in
#: :data:`BUILTIN_RULES` (enforced by a test).
RULE_CONSTRAINTS: dict[str, str] = {
    "empty": "整份乐章至少要有一个音符;完全空白的乐章直接判失败。先 submit_theme 建立主题。",
    "pf5th": "检测相邻两声部连续同向移动后都形成纯五度(平行五度)。"
    "修正:改变其中一个声部的方向或音程,不要连续同向构成纯五度。",
    "po8ve": "检测相邻两声部连续同向移动后都形成纯八度或同度(平行八/一度)。"
    "修正:让其中一个声部反向或换音。",
    "hf5th": "检测相邻两声部由同向进行进入纯五度(隐伏五度,含外声部):前一音程不是纯五度、"
    "后一音程是纯五度且两声部同向。修正:改为级进或反向进入。",
    "ho8ve": "检测相邻两声部由同向进行进入纯八度(隐伏八度,含外声部),条件同隐伏五度。"
    "修正:改级进/反向。",
    "crossing": "检测声部交叉:上方声部的音高低于相邻的下方声部。"
    "修正:调整音高,使上声部始终不低于下声部。",
    "spacing": "检测相邻声部间距超过十度(>16 个半音)。"
    "修正:把上方声部降低或下方声部升高,任何相邻两声部间距都要 ≤ 十度。",
    "final_outer": "检测最后一个多声部纵合的外声部音程;必须是纯五度、纯八度或同度。"
    "修正:让收尾的外声部构成纯五/八/同度。",
    "dom7res": "检测属七和弦(根音+大三度+纯五度+小七度)是否解决:"
    "下一个纵合必须包含其根音上方纯五度的音。修正:把属七解决到主和弦或临时主和弦。",
    "leading": "检测导音(主音下方小二度)是否上行小二度解决到主音;未解决即违规。",
    "omission": "检测导音是否被重复:同一纵合里导音出现两次及以上即违规。修正:导音只出现一次。",
    "diminterval": "检测旋律中的三全音或减七度跳进是否反向级进(≤2 个半音)解决;"
    "未解决即违规。修正:跳进后反向级进。",
    "tonality": "检测结构位置(尤其乐章终止处)是否出现预期主音级;没有即违规。",
    "cadence": "检测结构终点是否形成正格终止 V→I:末纵合含主音,"
    "且前一纵合含属音(主音上方纯五度)。修正:写出 V→I 收束。",
    "voices": "检测开头的声部是否在结尾凭空消失(结尾声部应是开头声部的子集);"
    "声部轮换或结尾新加入不算违规。",
}
