# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Core domain entities for compositions, themes, revisions and checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from harmoniatextor.domain.enums import Severity, ToolKind, WorkStatus
from harmoniatextor.domain.key import KeySpec

__all__ = [
    "CheckReport",
    "CheckViolation",
    "Movement",
    "Revision",
    "StyleSelection",
    "Theme",
    "ThemeNote",
    "Work",
    "theme_fingerprint",
]


@dataclass(frozen=True, slots=True)
class ThemeNote:
    """A single note of a theme melody.

    Attributes:
        pitch: Scientific pitch name such as ``"C4"``.
        quarter_length: Duration in quarter notes.
    """

    pitch: str
    quarter_length: float


def theme_fingerprint(notes: list[ThemeNote]) -> str:
    """Compute a stable fingerprint for a theme melody.

    Args:
        notes: The theme notes in performance order.

    Returns:
        A short hexadecimal digest of the pitch contour and rhythm.
    """
    payload = "|".join(f"{note.pitch}:{note.quarter_length}" for note in notes)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class Theme:
    """A referenced melody material with a globally increasing identifier.

    Attributes:
        id: Globally increasing theme number.
        movement_id: Owning movement.
        voice: Voice slot the theme was submitted in.
        start_measure: First measure of the theme.
        notes: The theme melody.
        created_revision: Revision id that introduced the theme.
        fingerprint: Stable digest of the melody.
    """

    id: int
    movement_id: str
    voice: str
    start_measure: int
    notes: list[ThemeNote]
    created_revision: str
    fingerprint: str = ""

    def __post_init__(self) -> None:
        """Fill the fingerprint when it was not supplied."""
        if not self.fingerprint:
            self.fingerprint = theme_fingerprint(self.notes)


@dataclass(frozen=True, slots=True)
class CheckViolation:
    """A single symbolic-layer rule violation.

    Attributes:
        rule_id: Stable rule identifier such as ``"pf5th"``.
        severity: Error or warning.
        measure: Measure number where the violation was detected.
        voice_a: First involved voice, if any.
        voice_b: Second involved voice, if any.
        kind: Machine-readable category.
        message: Human-readable English explanation.
        snippet: Short musical description of the offending spot.
    """

    rule_id: str
    severity: Severity
    measure: int
    voice_a: str | None
    voice_b: str | None
    kind: str
    message: str
    snippet: str


@dataclass(slots=True)
class CheckReport:
    """The aggregate result of running the checker over a score.

    Attributes:
        violations: All detected violations in score order.
    """

    violations: list[CheckViolation] = field(default_factory=list)

    @property
    def errors(self) -> list[CheckViolation]:
        """Return only error-severity violations.

        Returns:
            The errors result.
        """
        return [item for item in self.violations if item.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[CheckViolation]:
        """Return only warning-severity violations.

        Returns:
            The warnings result.
        """
        return [item for item in self.violations if item.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """Whether the score passed (no error-severity violations).

        Returns:
            Whether the condition holds.
        """
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-friendly dictionary.

        Returns:
            The resulting mapping.
        """
        return {
            "ok": self.ok,
            "violations": [
                {
                    "rule_id": item.rule_id,
                    "severity": item.severity.value,
                    "measure": item.measure,
                    "voice_a": item.voice_a,
                    "voice_b": item.voice_b,
                    "kind": item.kind,
                    "message": item.message,
                    "snippet": item.snippet,
                }
                for item in self.violations
            ],
        }


@dataclass(slots=True)
class Revision:
    """A full-score snapshot produced by a single tool invocation.

    Attributes:
        id: Revision identifier.
        seq: Monotonic sequence number within the movement.
        movement_id: Owning movement.
        full_xml: Complete MusicXML of the score.
        source: What produced the revision.
        technique: Technique identifier when applicable.
        params: Normalised parameters when applicable.
        check: Check report for the revision.
        ok: Whether the revision passed the checker.
        parent_seq: Sequence number this revision descends from.
    """

    id: str
    seq: int
    movement_id: str
    full_xml: str
    source: ToolKind
    technique: str | None = None
    params: dict[str, Any] | None = None
    check: CheckReport | None = None
    ok: bool = False
    parent_seq: int | None = None


@dataclass(slots=True)
class Movement:
    """A single movement backed by one MusicXML score.

    Attributes:
        id: Movement identifier such as ``"m01"``.
        work_id: Owning work.
        name: Display name such as ``"I. Allegro"``.
        time_signature: Time signature string.
        key: Movement key.
        tempo: Tempo in quarter notes per minute.
        voice_profile: Identifier of the texture profile in use.
        canonical_revision: Revision id of the current authoritative score.
        prompt: The Step 1 prompt describing what this movement must contain.
    """

    id: str
    work_id: str
    name: str
    time_signature: str
    key: KeySpec
    tempo: int
    voice_profile: str
    canonical_revision: str = ""
    prompt: str = ""


@dataclass(frozen=True, slots=True)
class StyleSelection:
    """A frozen snapshot of the style kit chosen for a work.

    Attributes:
        id: Kit identifier the snapshot came from.
        name: Display name at creation time.
        brief: Style description captured at creation time.
        rules: Enabled rule identifiers.
        techniques: Available technique identifiers.
    """

    id: str
    name: str
    brief: str = ""
    rules: frozenset[str] = field(default_factory=frozenset)
    techniques: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the selection.

        Returns:
            A JSON-friendly dictionary with sorted lists.
        """
        return {
            "id": self.id,
            "name": self.name,
            "brief": self.brief,
            "rules": sorted(self.rules),
            "techniques": sorted(self.techniques),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleSelection:
        """Rebuild a selection from stored primitives.

        Args:
            data: A mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed selection.
        """
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            brief=str(data.get("brief", "")),
            rules=frozenset(str(item) for item in data.get("rules", [])),
            techniques=frozenset(str(item) for item in data.get("techniques", [])),
        )


@dataclass(slots=True)
class Work:
    """A composition session containing one or more movements.

    Attributes:
        id: Work identifier.
        title: Display title.
        genre: Genre identifier.
        tonic: Home key.
        movements: Ordered movements.
        status: Lifecycle status.
        style: Frozen style-kit snapshot chosen at creation.
        prompt: The user's original composition prompt, or an empty string for
            works created before this was recorded.
        created_at: ISO timestamp.
        updated_at: ISO timestamp.
    """

    id: str
    title: str
    genre: str
    tonic: KeySpec
    movements: list[Movement] = field(default_factory=list)
    status: WorkStatus = WorkStatus.DRAFT
    style: StyleSelection | None = None
    prompt: str = ""
    created_at: str = ""
    updated_at: str = ""
