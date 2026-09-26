# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Genre abstraction for the pluggable multi-genre framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from harmoniatextor.checker.context import CheckerContext, StructuralExpectation
from harmoniatextor.checker.profile import RuleSetting, ValidationProfile
from harmoniatextor.domain.enums import Severity
from harmoniatextor.domain.key import parse_key
from harmoniatextor.domain.models import Movement
from harmoniatextor.techniques.registry import build_default_registry

__all__ = ["ALL_TECHNIQUES", "Genre", "MovementSpec", "default_profile", "relaxed_profile"]

ALL_TECHNIQUES: frozenset[str] = frozenset(build_default_registry().ids())


@dataclass(frozen=True, slots=True)
class MovementSpec:
    """Template describing one movement of a genre.

    Attributes:
        name: Display name such as ``"I. Allegro"``.
        time_signature: Time signature.
        tempo: Tempo in quarter notes per minute.
        key: Absolute key override; when ``None`` the work tonic is used.
        voice_profile: Texture profile identifier.
    """

    name: str
    time_signature: str = "4/4"
    tempo: int = 96
    key: str | None = None
    voice_profile: str = "four_part"


class Genre(ABC):
    """Base class for a pluggable genre.

    Attributes:
        id: Stable genre identifier.
        display_name: Display name.
        movement_specs: Movement templates.
        allowed_techniques: Technique identifiers usable in this genre.
        profile: Validation profile applied to the symbolic checker.
    """

    id: str
    display_name: str
    movement_specs: tuple[MovementSpec, ...]
    allowed_techniques: frozenset[str]
    profile: ValidationProfile

    def initialize_work(self, work_id: str, tonic: str) -> list[Movement]:
        """Create the movement skeleton for a new work.

        Args:
            work_id: Owning work identifier.
            tonic: Work tonic key.

        Returns:
            The initial movements.
        """
        movements: list[Movement] = []
        for index, spec in enumerate(self.movement_specs, start=1):
            movements.append(
                Movement(
                    id=f"m{index:02d}",
                    work_id=work_id,
                    name=spec.name,
                    time_signature=spec.time_signature,
                    key=parse_key(spec.key or tonic),
                    tempo=spec.tempo,
                    voice_profile=spec.voice_profile,
                )
            )
        return movements

    @abstractmethod
    def checker_context(
        self,
        tonic: str,
        measure_count: int,
        *,
        complete: bool = False,
    ) -> CheckerContext:
        """Build the checker context for a movement.

        Args:
            tonic: Movement tonic key.
            measure_count: Number of measures in the movement.
            complete: Whether the movement is being finalised.  Tonal and
                cadence expectations are only attached when ``complete`` is
                true, because the current last measure of a draft is not a
                structural endpoint.

        Returns:
            The checker context.
        """

    def _final_context(
        self,
        tonic: str,
        measure_count: int,
        *,
        enforce: bool,
        complete: bool,
    ) -> CheckerContext:
        """Build a context anchored on the final measure when complete.

        Args:
            tonic: Movement tonic key.
            measure_count: Number of measures in the movement.
            enforce: Whether the fixed voice count rule applies.
            complete: Whether the movement is being finalised.

        Returns:
            A checker context.
        """
        expectations: list[StructuralExpectation] = []
        if complete:
            expectations.append(
                StructuralExpectation(
                    measure=max(1, measure_count),
                    key=tonic,
                    cadence=True,
                    label="movement end",
                )
            )
        return CheckerContext(
            genre=self.id,
            tonic=tonic,
            expectations=expectations,
            enforce_voice_count=enforce,
        )


def default_profile() -> ValidationProfile:
    """Return the strict default profile with every rule enabled.

    Returns:
        An empty profile, which enables all rules at their default severity.
    """
    return ValidationProfile()


def relaxed_profile() -> ValidationProfile:
    """Return the multi-movement profile.

    Tonal unity and cadence are evaluated only at the movement end, the fixed
    voice count rule is not enforced, and the strict leading-tone and doubling
    rules are downgraded to warnings so that orchestral textures remain
    workable.  Every exemption mirrors the structural reasons documented in the
    project book.

    Returns:
        A validation profile for sonata, concerto and symphony.
    """
    return ValidationProfile(
        settings={
            "leading": RuleSetting(enabled=True, severity=Severity.WARNING),
            "omission": RuleSetting(enabled=True, severity=Severity.WARNING),
        }
    )
