# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Technique abstraction shared by all classical composition technique packs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from music21 import stream
from pydantic import BaseModel

from harmoniatextor.domain.models import Theme

__all__ = [
    "Technique",
    "TechniqueCategory",
    "TechniqueContext",
    "TechniqueError",
    "TechniqueResult",
]


class TechniqueCategory(StrEnum):
    """Category grouping for techniques."""

    MELODIC = "melodic"
    STRUCTURAL = "structural"
    HARMONIC = "harmonic"
    RHYTHMIC = "rhythmic"
    TEXTURE = "texture"


class TechniqueError(Exception):
    """Raised when a technique cannot be applied.

    Attributes:
        code: Stable machine-readable error code.
        message: Human readable explanation.
    """

    def __init__(self, code: str, message: str) -> None:
        """Initialise the error.

        Args:
            code: Stable error code such as ``"THEME_NOT_FOUND"``.
            message: Human readable explanation.
        """
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class TechniqueContext:
    """Everything a technique needs to transform a score.

    Attributes:
        score: The score being edited.
        themes: Theme registry keyed by theme number.
        genre: Active genre identifier.
        voice_profile: Active texture profile identifier.
    """

    score: stream.Score
    themes: Mapping[int, Theme]
    genre: str = "plain"
    voice_profile: str = "four_part"


@dataclass(slots=True)
class TechniqueResult:
    """Result of applying a technique.

    Attributes:
        score: The resulting score.
        warnings: Non-fatal notes produced during the transformation.
    """

    score: stream.Score
    warnings: list[str] = field(default_factory=list)


class Technique[P: BaseModel](ABC):
    """Base class for a single composition technique.

    Attributes:
        id: Stable technique identifier.
        name: Display name.
        category: Technique category.
        summary: One-line description used in prompts.
        params_model: Pydantic model describing the technique parameters.
    """

    id: str
    name: str
    category: TechniqueCategory
    summary: str
    params_model: type[P]
    #: Rules this technique waives (for ``free_voice_leading``).  Empty for
    #: ordinary techniques that only transform the score.
    exempts: frozenset[str] = frozenset()

    @abstractmethod
    def apply(self, ctx: TechniqueContext, params: P) -> TechniqueResult:
        """Apply the technique to a score.

        Args:
            ctx: The technique context.
            params: Validated parameters.

        Returns:
            The transformed score and any warnings.
        """

    def schema(self) -> dict[str, Any]:
        """Return the JSON schema of the technique parameters.

        Returns:
            A JSON-schema dictionary.
        """
        return self.params_model.model_json_schema()
