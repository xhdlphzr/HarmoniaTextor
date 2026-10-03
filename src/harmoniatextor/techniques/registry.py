# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Registry for technique packs.

All techniques are registered under stable identifiers.  The registry is the
single source of truth used by the agent tools, the HTTP API and the UI.
"""

from __future__ import annotations

from typing import Any

from harmoniatextor.techniques.base import Technique
from harmoniatextor.techniques.exemption import FreeVoiceLeadingTechnique
from harmoniatextor.techniques.harmonic import (
    ChromaticHarmonyTechnique,
    ChromaticModulationTechnique,
    ColorChordTechnique,
    DiminishedSeventhTechnique,
    DominantSeventhTechnique,
    ExtendedHarmonyTechnique,
    FunctionalCycleTechnique,
    HarmonicSequenceTechnique,
    ModalHarmonyTechnique,
    ModulationBridgeTechnique,
    WholeToneTechnique,
)
from harmoniatextor.techniques.melodic import (
    AugmentationTechnique,
    DiminutionTechnique,
    ImitationTechnique,
    InversionTechnique,
    RetrogradeTechnique,
    SequenceTechnique,
    TranspositionTechnique,
    VoiceExchangeTechnique,
)
from harmoniatextor.techniques.rhythmic import (
    CounterRhythmTechnique,
    RhythmicIndependenceTechnique,
    RubatoTechnique,
    SyncopationTechnique,
    VoiceMotionTechnique,
)
from harmoniatextor.techniques.structural import (
    DevelopmentTechnique,
    ExpositionTechnique,
    PedalPointTechnique,
    PedalToneTechnique,
    RecapitulationTechnique,
    RondoTechnique,
    StrettoTechnique,
)
from harmoniatextor.techniques.texture import (
    AlbertiBassTechnique,
    BrokenChordTechnique,
    ParallelChordsTechnique,
    PlaningTechnique,
)

__all__ = ["TechniqueRegistry", "build_default_registry"]

_DEFAULT_TECHNIQUES: tuple[type[Technique[Any]], ...] = (
    ImitationTechnique,
    InversionTechnique,
    RetrogradeTechnique,
    AugmentationTechnique,
    DiminutionTechnique,
    TranspositionTechnique,
    SequenceTechnique,
    VoiceExchangeTechnique,
    ExpositionTechnique,
    DevelopmentTechnique,
    RecapitulationTechnique,
    RondoTechnique,
    StrettoTechnique,
    PedalPointTechnique,
    PedalToneTechnique,
    FunctionalCycleTechnique,
    DominantSeventhTechnique,
    DiminishedSeventhTechnique,
    HarmonicSequenceTechnique,
    ChromaticHarmonyTechnique,
    ModulationBridgeTechnique,
    SyncopationTechnique,
    RhythmicIndependenceTechnique,
    CounterRhythmTechnique,
    VoiceMotionTechnique,
    AlbertiBassTechnique,
    BrokenChordTechnique,
    ParallelChordsTechnique,
    PlaningTechnique,
    ChromaticModulationTechnique,
    ExtendedHarmonyTechnique,
    ModalHarmonyTechnique,
    ColorChordTechnique,
    WholeToneTechnique,
    RubatoTechnique,
    FreeVoiceLeadingTechnique,
)


class TechniqueRegistry:
    """A registry of technique instances keyed by identifier.

    Attributes:
        _items: Internal mapping from identifier to technique instance.
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._items: dict[str, Technique[Any]] = {}

    def register(self, technique: Technique[Any]) -> None:
        """Register a technique.

        Args:
            technique: The technique to register.

        Raises:
            ValueError: When the identifier is already registered.
        """
        if technique.id in self._items:
            raise ValueError(f"duplicate technique id: {technique.id}")
        self._items[technique.id] = technique

    def get(self, technique_id: str) -> Technique[Any]:
        """Return a technique by identifier.

        Args:
            technique_id: Technique identifier.

        Returns:
            The technique instance.

        Raises:
            KeyError: When the identifier is unknown.
        """
        if technique_id not in self._items:
            raise KeyError(technique_id)
        return self._items[technique_id]

    def ids(self) -> list[str]:
        """Return all registered technique identifiers.

        Returns:
            Identifiers in registration order.
        """
        return list(self._items)

    def all(self) -> list[Technique[Any]]:
        """Return all registered techniques.

        Returns:
            Techniques in registration order.
        """
        return list(self._items.values())

    def __contains__(self, technique_id: object) -> bool:
        """Return whether an identifier is registered.

        Args:
            technique_id: The technique id.

        Returns:
            Whether the condition holds.
        """
        return technique_id in self._items

    def __len__(self) -> int:
        """Return the number of registered techniques.

        Returns:
            The resulting number.
        """
        return len(self._items)


def build_default_registry() -> TechniqueRegistry:
    """Build a registry containing all 36 built-in techniques.

    Returns:
        A populated :class:`TechniqueRegistry`.
    """
    registry = TechniqueRegistry()
    for technique_type in _DEFAULT_TECHNIQUES:
        registry.register(technique_type())
    return registry
