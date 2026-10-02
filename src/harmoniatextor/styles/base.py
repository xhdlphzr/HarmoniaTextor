# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""The style-kit abstraction: a named set of rules and techniques."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["StyleKit"]


@dataclass(frozen=True, slots=True)
class StyleKit:
    """A named selection of symbolic rules and composition techniques.

    Attributes:
        id: Stable identifier (``"baroque"`` or ``"s-xxxxxxxx"`` for custom kits).
        name: Display name.
        brief: A short prose description injected into the AI prompts.
        rules: Enabled rule identifiers.
        techniques: Available technique identifiers.
        builtin: Whether this is a built-in kit that cannot be edited.
    """

    id: str
    name: str
    brief: str
    rules: frozenset[str] = field(default_factory=frozenset)
    techniques: frozenset[str] = field(default_factory=frozenset)
    builtin: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise the kit to JSON-friendly primitives.

        Returns:
            A dictionary with sorted rule and technique lists.
        """
        return {
            "id": self.id,
            "name": self.name,
            "brief": self.brief,
            "rules": sorted(self.rules),
            "techniques": sorted(self.techniques),
            "builtin": self.builtin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleKit:
        """Rebuild a kit from stored primitives.

        Args:
            data: A mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed kit.
        """
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            brief=str(data.get("brief", "")),
            rules=frozenset(str(item) for item in data.get("rules", [])),
            techniques=frozenset(str(item) for item in data.get("techniques", [])),
            builtin=bool(data.get("builtin", False)),
        )
