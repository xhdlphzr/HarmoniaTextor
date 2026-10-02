# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Registry combining the built-in and user-defined style kits."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from harmoniatextor.styles.base import StyleKit
from harmoniatextor.styles.builtin import (
    ALL_RULE_IDS,
    ALL_TECHNIQUE_IDS,
    BUILTIN_KITS,
    DEFAULT_STYLE_ID,
)
from harmoniatextor.styles.store import StyleKitStore

__all__ = ["StyleRegistry"]


class StyleRegistry:
    """Built-in style kits plus the user's custom kits.

    Attributes:
        store: Persistence for custom kits.
        builtins: Built-in kits keyed by id.
    """

    def __init__(
        self,
        store: StyleKitStore | None = None,
        builtins: dict[str, StyleKit] | None = None,
    ) -> None:
        """Initialise the registry.

        Args:
            store: Custom-kit persistence; defaults to the config directory.
            builtins: Built-in kits; defaults to the shipped presets.
        """
        self.store = store if store is not None else StyleKitStore()
        self.builtins: dict[str, StyleKit] = dict(
            builtins if builtins is not None else BUILTIN_KITS
        )

    def all(self) -> list[StyleKit]:
        """Return the built-in kits then the custom kits by name.

        Returns:
            Every known kit.
        """
        custom = self.store.load()
        return [*self.builtins.values(), *sorted(custom.values(), key=lambda kit: kit.name)]

    def get(self, kit_id: str) -> StyleKit:
        """Return a kit by id.

        Args:
            kit_id: Kit identifier.

        Returns:
            The kit.

        Raises:
            KeyError: When the id is unknown.
        """
        if kit_id in self.builtins:
            return self.builtins[kit_id]
        custom = self.store.load()
        if kit_id in custom:
            return custom[kit_id]
        raise KeyError(kit_id)

    def resolve(self, kit_id: str | None) -> StyleKit:
        """Return a kit by id, falling back to the default built-in.

        Args:
            kit_id: Kit identifier, or ``None``.

        Returns:
            The requested kit, or the default Baroque kit.
        """
        if kit_id:
            try:
                return self.get(kit_id)
            except KeyError:
                pass
        return self.builtins[DEFAULT_STYLE_ID]

    def create(self, name: str, rules: Iterable[str], techniques: Iterable[str]) -> StyleKit:
        """Create and persist a custom kit.

        Args:
            name: Display name.
            rules: Enabled rule identifiers.
            techniques: Available technique identifiers.

        Returns:
            The created kit.

        Raises:
            ValueError: When the name is empty or an id is unknown.
        """
        clean = name.strip()
        if not clean:
            raise ValueError("style kit name is required")
        rule_set = frozenset(rules)
        technique_set = frozenset(techniques)
        self._validate(rule_set, technique_set)
        kit = StyleKit(
            id=f"s-{uuid4().hex[:8]}",
            name=clean,
            brief="",
            rules=rule_set,
            techniques=technique_set,
        )
        kits = self.store.load()
        kits[kit.id] = kit
        self.store.save(kits)
        return kit

    def rename(self, kit_id: str, name: str) -> StyleKit:
        """Rename a custom kit.

        Args:
            kit_id: Custom kit identifier.
            name: New display name.

        Returns:
            The renamed kit.

        Raises:
            ValueError: When the kit is built-in or the name is empty.
            KeyError: When the custom kit does not exist.
        """
        if kit_id in self.builtins:
            raise ValueError("cannot rename a built-in style kit")
        clean = name.strip()
        if not clean:
            raise ValueError("style kit name is required")
        kits = self.store.load()
        if kit_id not in kits:
            raise KeyError(kit_id)
        old = kits[kit_id]
        renamed = StyleKit(
            id=old.id,
            name=clean,
            brief=old.brief,
            rules=old.rules,
            techniques=old.techniques,
        )
        kits[kit_id] = renamed
        self.store.save(kits)
        return renamed

    def delete(self, kit_id: str) -> None:
        """Delete a custom kit.

        Args:
            kit_id: Custom kit identifier.

        Raises:
            ValueError: When the kit is built-in.
            KeyError: When the custom kit does not exist.
        """
        if kit_id in self.builtins:
            raise ValueError("cannot delete a built-in style kit")
        kits = self.store.load()
        if kit_id not in kits:
            raise KeyError(kit_id)
        del kits[kit_id]
        self.store.save(kits)

    @staticmethod
    def _validate(rules: frozenset[str], techniques: frozenset[str]) -> None:
        """Reject unknown rule or technique identifiers.

        Args:
            rules: Enabled rule identifiers.
            techniques: Available technique identifiers.

        Raises:
            ValueError: When an identifier is unknown.
        """
        unknown_rules = rules - ALL_RULE_IDS
        if unknown_rules:
            raise ValueError(f"unknown rules: {sorted(unknown_rules)}")
        unknown_techniques = techniques - ALL_TECHNIQUE_IDS
        if unknown_techniques:
            raise ValueError(f"unknown techniques: {sorted(unknown_techniques)}")
