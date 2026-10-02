# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for styles.registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from harmoniatextor.styles.registry import StyleRegistry
from harmoniatextor.styles.store import StyleKitStore


def _registry(tmp_path: Path) -> StyleRegistry:
    """Return a registry backed by a temporary store."""
    return StyleRegistry(StyleKitStore(tmp_path))


class TestStyleRegistry:
    """Built-in plus custom kit management."""

    def test_all(self, tmp_path: Path) -> None:
        """Both built-in and custom kits are listed."""
        registry = _registry(tmp_path)
        registry.create("Mine", ["empty"], ["imitation"])
        names = [kit.name for kit in registry.all()]
        assert "巴洛克" in names
        assert "Mine" in names

    def test_get_builtin(self, tmp_path: Path) -> None:
        """A built-in kit is returned by id."""
        assert _registry(tmp_path).get("baroque").builtin is True

    def test_get_custom(self, tmp_path: Path) -> None:
        """A custom kit is returned by id."""
        registry = _registry(tmp_path)
        kit = registry.create("Mine", ["empty"], ["imitation"])
        assert registry.get(kit.id).name == "Mine"

    def test_get_unknown(self, tmp_path: Path) -> None:
        """An unknown id raises a key error."""
        with pytest.raises(KeyError):
            _registry(tmp_path).get("nope")

    def test_resolve_existing(self, tmp_path: Path) -> None:
        """A known id resolves to itself."""
        assert _registry(tmp_path).resolve("classical").id == "classical"

    def test_resolve_defaults(self, tmp_path: Path) -> None:
        """An empty or unknown id resolves to the default kit."""
        registry = _registry(tmp_path)
        assert registry.resolve(None).id == "baroque"
        assert registry.resolve("nope").id == "baroque"

    def test_create(self, tmp_path: Path) -> None:
        """A custom kit is created with the given selections."""
        kit = _registry(tmp_path).create("  Mine  ", ["empty"], ["imitation"])
        assert kit.name == "Mine"
        assert kit.rules == frozenset({"empty"})
        assert kit.techniques == frozenset({"imitation"})
        assert kit.id.startswith("s-")

    def test_create_requires_name(self, tmp_path: Path) -> None:
        """An empty name is rejected."""
        with pytest.raises(ValueError):
            _registry(tmp_path).create("   ", ["empty"], ["imitation"])

    def test_create_rejects_unknown_rule(self, tmp_path: Path) -> None:
        """An unknown rule id is rejected."""
        with pytest.raises(ValueError):
            _registry(tmp_path).create("Mine", ["nope"], ["imitation"])

    def test_create_rejects_unknown_technique(self, tmp_path: Path) -> None:
        """An unknown technique id is rejected."""
        with pytest.raises(ValueError):
            _registry(tmp_path).create("Mine", ["empty"], ["nope"])

    def test_rename(self, tmp_path: Path) -> None:
        """A custom kit can be renamed."""
        registry = _registry(tmp_path)
        kit = registry.create("Mine", ["empty"], ["imitation"])
        assert registry.rename(kit.id, "Yours").name == "Yours"
        assert registry.get(kit.id).name == "Yours"

    def test_rename_builtin(self, tmp_path: Path) -> None:
        """A built-in kit cannot be renamed."""
        with pytest.raises(ValueError):
            _registry(tmp_path).rename("baroque", "X")

    def test_rename_requires_name(self, tmp_path: Path) -> None:
        """A blank new name is rejected."""
        registry = _registry(tmp_path)
        kit = registry.create("Mine", ["empty"], ["imitation"])
        with pytest.raises(ValueError):
            registry.rename(kit.id, "  ")

    def test_rename_unknown(self, tmp_path: Path) -> None:
        """Renaming a missing custom kit raises a key error."""
        with pytest.raises(KeyError):
            _registry(tmp_path).rename("s-missing", "X")

    def test_delete(self, tmp_path: Path) -> None:
        """A custom kit can be deleted."""
        registry = _registry(tmp_path)
        kit = registry.create("Mine", ["empty"], ["imitation"])
        registry.delete(kit.id)
        with pytest.raises(KeyError):
            registry.get(kit.id)

    def test_delete_builtin(self, tmp_path: Path) -> None:
        """A built-in kit cannot be deleted."""
        with pytest.raises(ValueError):
            _registry(tmp_path).delete("baroque")

    def test_delete_unknown(self, tmp_path: Path) -> None:
        """Deleting a missing custom kit raises a key error."""
        with pytest.raises(KeyError):
            _registry(tmp_path).delete("s-missing")
