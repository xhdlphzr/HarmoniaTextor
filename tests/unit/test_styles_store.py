# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for styles.store."""

from __future__ import annotations

from pathlib import Path

import pytest

from harmoniatextor import config
from harmoniatextor.styles.base import StyleKit
from harmoniatextor.styles.store import StyleKitStore


def _kit() -> StyleKit:
    """Return a small custom kit."""
    return StyleKit(
        id="s-1",
        name="Mine",
        brief="",
        rules=frozenset({"empty"}),
        techniques=frozenset({"imitation"}),
    )


class TestStyleKitStore:
    """Custom-kit JSON persistence."""

    def test_default_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a root the configuration directory is used."""
        monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
        assert StyleKitStore().root == tmp_path

    def test_load_missing(self, tmp_path: Path) -> None:
        """A missing file yields no kits."""
        assert StyleKitStore(tmp_path).load() == {}

    def test_round_trip(self, tmp_path: Path) -> None:
        """Saved kits can be loaded again."""
        store = StyleKitStore(tmp_path)
        store.save({"s-1": _kit()})
        loaded = store.load()
        assert loaded["s-1"].name == "Mine"
        assert loaded["s-1"].builtin is False

    def test_load_malformed(self, tmp_path: Path) -> None:
        """Malformed JSON yields no kits."""
        store = StyleKitStore(tmp_path)
        store.path.write_text("{not json", encoding="utf-8")
        assert store.load() == {}

    def test_load_non_object(self, tmp_path: Path) -> None:
        """A non-object document yields no kits."""
        store = StyleKitStore(tmp_path)
        store.path.write_text("[]", encoding="utf-8")
        assert store.load() == {}

    def test_load_skips_invalid_entries(self, tmp_path: Path) -> None:
        """Entries without an id and name are skipped."""
        store = StyleKitStore(tmp_path)
        store.path.write_text(
            '{"kits": [{"id": "a"}, {"id": "b", "name": "B"}, 5]}',
            encoding="utf-8",
        )
        loaded = store.load()
        assert set(loaded) == {"b"}
