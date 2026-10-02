# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for file-backed project storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from harmoniatextor.domain.enums import ToolKind, WorkStatus
from harmoniatextor.domain.key import parse_key
from harmoniatextor.domain.models import (
    Movement,
    Revision,
    StyleSelection,
    Theme,
    ThemeNote,
    Work,
)
from harmoniatextor.storage.project_store import ProjectStore


def make_work() -> Work:
    """Return a work with one movement."""
    movement = Movement(
        id="m01",
        work_id="w-1",
        name="I. Allegro",
        time_signature="4/4",
        key=parse_key("C"),
        tempo=120,
        voice_profile="four_part",
    )
    return Work(
        id="w-1",
        title="Demo",
        genre="plain",
        tonic=parse_key("C"),
        movements=[movement],
        status=WorkStatus.DRAFT,
        created_at="t0",
        updated_at="t0",
    )


class TestProjectStore:
    """Persistence round trips."""

    def test_work_roundtrip(self, tmp_path: Path) -> None:
        """A work survives save/load."""
        store = ProjectStore(tmp_path / "data")
        work = make_work()
        store.save_work(work)
        loaded = store.load_work("w-1")
        assert loaded.title == "Demo"
        assert loaded.movements[0].key.raw == "C"
        assert store.list_works() == ["w-1"]

    def test_missing_work(self, tmp_path: Path) -> None:
        """Loading a missing work raises."""
        with pytest.raises(FileNotFoundError):
            ProjectStore(tmp_path / "data").load_work("nope")

    def test_style_roundtrip(self, tmp_path: Path) -> None:
        """A work's style snapshot survives save/load."""
        store = ProjectStore(tmp_path / "data")
        work = make_work()
        work.style = StyleSelection(
            id="baroque",
            name="巴洛克",
            rules=frozenset({"empty"}),
            techniques=frozenset({"imitation"}),
        )
        store.save_work(work)
        loaded = store.load_work("w-1")
        assert loaded.style is not None
        assert loaded.style.name == "巴洛克"
        assert loaded.style.rules == frozenset({"empty"})

    def test_style_absent(self, tmp_path: Path) -> None:
        """A work without a style loads as None."""
        store = ProjectStore(tmp_path / "data")
        store.save_work(make_work())
        assert store.load_work("w-1").style is None

    def test_works_style_migration(self, tmp_path: Path) -> None:
        """A works table created before the style column is migrated."""
        connection = sqlite3.connect(tmp_path / "history.db")
        connection.execute(
            "CREATE TABLE works (id TEXT PRIMARY KEY, title TEXT, genre TEXT, tonic TEXT,"
            " status TEXT, created_at TEXT, updated_at TEXT)"
        )
        connection.commit()
        connection.close()
        store = ProjectStore(tmp_path)
        store.save_work(make_work())
        assert store.load_work("w-1").style is None

    def test_revision_roundtrip(self, tmp_path: Path) -> None:
        """Revisions store XML and metadata."""
        store = ProjectStore(tmp_path / "data")
        revision = Revision(
            id="r-m01-0",
            seq=0,
            movement_id="m01",
            full_xml="<score/>",
            source=ToolKind.IMPORT,
            ok=True,
        )
        store.save_revision_for("w-1", revision)
        assert store.load_revision_xml("w-1", "m01", 0) == "<score/>"
        meta = store.load_revision_meta("w-1", "m01")
        assert meta[0]["id"] == "r-m01-0"

    def test_missing_revision(self, tmp_path: Path) -> None:
        """Loading a missing revision raises."""
        store = ProjectStore(tmp_path / "data")
        with pytest.raises(FileNotFoundError):
            store.load_revision_xml("w-1", "m01", 9)
        assert store.load_revision_meta("w-1", "m01") == []

    def test_theme_roundtrip(self, tmp_path: Path) -> None:
        """Themes survive save/load."""
        store = ProjectStore(tmp_path / "data")
        theme = Theme(1, "m01", "soprano", 1, [ThemeNote("C5", 1.0)], "r-0")
        store.save_themes("w-1", "m01", {1: theme})
        loaded = store.load_themes("w-1", "m01")
        assert loaded[1].notes[0].pitch == "C5"
        assert loaded[1].fingerprint

    def test_themes_missing(self, tmp_path: Path) -> None:
        """A movement without themes returns an empty mapping."""
        assert ProjectStore(tmp_path / "data").load_themes("w-1", "m01") == {}

    def test_load_all_themes(self, tmp_path: Path) -> None:
        """Themes are loaded across every target of a work."""
        store = ProjectStore(tmp_path / "data")
        store.save_themes(
            "w-1", "m01", {1: Theme(1, "m01", "soprano", 1, [ThemeNote("C5", 1.0)], "r-0")}
        )
        store.save_themes(
            "w-1", "m02", {2: Theme(2, "m02", "alto", 1, [ThemeNote("E5", 1.0)], "r-1")}
        )
        loaded = store.load_all_themes("w-1")
        assert set(loaded) == {("m01", 1), ("m02", 2)}
        assert loaded[("m02", 2)].notes[0].pitch == "E5"

    def test_load_all_themes_missing(self, tmp_path: Path) -> None:
        """A work without themes returns an empty mapping."""
        assert ProjectStore(tmp_path / "data").load_all_themes("w-1") == {}

    def test_journal(self, tmp_path: Path) -> None:
        """Journal events append and reload."""
        store = ProjectStore(tmp_path / "data")
        store.append_journal("w-1", {"event": "a"})
        store.append_journal("w-1", {"event": "b"})
        assert [item["event"] for item in store.load_journal("w-1")] == ["a", "b"]
        assert ProjectStore(tmp_path / "other").load_journal("w-2") == []

    def test_movement_prompt_roundtrip(self, tmp_path: Path) -> None:
        """A movement's Step 1 prompt survives save/load."""
        store = ProjectStore(tmp_path / "data")
        work = make_work()
        work.movements[0].prompt = "写一个8小节的赋格主题"
        store.save_work(work)
        assert store.load_work("w-1").movements[0].prompt == "写一个8小节的赋格主题"

    def test_column_migration(self, tmp_path: Path) -> None:
        """A database created before the new columns is migrated in place."""
        connection = sqlite3.connect(tmp_path / "history.db")
        connection.execute(
            "CREATE TABLE movements (work_id TEXT, id TEXT, name TEXT, time_signature TEXT,"
            " key TEXT, tempo INTEGER, voice_profile TEXT, canonical_revision TEXT,"
            " ordinal INTEGER, PRIMARY KEY (work_id, id))"
        )
        connection.commit()
        connection.close()
        store = ProjectStore(tmp_path)
        store.save_work(make_work())
        loaded = store.load_work("w-1").movements[0]
        assert loaded.prompt == ""
