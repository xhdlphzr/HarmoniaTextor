# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""SQLite-backed local project storage.

Every work, movement, revision, theme and journal event lives in a single
SQLite database (``history.db``) inside the store root, which by default is
``~/.harmonia_textor``.  The public API mirrors the previous file-based store,
so the service, exporter and web layers are unchanged.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from harmoniatextor.domain.enums import WorkStatus
from harmoniatextor.domain.key import parse_key
from harmoniatextor.domain.models import (
    Movement,
    Revision,
    StyleSelection,
    Theme,
    ThemeNote,
    Work,
)

__all__ = ["ProjectStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS works (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    genre TEXT NOT NULL,
    tonic TEXT NOT NULL,
    status TEXT NOT NULL,
    style TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS movements (
    work_id TEXT NOT NULL,
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    time_signature TEXT NOT NULL,
    key TEXT NOT NULL,
    tempo INTEGER NOT NULL,
    voice_profile TEXT NOT NULL,
    canonical_revision TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (work_id, id),
    FOREIGN KEY (work_id) REFERENCES works (id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS revisions (
    work_id TEXT NOT NULL,
    movement_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    id TEXT NOT NULL,
    source TEXT NOT NULL,
    technique TEXT,
    params TEXT,
    ok INTEGER NOT NULL,
    parent_seq INTEGER,
    check_report TEXT,
    full_xml TEXT NOT NULL,
    PRIMARY KEY (work_id, movement_id, seq)
);
CREATE TABLE IF NOT EXISTS themes (
    work_id TEXT NOT NULL,
    movement_id TEXT NOT NULL,
    theme_id INTEGER NOT NULL,
    voice TEXT NOT NULL,
    start_measure INTEGER NOT NULL,
    created_revision TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    notes TEXT NOT NULL,
    PRIMARY KEY (work_id, movement_id, theme_id)
);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id TEXT NOT NULL,
    event TEXT NOT NULL
);
"""


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Add a column to an existing table when it is missing.

    Args:
        conn: Open database connection.
        table: Table name.
        column: Column name.
        ddl: Column definition appended after the column name.
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


class ProjectStore:
    """SQLite-backed storage for works, revisions, themes and journals.

    Attributes:
        root: Root directory holding the database and export folders.
        db_path: Path to the SQLite database file.
    """

    def __init__(self, root: Path) -> None:
        """Initialise the store.

        Args:
            root: Root directory; created when missing.
        """
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "history.db"
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            _ensure_column(conn, "movements", "prompt", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(conn, "works", "style", "TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a transaction against the database.

        Yields:
            A SQLite connection, committed when the block succeeds.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_work(self, work: Work) -> None:
        """Persist a work's metadata.

        Args:
            work: Work to persist.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO works (id, title, genre, tonic, status, style, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    title = excluded.title,
                    genre = excluded.genre,
                    tonic = excluded.tonic,
                    status = excluded.status,
                    style = excluded.style,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    work.id,
                    work.title,
                    work.genre,
                    work.tonic.raw,
                    work.status.value,
                    json.dumps(work.style.to_dict(), ensure_ascii=False)
                    if work.style is not None
                    else None,
                    work.created_at,
                    work.updated_at,
                ),
            )
            conn.execute("DELETE FROM movements WHERE work_id = ?", (work.id,))
            conn.executemany(
                """
                INSERT INTO movements
                    (work_id, id, name, time_signature, key, tempo, voice_profile,
                     canonical_revision, ordinal, prompt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        work.id,
                        movement.id,
                        movement.name,
                        movement.time_signature,
                        movement.key.raw,
                        movement.tempo,
                        movement.voice_profile,
                        movement.canonical_revision,
                        index,
                        movement.prompt,
                    )
                    for index, movement in enumerate(work.movements)
                ],
            )

    def load_work(self, work_id: str) -> Work:
        """Load a work's metadata.

        Args:
            work_id: Work identifier.

        Returns:
            The loaded work.

        Raises:
            FileNotFoundError: When the work does not exist.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, genre, tonic, status, style, created_at, updated_at
                FROM works WHERE id = ?
                """,
                (work_id,),
            ).fetchone()
            if row is None:
                raise FileNotFoundError(work_id)
            movement_rows = conn.execute(
                """
                SELECT id, name, time_signature, key, tempo, voice_profile,
                       canonical_revision, prompt
                FROM movements WHERE work_id = ? ORDER BY ordinal
                """,
                (work_id,),
            ).fetchall()
        movements = [
            Movement(
                id=item[0],
                work_id=work_id,
                name=item[1],
                time_signature=item[2],
                key=parse_key(item[3]),
                tempo=item[4],
                voice_profile=item[5],
                canonical_revision=item[6],
                prompt=item[7],
            )
            for item in movement_rows
        ]
        return Work(
            id=row[0],
            title=row[1],
            genre=row[2],
            tonic=parse_key(row[3]),
            movements=movements,
            status=WorkStatus(row[4]),
            style=StyleSelection.from_dict(json.loads(row[5])) if row[5] else None,
            created_at=row[6],
            updated_at=row[7],
        )

    def list_works(self) -> list[str]:
        """List all stored work identifiers, newest first.

        Works are ordered by creation time so the history is always presented in
        strict chronological order rather than by the random identifier.

        Returns:
            Work identifiers, most recently created first.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM works ORDER BY created_at DESC, id").fetchall()
        return [str(row[0]) for row in rows]

    def save_revision_for(self, work_id: str, revision: Revision) -> None:
        """Persist a revision for a specific work and movement.

        Args:
            work_id: Owning work.
            revision: Revision to persist.
        """
        check = revision.check.to_dict() if revision.check is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO revisions
                    (work_id, movement_id, seq, id, source, technique, params, ok,
                     parent_seq, check_report, full_xml)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    work_id,
                    revision.movement_id,
                    revision.seq,
                    revision.id,
                    revision.source.value,
                    revision.technique,
                    json.dumps(revision.params, ensure_ascii=False),
                    int(revision.ok),
                    revision.parent_seq,
                    json.dumps(check, ensure_ascii=False) if check is not None else None,
                    revision.full_xml,
                ),
            )

    def load_revision_xml(self, work_id: str, movement_id: str, seq: int) -> str:
        """Load the MusicXML of a revision.

        Args:
            work_id: Owning work.
            movement_id: Movement identifier.
            seq: Revision sequence number.

        Returns:
            The MusicXML text.

        Raises:
            FileNotFoundError: When the revision does not exist.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT full_xml FROM revisions
                WHERE work_id = ? AND movement_id = ? AND seq = ?
                """,
                (work_id, movement_id, seq),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"{work_id}/{movement_id}/r{seq}")
        return str(row[0])

    def load_revision_meta(self, work_id: str, movement_id: str) -> list[dict[str, Any]]:
        """Load revision metadata for a movement.

        Args:
            work_id: Owning work.
            movement_id: Movement identifier.

        Returns:
            A list of metadata dictionaries in sequence order.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, seq, movement_id, source, technique, params, ok,
                       parent_seq, check_report
                FROM revisions
                WHERE work_id = ? AND movement_id = ? ORDER BY seq
                """,
                (work_id, movement_id),
            ).fetchall()
        return [
            {
                "id": row[0],
                "seq": row[1],
                "movement_id": row[2],
                "source": row[3],
                "technique": row[4],
                "params": json.loads(row[5]) if row[5] is not None else None,
                "ok": bool(row[6]),
                "parent_seq": row[7],
                "check": json.loads(row[8]) if row[8] is not None else None,
            }
            for row in rows
        ]

    def save_themes(self, work_id: str, movement_id: str, themes: dict[int, Theme]) -> None:
        """Persist the theme registry of a movement.

        Args:
            work_id: Owning work.
            movement_id: Movement identifier.
            themes: Theme registry keyed by theme number.
        """
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM themes WHERE work_id = ? AND movement_id = ?",
                (work_id, movement_id),
            )
            conn.executemany(
                """
                INSERT INTO themes
                    (work_id, movement_id, theme_id, voice, start_measure,
                     created_revision, fingerprint, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        work_id,
                        movement_id,
                        theme_id,
                        theme.voice,
                        theme.start_measure,
                        theme.created_revision,
                        theme.fingerprint,
                        json.dumps(
                            [
                                {"pitch": note.pitch, "quarter_length": note.quarter_length}
                                for note in theme.notes
                            ],
                            ensure_ascii=False,
                        ),
                    )
                    for theme_id, theme in sorted(themes.items())
                ],
            )

    def load_themes(self, work_id: str, movement_id: str) -> dict[int, Theme]:
        """Load the theme registry of a movement.

        Args:
            work_id: Owning work.
            movement_id: Movement identifier.

        Returns:
            A theme registry keyed by theme number.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT theme_id, voice, start_measure, created_revision,
                       fingerprint, notes
                FROM themes
                WHERE work_id = ? AND movement_id = ? ORDER BY theme_id
                """,
                (work_id, movement_id),
            ).fetchall()
        registry: dict[int, Theme] = {}
        for row in rows:
            notes = [
                ThemeNote(pitch=item["pitch"], quarter_length=item["quarter_length"])
                for item in json.loads(row[5])
            ]
            registry[row[0]] = Theme(
                id=row[0],
                movement_id=movement_id,
                voice=row[1],
                start_measure=row[2],
                notes=notes,
                created_revision=row[3],
                fingerprint=row[4],
            )
        return registry

    def load_all_themes(self, work_id: str) -> dict[tuple[str, int], Theme]:
        """Load every theme of a work across its movements.

        Args:
            work_id: Owning work.

        Returns:
            A theme registry keyed by ``(target_id, theme_id)``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT movement_id, theme_id, voice, start_measure, created_revision,
                       fingerprint, notes
                FROM themes
                WHERE work_id = ? ORDER BY movement_id, theme_id
                """,
                (work_id,),
            ).fetchall()
        registry: dict[tuple[str, int], Theme] = {}
        for row in rows:
            notes = [
                ThemeNote(pitch=item["pitch"], quarter_length=item["quarter_length"])
                for item in json.loads(row[6])
            ]
            registry[(row[0], row[1])] = Theme(
                id=row[1],
                movement_id=row[0],
                voice=row[2],
                start_measure=row[3],
                notes=notes,
                created_revision=row[4],
                fingerprint=row[5],
            )
        return registry

    def append_journal(self, work_id: str, event: dict[str, Any]) -> None:
        """Append an event to a work's journal.

        Args:
            work_id: Owning work.
            event: JSON-serialisable event payload.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO journal (work_id, event) VALUES (?, ?)",
                (work_id, json.dumps(event, ensure_ascii=False)),
            )

    def load_journal(self, work_id: str) -> list[dict[str, Any]]:
        """Load a work's journal.

        Args:
            work_id: Owning work.

        Returns:
            A list of event dictionaries in insertion order.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event FROM journal WHERE work_id = ? ORDER BY id",
                (work_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
