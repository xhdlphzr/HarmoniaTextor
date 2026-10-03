# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from music21 import stream

from harmoniatextor.domain.models import Theme, ThemeNote
from harmoniatextor.score.io import new_score
from harmoniatextor.service.service import CompositionService
from harmoniatextor.storage.project_store import ProjectStore
from harmoniatextor.styles.registry import StyleRegistry
from harmoniatextor.styles.store import StyleKitStore


@pytest.fixture
def store(tmp_path: Path) -> ProjectStore:
    """Return a project store rooted in a temporary directory."""
    return ProjectStore(tmp_path / "data")


@pytest.fixture
def service(store: ProjectStore, tmp_path: Path) -> CompositionService:
    """Return a composition service with isolated storage and style kits."""
    styles = StyleRegistry(StyleKitStore(tmp_path / "config"))
    return CompositionService(store, styles=styles)


@pytest.fixture
def score4() -> stream.Score:
    """Return an empty four-voice C major score."""
    return new_score(
        key="C",
        time_signature="4/4",
        tempo_bpm=96,
        voices=["soprano", "alto", "tenor", "bass"],
    )


@pytest.fixture
def theme() -> Theme:
    """Return a simple C major theme."""
    return Theme(
        id=1,
        movement_id="m01",
        voice="soprano",
        start_measure=1,
        notes=[
            ThemeNote("C5", 1.0),
            ThemeNote("D5", 1.0),
            ThemeNote("E5", 1.0),
            ThemeNote("F5", 1.0),
        ],
        created_revision="r-0",
    )
