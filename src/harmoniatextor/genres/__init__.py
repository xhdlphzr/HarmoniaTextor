# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Pluggable multi-genre framework."""

from __future__ import annotations

from harmoniatextor.genres.base import Genre, MovementSpec
from harmoniatextor.genres.builtin import (
    ConcertoGenre,
    PlainGenre,
    SonataGenre,
    SymphonyGenre,
    VocalGenre,
)
from harmoniatextor.genres.registry import GenreRegistry, build_default_registry

__all__ = [
    "ConcertoGenre",
    "Genre",
    "GenreRegistry",
    "MovementSpec",
    "PlainGenre",
    "SonataGenre",
    "SymphonyGenre",
    "VocalGenre",
    "build_default_registry",
]
