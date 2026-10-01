# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for genres.registry."""

from __future__ import annotations

import pytest

from harmoniatextor.genres import (
    GenreRegistry,
    PlainGenre,
    build_default_registry,
)

_GENRE_COUNT = 4


class TestGenreRegistry:
    """Genre registration and lookup."""

    def test_defaults(self) -> None:
        """Four genres are registered."""
        registry = build_default_registry()
        assert registry.ids() == ["plain", "sonata", "concerto", "symphony"]
        assert len(registry.all()) == _GENRE_COUNT

    def test_unknown(self) -> None:
        """Unknown genres raise KeyError."""
        with pytest.raises(KeyError):
            build_default_registry().get("opera")

    def test_duplicate(self) -> None:
        """Duplicate registration is rejected."""
        registry = GenreRegistry()
        registry.register(PlainGenre())
        with pytest.raises(ValueError, match="duplicate"):
            registry.register(PlainGenre())
