# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Registry for the pluggable genre framework."""

from __future__ import annotations

from harmoniatextor.genres.base import Genre
from harmoniatextor.genres.builtin import (
    ConcertoGenre,
    PlainGenre,
    SonataGenre,
    SymphonyGenre,
)

__all__ = ["GenreRegistry", "build_default_registry"]

_DEFAULT_GENRES: tuple[type[Genre], ...] = (
    PlainGenre,
    SonataGenre,
    ConcertoGenre,
    SymphonyGenre,
)


class GenreRegistry:
    """A registry of genre instances keyed by identifier.

    Attributes:
        _items: Internal mapping from identifier to genre instance.
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._items: dict[str, Genre] = {}

    def register(self, genre: Genre) -> None:
        """Register a genre.

        Args:
            genre: Genre to register.

        Raises:
            ValueError: When the identifier is already registered.
        """
        if genre.id in self._items:
            raise ValueError(f"duplicate genre id: {genre.id}")
        self._items[genre.id] = genre

    def get(self, genre_id: str) -> Genre:
        """Return a genre by identifier.

        Args:
            genre_id: Genre identifier.

        Returns:
            The genre instance.

        Raises:
            KeyError: When the identifier is unknown.
        """
        if genre_id not in self._items:
            raise KeyError(genre_id)
        return self._items[genre_id]

    def ids(self) -> list[str]:
        """Return all registered genre identifiers.

        Returns:
            Identifiers in registration order.
        """
        return list(self._items)

    def all(self) -> list[Genre]:
        """Return all registered genres.

        Returns:
            Genres in registration order.
        """
        return list(self._items.values())


def build_default_registry() -> GenreRegistry:
    """Build a registry with the four built-in genres.

    Returns:
        A populated :class:`GenreRegistry`.
    """
    registry = GenreRegistry()
    for genre_type in _DEFAULT_GENRES:
        registry.register(genre_type())
    return registry
