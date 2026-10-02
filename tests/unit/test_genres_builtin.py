# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for genres.builtin."""

from __future__ import annotations

from harmoniatextor.checker.context import CheckerContext
from harmoniatextor.genres import (
    ConcertoGenre,
    PlainGenre,
    SonataGenre,
    SymphonyGenre,
    build_default_registry,
)
from harmoniatextor.genres.base import Genre, MovementSpec

_SONATA_MOVEMENTS = 3
_CONCERTO_MOVEMENTS = 3
_SYMPHONY_MOVEMENTS = 4
_GENRE_COUNT = 4


class TestGenres:
    """Genre behaviour."""

    def test_registry(self) -> None:
        """The registry exposes the four built-in genres."""
        registry = build_default_registry()
        assert len(registry.all()) == _GENRE_COUNT
        assert {genre.id for genre in registry.all()} == {
            "plain",
            "sonata",
            "concerto",
            "symphony",
        }

    def test_plain_movements(self) -> None:
        """Plain works have one movement."""
        movements = PlainGenre().initialize_work("w", "C")
        assert len(movements) == 1
        assert movements[0].key.raw == "C"

    def test_sonata_movements(self) -> None:
        """Sonatas have three movements."""
        assert len(SonataGenre().initialize_work("w", "C")) == _SONATA_MOVEMENTS

    def test_concerto_movements(self) -> None:
        """Concertos have three movements with a solo/tutti profile."""
        movements = ConcertoGenre().initialize_work("w", "C")
        assert len(movements) == _CONCERTO_MOVEMENTS
        assert movements[0].voice_profile == "solo_tutti"

    def test_symphony_movements(self) -> None:
        """Symphonies have four movements."""
        movements = SymphonyGenre().initialize_work("w", "C")
        assert len(movements) == _SYMPHONY_MOVEMENTS
        assert movements[0].voice_profile == "orchestra"

    def test_plain_context_enforces_voice_count(self) -> None:
        """Plain drafting enforces voice count but has no structural end."""
        context = PlainGenre().checker_context("C", 16)
        assert context.enforce_voice_count
        assert context.expectations == []

    def test_plain_context_complete(self) -> None:
        """A completed plain movement expects a final cadence."""
        context = PlainGenre().checker_context("C", 16, complete=True)
        assert context.expectations[0].cadence

    def test_multi_movement_context(self) -> None:
        """Multi-movement genres skip voice-count enforcement."""
        context = SonataGenre().checker_context("C", 16)
        assert not context.enforce_voice_count

    def test_concerto_context(self) -> None:
        """Concerto contexts expect a final cadence."""
        context = ConcertoGenre().checker_context("C", 20, complete=True)
        assert context.expectations[0].cadence

    def test_symphony_context(self) -> None:
        """Symphony contexts skip voice-count enforcement."""
        context = SymphonyGenre().checker_context("C", 20)
        assert not context.enforce_voice_count

    def test_movement_key_override(self) -> None:
        """A movement key override is honoured."""

        class Custom(Genre):
            id = "custom"
            display_name = "Custom"
            movement_specs = (MovementSpec(name="I", key="G"),)

            def checker_context(
                self, tonic: str, measure_count: int, *, complete: bool = False
            ) -> CheckerContext:
                return self._final_context(tonic, measure_count, enforce=True, complete=complete)

        movement = Custom().initialize_work("w", "C")[0]
        assert movement.key.raw == "G"
