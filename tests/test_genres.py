# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the genre framework."""

from __future__ import annotations

import pytest

from harmoniatextor.checker.context import CheckerContext
from harmoniatextor.genres import (
    ConcertoGenre,
    GenreRegistry,
    PlainGenre,
    SonataGenre,
    SymphonyGenre,
    build_default_registry,
)
from harmoniatextor.genres.base import ALL_TECHNIQUES, Genre, MovementSpec

_GENRE_COUNT = 4
_TECHNIQUE_COUNT = 25
_SONATA_MOVEMENTS = 3
_CONCERTO_MOVEMENTS = 3
_SYMPHONY_MOVEMENTS = 4


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


class TestGenres:
    """Genre behaviour."""

    def test_all_techniques_available(self) -> None:
        """Every genre allows all 25 techniques."""
        assert len(ALL_TECHNIQUES) == _TECHNIQUE_COUNT
        for genre in build_default_registry().all():
            assert genre.allowed_techniques == ALL_TECHNIQUES

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

    def test_relaxed_profile(self) -> None:
        """Multi-movement genres downgrade leading-tone and doubling rules."""
        context = SonataGenre().checker_context("C", 16)
        assert not context.enforce_voice_count
        profile = SonataGenre().profile
        assert profile.setting_for("leading").severity is not None
        assert profile.setting_for("omission").severity is not None

    def test_concerto_context(self) -> None:
        """Concerto contexts skip voice-count enforcement."""
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
            allowed_techniques = ALL_TECHNIQUES
            profile = PlainGenre().profile

            def checker_context(
                self, tonic: str, measure_count: int, *, complete: bool = False
            ) -> CheckerContext:
                return self._final_context(tonic, measure_count, enforce=True, complete=complete)

        movement = Custom().initialize_work("w", "C")[0]
        assert movement.key.raw == "G"
