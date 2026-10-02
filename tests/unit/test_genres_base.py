# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for genres.base."""

from __future__ import annotations

from harmoniatextor.genres.base import MovementSpec
from harmoniatextor.genres.builtin import PlainGenre

_DEFAULT_TEMPO = 96
_MEASURES = 12


class TestGenresBase:
    """Movement specs, work initialisation and profiles."""

    def test_movement_spec_defaults(self) -> None:
        """Movement specs default to 4/4 and four parts."""
        spec = MovementSpec(name="x")
        assert spec.time_signature == "4/4"
        assert spec.tempo == _DEFAULT_TEMPO
        assert spec.voice_profile == "four_part"

    def test_initialize_work(self) -> None:
        """A genre builds its movement skeleton."""
        movements = PlainGenre().initialize_work("w-1", "C")
        assert movements
        assert movements[0].work_id == "w-1"

    def test_complete_context(self) -> None:
        """A completed movement expects a final cadence at its last measure."""
        context = PlainGenre().checker_context("C", _MEASURES, complete=True)
        assert context.expectations[0].measure == _MEASURES
        assert context.expectations[0].cadence
