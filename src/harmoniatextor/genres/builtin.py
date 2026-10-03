# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""The four built-in genres."""

from __future__ import annotations

from harmoniatextor.checker.context import CheckerContext
from harmoniatextor.genres.base import Genre, MovementSpec

__all__ = [
    "ConcertoGenre",
    "PlainGenre",
    "SonataGenre",
    "SymphonyGenre",
]


class PlainGenre(Genre):
    """A single-movement piece or fugue."""

    id = "plain"
    display_name = "单曲/赋格"
    movement_specs = (MovementSpec(name="第一乐章 · 乐曲", tempo=84),)

    def checker_context(
        self, tonic: str, measure_count: int, *, complete: bool = False
    ) -> CheckerContext:
        """Build a strict single-movement checker context.

        Args:
            tonic: Home key.
            measure_count: The measure count.
            complete: The complete.

        Returns:
            The checker context result.
        """
        return self._final_context(
            tonic, measure_count, enforce=True, complete=complete
        )


class SonataGenre(Genre):
    """A three-movement sonata."""

    id = "sonata"
    display_name = "奏鸣曲"
    movement_specs = (
        MovementSpec(name="第一乐章 · 快板", tempo=120),
        MovementSpec(name="第二乐章 · 慢板", time_signature="3/4", tempo=60),
        MovementSpec(name="第三乐章 · 快板", tempo=132),
    )

    def checker_context(
        self, tonic: str, measure_count: int, *, complete: bool = False
    ) -> CheckerContext:
        """Build a sonata checker context.

        Args:
            tonic: Home key.
            measure_count: The measure count.
            complete: The complete.

        Returns:
            The checker context result.
        """
        return self._final_context(
            tonic, measure_count, enforce=False, complete=complete
        )


class ConcertoGenre(Genre):
    """A three-movement concerto with solo and tutti textures."""

    id = "concerto"
    display_name = "协奏曲"
    movement_specs = (
        MovementSpec(name="第一乐章 · 快板", tempo=120, voice_profile="solo_tutti"),
        MovementSpec(
            name="第二乐章 · 广板",
            time_signature="3/4",
            tempo=54,
            voice_profile="solo_tutti",
        ),
        MovementSpec(name="第三乐章 · 急板", tempo=144, voice_profile="solo_tutti"),
    )

    def checker_context(
        self, tonic: str, measure_count: int, *, complete: bool = False
    ) -> CheckerContext:
        """Build a concerto checker context.

        Args:
            tonic: Home key.
            measure_count: The measure count.
            complete: The complete.

        Returns:
            The checker context result.
        """
        return self._final_context(
            tonic, measure_count, enforce=False, complete=complete
        )


class SymphonyGenre(Genre):
    """A four-movement symphony."""

    id = "symphony"
    display_name = "交响曲"
    movement_specs = (
        MovementSpec(name="第一乐章 · 快板", tempo=120, voice_profile="orchestra"),
        MovementSpec(
            name="第二乐章 · 行板",
            time_signature="3/4",
            tempo=66,
            voice_profile="orchestra",
        ),
        MovementSpec(
            name="第三乐章 · 小步舞曲",
            time_signature="3/4",
            tempo=108,
            voice_profile="orchestra",
        ),
        MovementSpec(name="第四乐章 · 快板", tempo=132, voice_profile="orchestra"),
    )

    def checker_context(
        self, tonic: str, measure_count: int, *, complete: bool = False
    ) -> CheckerContext:
        """Build a symphony checker context.

        Args:
            tonic: Home key.
            measure_count: The measure count.
            complete: The complete.

        Returns:
            The checker context result.
        """
        return self._final_context(
            tonic, measure_count, enforce=False, complete=complete
        )
