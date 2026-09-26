# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Interval parsing shared by every technique that transposes material.

The public convention (as used by the tool parameters) is a signed integer:

* ``+5`` / ``-4`` / ``+2`` / ``-3`` are *diatonic interval numbers* (a fifth,
  a fourth, a second, a third).  ``+5`` therefore means "up a fifth", the
  dominant direction, and ``-4`` means "down a fourth", the subdominant
  direction.
* Values whose magnitude is greater than 7 are interpreted as *semitones*;
  ``+12`` is an octave.

Perfect intervals (unison, fourth, fifth, octave) use perfect quality, all
other numbers use major quality.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DiatonicInterval", "parse_interval"]

_PERFECT: dict[int, tuple[int, int]] = {
    1: (0, 0),
    4: (5, 3),
    5: (7, 4),
    8: (12, 7),
}
_MAJOR: dict[int, tuple[int, int]] = {
    2: (2, 1),
    3: (4, 2),
    6: (9, 5),
    7: (11, 6),
}
_QUALITY: dict[int, str] = {1: "P", 4: "P", 5: "P", 8: "P", 2: "M", 3: "M", 6: "M", 7: "M"}
_MAX_DIATONIC = 7


@dataclass(frozen=True, slots=True)
class DiatonicInterval:
    """A signed interval described by semitones and diatonic steps.

    Attributes:
        semitones: Signed chromatic distance in semitones.
        steps: Signed number of diatonic staff steps.
        name: A music21-compatible interval name such as ``"P5"`` or ``"-M3"``.
    """

    semitones: int
    steps: int
    name: str


def parse_interval(value: int) -> DiatonicInterval:
    """Parse a signed interval specification into a :class:`DiatonicInterval`.

    Args:
        value: Signed interval number (diatonic for ``1..7``, semitones above).

    Returns:
        The normalised interval.

    Raises:
        ValueError: If ``value`` is zero (a no-op interval).
    """
    if value == 0:
        raise ValueError("interval must be non-zero")
    direction = 1 if value > 0 else -1
    magnitude = abs(value)
    if magnitude <= _MAX_DIATONIC:
        if magnitude in _PERFECT:
            semitones, steps = _PERFECT[magnitude]
        else:
            semitones, steps = _MAJOR[magnitude]
        quality = _QUALITY[magnitude]
        prefix = "-" if direction < 0 else ""
        name = f"{prefix}{quality}{magnitude}"
    else:
        semitones = magnitude
        steps = round(magnitude * 7 / 12)
        name = f"{'-' if direction < 0 else ''}P{steps + 1}"
    return DiatonicInterval(
        semitones=direction * semitones,
        steps=direction * steps,
        name=name,
    )
