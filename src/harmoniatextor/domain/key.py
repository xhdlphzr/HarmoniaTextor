# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Key-signature value object.

A key is written as a tonic letter with optional accidental; upper case means
major and lower case means minor, e.g. ``"C"`` (C major), ``"a"`` (A minor),
``"Eb"`` (E-flat major), ``"f#"`` (F-sharp minor).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["KeySpec", "parse_key"]

_KEY_RE = re.compile(r"^([A-Ga-g])([#b]?)$")


@dataclass(frozen=True, slots=True)
class KeySpec:
    """A parsed key signature.

    Attributes:
        raw: The original textual specification.
        tonic: Upper-case tonic letter with accidental, e.g. ``"Eb"``.
        mode: Either ``"major"`` or ``"minor"``.
    """

    raw: str
    tonic: str
    mode: str

    @property
    def music21_name(self) -> str:
        """Return a music21-compatible key name such as ``"E-"`` or ``"f#"``."""
        letter = self.tonic[0]
        accidental = {"b": "-", "#": "#"}.get(self.tonic[1:], "")
        if self.mode == "major":
            return f"{letter}{accidental}"
        return f"{letter.lower()}{accidental}"

    @property
    def is_major(self) -> bool:
        """Whether the key is major."""
        return self.mode == "major"


def parse_key(value: str) -> KeySpec:
    """Parse a textual key specification.

    Args:
        value: A key string such as ``"C"``, ``"a"``, ``"Eb"`` or ``"f#"``.

    Returns:
        The parsed :class:`KeySpec`.

    Raises:
        ValueError: If the specification is not a valid key.
    """
    match = _KEY_RE.match(value)
    if match is None:
        raise ValueError(f"invalid key: {value!r}")
    letter, accidental = match.groups()
    mode = "major" if letter.isupper() else "minor"
    tonic = f"{letter.upper()}{accidental}"
    return KeySpec(raw=value, tonic=tonic, mode=mode)
