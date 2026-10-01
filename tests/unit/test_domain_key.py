# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for domain.key."""

from __future__ import annotations

import pytest

from harmoniatextor.domain.key import parse_key


class TestKey:
    """Key parsing."""

    def test_major(self) -> None:
        """Upper-case keys are major."""
        key = parse_key("C")
        assert key.is_major
        assert key.music21_name == "C"

    def test_minor(self) -> None:
        """Lower-case keys are minor."""
        key = parse_key("a")
        assert not key.is_major
        assert key.music21_name == "a"

    def test_flat(self) -> None:
        """Flats become dashes for music21."""
        assert parse_key("Eb").music21_name == "E-"

    def test_sharp(self) -> None:
        """Sharps are preserved."""
        assert parse_key("f#").music21_name == "f#"

    def test_invalid(self) -> None:
        """Invalid keys are rejected."""
        with pytest.raises(ValueError, match="invalid key"):
            parse_key("H")
