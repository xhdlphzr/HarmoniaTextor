# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for styles.base."""

from __future__ import annotations

from harmoniatextor.styles.base import StyleKit


class TestStyleKit:
    """Style-kit serialisation."""

    def test_round_trip(self) -> None:
        """A kit survives a dictionary round trip."""
        kit = StyleKit(
            id="x",
            name="X",
            brief="brief",
            rules=frozenset({"empty", "voices"}),
            techniques=frozenset({"imitation"}),
        )
        data = kit.to_dict()
        assert data["rules"] == ["empty", "voices"]
        assert data["techniques"] == ["imitation"]
        assert StyleKit.from_dict(data) == kit

    def test_defaults(self) -> None:
        """Empty sets and builtin default to falsy values."""
        kit = StyleKit(id="x", name="X", brief="")
        assert kit.rules == frozenset()
        assert kit.techniques == frozenset()
        assert kit.builtin is False

    def test_from_dict_minimal(self) -> None:
        """Missing keys fall back to empty defaults."""
        kit = StyleKit.from_dict({"id": "y", "name": "Y"})
        assert kit.brief == ""
        assert kit.rules == frozenset()
        assert kit.techniques == frozenset()
        assert kit.builtin is False
