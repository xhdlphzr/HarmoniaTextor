# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for styles.builtin."""

from __future__ import annotations

from harmoniatextor.styles.builtin import (
    ALL_RULE_IDS,
    ALL_TECHNIQUE_IDS,
    BUILTIN_KITS,
    DEFAULT_STYLE_ID,
    FULL_STYLE_ID,
    STRUCTURAL_TECHNIQUES,
)

_EXPECTED_TECHNIQUES = {
    "full": 36,
    "baroque": 30,
    "classical": 31,
    "romantic": 34,
    "impressionist": 33,
}


class TestBuiltinKits:
    """The shipped style kits."""

    def test_ids(self) -> None:
        """The default and full ids are stable."""
        assert DEFAULT_STYLE_ID == "baroque"
        assert FULL_STYLE_ID == "full"
        assert set(BUILTIN_KITS) == {"full", "baroque", "classical", "romantic", "impressionist"}

    def test_full_preset(self) -> None:
        """The full preset carries every rule and technique."""
        kit = BUILTIN_KITS[FULL_STYLE_ID]
        assert kit.rules == ALL_RULE_IDS
        assert kit.techniques == ALL_TECHNIQUE_IDS

    def test_technique_counts(self) -> None:
        """Each preset exposes the agreed number of techniques."""
        for kit_id, expected in _EXPECTED_TECHNIQUES.items():
            assert len(BUILTIN_KITS[kit_id].techniques) == expected

    def test_subsets(self) -> None:
        """Every kit only references known rules and techniques."""
        for kit in BUILTIN_KITS.values():
            assert kit.rules <= ALL_RULE_IDS
            assert kit.techniques <= ALL_TECHNIQUE_IDS

    def test_structural_always_present(self) -> None:
        """Structural techniques are kept by every named style."""
        for kit_id in ("baroque", "classical", "romantic", "impressionist"):
            assert BUILTIN_KITS[kit_id].techniques >= STRUCTURAL_TECHNIQUES

    def test_free_voice_leading_membership(self) -> None:
        """Only the expressive styles (and full) offer the exemption."""
        assert "free_voice_leading" in BUILTIN_KITS["romantic"].techniques
        assert "free_voice_leading" in BUILTIN_KITS["impressionist"].techniques
        assert "free_voice_leading" in BUILTIN_KITS["full"].techniques
        assert "free_voice_leading" not in BUILTIN_KITS["baroque"].techniques
