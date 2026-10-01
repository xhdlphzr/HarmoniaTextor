# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for render.exporter (file-name helpers)."""

from __future__ import annotations

from harmoniatextor.render.exporter import safe_filename

_MAX_FILENAME = 120


class TestSafeFilename:
    """Title sanitisation for exported file names."""

    def test_removes_forbidden(self) -> None:
        """Characters forbidden by the filesystems are removed."""
        assert safe_filename('a<b>c:d"e/f\\g|h?i*j', "fb") == "abcdefghij"

    def test_collapses_and_trims(self) -> None:
        """Whitespace is collapsed and trailing dots stripped."""
        assert safe_filename("  你好   世界.  ", "fb") == "你好 世界"

    def test_empty_falls_back(self) -> None:
        """A blank or dot-only title uses the fallback."""
        assert safe_filename("   ", "fb") == "fb"
        assert safe_filename("...", "fb") == "fb"

    def test_reserved_falls_back(self) -> None:
        """Reserved device names use the fallback."""
        assert safe_filename("CON", "fb") == "fb"

    def test_too_long_truncated(self) -> None:
        """Over-long titles are capped."""
        assert len(safe_filename("x" * 500, "fb")) == _MAX_FILENAME
