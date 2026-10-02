# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for techniques.registry."""

from __future__ import annotations

import pytest

from harmoniatextor.techniques import build_default_registry

REGISTRY = build_default_registry()


_TECHNIQUE_COUNT = 36


class TestRegistry:
    """Registry behaviour."""

    def test_len_and_ids(self) -> None:
        """All 25 techniques are registered."""
        assert len(REGISTRY) == _TECHNIQUE_COUNT
        assert "imitation" in REGISTRY

    def test_unknown(self) -> None:
        """Unknown techniques raise KeyError."""
        with pytest.raises(KeyError):
            REGISTRY.get("nope")

    def test_duplicate(self) -> None:
        """Duplicate registration is rejected."""
        with pytest.raises(ValueError, match="duplicate"):
            REGISTRY.register(REGISTRY.get("imitation"))

    def test_schema(self) -> None:
        """Techniques expose a JSON schema."""
        assert "properties" in REGISTRY.get("imitation").schema()
