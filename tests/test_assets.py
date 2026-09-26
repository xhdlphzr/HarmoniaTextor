# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for bundled asset resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app import assets


class TestAssets:
    """Asset path resolution."""

    def test_assets_dir_repository(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """During development the assets live next to the repository root."""
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
        directory = assets.assets_dir()
        assert directory.name == "assets"
        assert (directory / "Franx.png").exists()

    def test_assets_dir_frozen(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When frozen the assets live inside the bundle directory."""
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        assert assets.assets_dir() == tmp_path / "assets"

    def test_favicon_icon(self) -> None:
        """The favicon is the PNG icon."""
        icon = assets.favicon_icon()
        assert icon is not None
        assert icon.name == "Franx.png"

    def test_favicon_icon_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A missing favicon returns None."""
        monkeypatch.setattr(assets, "assets_dir", lambda: tmp_path)
        assert assets.favicon_icon() is None
