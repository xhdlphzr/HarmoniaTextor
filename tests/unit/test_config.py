# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the user configuration file."""

from __future__ import annotations

from pathlib import Path

import pytest

from harmoniatextor import config

_SAVED_TEMPERATURE = 0.2
_DEFAULT_WINDOW_K = 200
_DEFAULT_WINDOW_TOKENS = 200_000
_SMALL_WINDOW_K = 64
_SMALL_WINDOW_TOKENS = 64_000
_MIN_WINDOW_TOKENS = 1_000
_TINY_WINDOW_K = 8
_TINY_WINDOW_TOKENS = 8_000


class TestConfig:
    """Configuration loading and creation."""

    def test_defaults(self) -> None:
        """The default configuration carries the expected keys."""
        defaults = config.default_config()
        assert defaults["model"] == "gpt-4o-mini"
        assert defaults["api_key"] == ""
        assert defaults["context_window"] == _DEFAULT_WINDOW_K
        assert defaults["language"] == "en"
        assert "temperature" not in defaults

    def test_context_window_values(self) -> None:
        """The context window accepts numbers and strings and is clamped."""
        assert config.context_window_tokens({"context_window": _DEFAULT_WINDOW_K}) == (
            _DEFAULT_WINDOW_TOKENS
        )
        assert config.context_window_tokens({"context_window": str(_SMALL_WINDOW_K)}) == (
            _SMALL_WINDOW_TOKENS
        )
        assert config.context_window_tokens({"context_window": 0}) == _MIN_WINDOW_TOKENS
        assert config.context_window_tokens({"context_window": "bad"}) == _DEFAULT_WINDOW_TOKENS
        assert config.context_window_tokens({}) == _DEFAULT_WINDOW_TOKENS

    def test_context_window_from_current(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no mapping the effective configuration is used."""
        monkeypatch.setattr(config, "current_config", lambda: {"context_window": _TINY_WINDOW_K})
        assert config.context_window_tokens() == _TINY_WINDOW_TOKENS

    def test_config_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The config directory lives under the home directory."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert config.config_dir() == tmp_path / config.CONFIG_DIR_NAME

    def test_config_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The config file is config.json inside the config directory."""
        monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
        assert config.config_path() == tmp_path / "config.json"

    def test_load_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A missing file yields an empty mapping."""
        monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
        assert config.load_config() == {}

    def test_ensure_and_load(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Ensuring creates the file once and it can be loaded."""
        monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
        path = config.ensure_config()
        assert path.exists()
        assert config.load_config()["model"] == "gpt-4o-mini"
        assert config.ensure_config() == path

    def test_load_invalid_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Malformed JSON yields an empty mapping."""
        monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
        (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
        assert config.load_config() == {}

    def test_load_non_dict(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A non-object JSON document yields an empty mapping."""
        monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
        (tmp_path / "config.json").write_text("[1, 2]", encoding="utf-8")
        assert config.load_config() == {}

    def test_load_read_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An unreadable path yields an empty mapping."""
        directory = tmp_path / "config.json"
        directory.mkdir()
        monkeypatch.setattr(config, "config_path", lambda: directory)
        assert config.load_config() == {}

    def test_save_and_current(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Saving merges allowed values over the defaults."""
        monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
        path = config.save_config({"model": "custom", "temperature": _SAVED_TEMPERATURE})
        assert path.exists()
        current = config.current_config()
        assert current["model"] == "custom"
        assert current["base_url"] == "https://api.openai.com/v1"
        assert "temperature" not in current

    def test_load_drops_unknown_keys(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Unknown keys such as temperature are ignored when loading."""
        monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
        (tmp_path / "config.json").write_text(
            '{"model": "custom", "temperature": 0.2}', encoding="utf-8"
        )
        loaded = config.load_config()
        assert loaded == {"model": "custom"}

    def test_drops_checker(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A legacy checker section is ignored."""
        monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
        (tmp_path / "config.json").write_text('{"model": "custom", "checker": 5}', encoding="utf-8")
        assert "checker" not in config.load_config()
        config.save_config({"checker": "bad"})
        assert "checker" not in config.current_config()
