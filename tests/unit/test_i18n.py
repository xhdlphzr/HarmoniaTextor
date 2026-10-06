# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for the interface localisation module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from harmoniatextor.i18n import (
    CATALOG,
    DEFAULT_LANGUAGE,
    LANGUAGES,
    catalog_for,
    i18n_dir,
    normalize_language,
    translate,
)


class TestI18nDir:
    """Catalogue directory resolution."""

    def test_development(self) -> None:
        """Without a frozen bundle the repository ``i18n`` folder is used."""
        assert i18n_dir().name == "i18n"
        assert (i18n_dir() / "en.yaml").is_file()

    def test_frozen(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A frozen build resolves ``i18n`` next to the bundled files.

        Args:
            tmp_path: The pytest temporary path fixture.
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        assert i18n_dir() == tmp_path / "i18n"


class TestNormalizeLanguage:
    """Language code clamping."""

    def test_supported(self) -> None:
        """Supported codes are kept, case-insensitively."""
        assert normalize_language("en") == "en"
        assert normalize_language("zh") == "zh"
        assert normalize_language("ZH") == "zh"

    def test_fallback(self) -> None:
        """Unknown or empty values fall back to the default."""
        assert DEFAULT_LANGUAGE == "en"
        assert normalize_language(None) == "en"
        assert normalize_language("fr") == "en"


class TestCatalog:
    """Loaded YAML catalogues."""

    def test_languages(self) -> None:
        """The supported languages are en and zh."""
        assert LANGUAGES == ("en", "zh")
        assert set(CATALOG) == set(LANGUAGES)

    def test_ids_align(self) -> None:
        """Every language defines exactly the same message ids."""
        assert set(CATALOG["en"]) == set(CATALOG["zh"])

    def test_catalog_for(self) -> None:
        """The catalogue is selected by language code."""
        assert catalog_for("en") is CATALOG["en"]
        assert catalog_for("zh") is CATALOG["zh"]


class TestTranslate:
    """Translation and placeholder substitution."""

    def test_english(self) -> None:
        """A known id is translated to English."""
        assert translate("nav.create", "en") == "Create"

    def test_chinese(self) -> None:
        """A known id is translated to Chinese."""
        assert translate("nav.create", "zh") == "创作"

    def test_missing(self) -> None:
        """An unknown id falls back to the id itself."""
        assert translate("does.not.exist", "en") == "does.not.exist"

    def test_missing_falls_back_to_english(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing Chinese id falls back to the English text.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.delitem(CATALOG["zh"], "nav.create")
        assert translate("nav.create", "zh") == "Create"

    def test_placeholders(self) -> None:
        """Placeholders are substituted."""
        assert translate("theme.measure", "en", measure=3) == "bar 3"

    def test_rule_labels(self) -> None:
        """Rule names are localised."""
        assert translate("rule.pf5th", "en") == "Parallel fifths"
        assert translate("rule.pf5th", "zh") == "平行五度"

    def test_technique_labels(self) -> None:
        """Technique names are localised."""
        assert translate("technique.imitation", "en") == "Imitation"
        assert translate("technique.imitation", "zh") == "模仿"

    def test_violation_labels(self) -> None:
        """Violation phrases are localised."""
        assert translate("violation.parallel_fifth", "en") == "parallel perfect fifth"
        assert translate("violation.parallel_fifth", "zh") == "平行纯五度"
