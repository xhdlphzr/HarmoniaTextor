# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Flask application factory."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response

from app.context import get_export_service, get_service
from app.routes import register_routes
from harmoniatextor import __version__
from harmoniatextor.config import config_dir, current_config
from harmoniatextor.i18n import (
    DEFAULT_LANGUAGE,
    catalog_for,
    normalize_language,
    translate,
)
from harmoniatextor.render.exporter import ExportService
from harmoniatextor.service.service import CompositionService
from harmoniatextor.storage.project_store import ProjectStore

__all__ = ["create_app", "get_export_service", "get_service", "local_time"]

_ASSET_FILES = ("js/app.js", "css/style.css")

#: Responses that must never be cached, so the desktop webview always shows the
#: current works, live score and progress instead of a stale copy.
_NO_STORE_MIMETYPES = frozenset(
    {
        "text/html",
        "application/json",
        "application/vnd.recordare.musicxml+xml",
    }
)


def local_time(value: str) -> str:
    """Render a UTC ISO timestamp in the server's local timezone.

    Args:
        value: An ISO-8601 timestamp, normally in UTC.

    Returns:
        A ``YYYY-MM-DD HH:MM:SS`` local-time string, or ``value`` unchanged when
        it cannot be parsed.
    """
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _asset_version(static_folder: str | None) -> str:
    """Return a cache-busting token derived from static file timestamps.

    Args:
        static_folder: Flask static folder path.

    Returns:
        A version string such as ``"0.1.0-1726000000"``.
    """
    if static_folder is None:
        return __version__
    stamp = 0.0
    for name in _ASSET_FILES:
        path = Path(static_folder) / name
        if path.exists():
            stamp = max(stamp, path.stat().st_mtime)
    return f"{__version__}-{int(stamp)}"


def create_app(
    data_dir: str | Path | None = None,
    vendor_dir: str | Path | None = None,
    *,
    testing: bool = False,
) -> Flask:
    """Create the Flask application.

    Args:
        data_dir: Directory for project data; defaults to ``HARMONIA_DATA`` or
            ``~/.harmonia_textor``.
        vendor_dir: Directory of vendored binaries; defaults to
            ``HARMONIA_VENDOR`` or ``vendor``.
        testing: Whether to enable testing mode.

    Returns:
        The configured Flask application.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["TESTING"] = testing
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    data = Path(data_dir or os.environ.get("HARMONIA_DATA") or config_dir())
    vendor = Path(
        vendor_dir or os.environ.get("HARMONIA_VENDOR") or (config_dir() / "vendor")
    )
    service = CompositionService(ProjectStore(data))
    service.interrupt_stale_generations()
    app.extensions["harmonia_service"] = service
    app.extensions["harmonia_export"] = ExportService(service, vendor)

    @app.after_request
    def _disable_caching(response: Response) -> Response:
        """Stop the desktop webview from caching pages and API responses.

        Args:
            response: The response about to be sent.

        Returns:
            The response with no-store headers for cacheable HTML/JSON/score
            payloads.
        """
        if response.mimetype in _NO_STORE_MIMETYPES:
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.context_processor
    def _inject_assets() -> dict[str, Any]:
        """Expose the static asset version and localisation to templates.

        Returns:
            A mapping with ``asset_version``, ``lang``, ``t`` and ``i18n_json``.
        """
        language = normalize_language(
            current_config().get("language", DEFAULT_LANGUAGE)
        )
        return {
            "asset_version": _asset_version(app.static_folder),
            "lang": language,
            "t": lambda text, **kwargs: translate(text, language, **kwargs),
            "i18n_json": json.dumps(catalog_for(language), ensure_ascii=False),
        }

    app.add_template_filter(local_time, "localtime")
    register_routes(app)
    return app
