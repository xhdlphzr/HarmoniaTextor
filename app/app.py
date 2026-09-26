# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Flask application factory."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from flask import Flask

from app.context import get_export_service, get_service
from app.routes import register_routes
from harmoniatextor import __version__
from harmoniatextor.config import config_dir
from harmoniatextor.render.exporter import ExportService
from harmoniatextor.service.service import CompositionService
from harmoniatextor.storage.project_store import ProjectStore

__all__ = ["create_app", "get_export_service", "get_service", "local_time"]

_ASSET_FILES = ("js/app.js", "css/style.css")


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
    vendor = Path(vendor_dir or os.environ.get("HARMONIA_VENDOR", "vendor"))
    service = CompositionService(ProjectStore(data))
    service.interrupt_stale_generations()
    app.extensions["harmonia_service"] = service
    app.extensions["harmonia_export"] = ExportService(service, vendor)

    @app.context_processor
    def _inject_assets() -> dict[str, str]:
        """Expose the static asset version to templates.

        Returns:
            A mapping with the ``asset_version`` key.
        """
        return {"asset_version": _asset_version(app.static_folder)}

    app.add_template_filter(local_time, "localtime")
    register_routes(app)
    return app
