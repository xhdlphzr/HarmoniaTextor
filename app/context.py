# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Accessors for services attached to a Flask application."""

from __future__ import annotations

from flask import Flask

from harmoniatextor.render.exporter import ExportService
from harmoniatextor.service.service import CompositionService

__all__ = ["get_export_service", "get_service"]


def get_service(app: Flask) -> CompositionService:
    """Return the composition service bound to an app.

    Args:
        app: Flask application.

    Returns:
        The composition service.
    """
    service: CompositionService = app.extensions["harmonia_service"]
    return service


def get_export_service(app: Flask) -> ExportService:
    """Return the export service bound to an app.

    Args:
        app: Flask application.

    Returns:
        The export service.
    """
    export: ExportService = app.extensions["harmonia_export"]
    return export
