# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Desktop shell for HarmoniaTextor built on pywebview.

The Flask application runs on a background thread and is displayed inside a
native window.  The webview module is imported lazily so that the import graph
stays free of the optional desktop dependencies until launch.
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from werkzeug.serving import BaseWSGIServer, make_server

from app.app import create_app
from harmoniatextor.render.vendor import ensure_async

__all__ = ["DesktopApp", "main"]

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 860
DEFAULT_TITLE = "HarmoniaTextor"


def _load_webview() -> ModuleType:
    """Import and return the pywebview module.

    Returns:
        The imported ``webview`` module.
    """
    return importlib.import_module("webview")


def _icon_name() -> str | None:
    """Return the window-icon file name for the current platform.

    Returns:
        ``"Franx.ico"`` on Windows, ``"Franx.icns"`` on macOS, else ``None``.
    """
    platform: str = sys.platform
    if platform.startswith("win"):
        return "Franx.ico"
    if platform == "darwin":
        return "Franx.icns"
    return None


def _icon_candidates() -> list[Path]:
    """Return candidate paths for the native window icon.

    Returns:
        Candidate paths, most specific (bundled) first.
    """
    name = _icon_name()
    if name is None:
        return []
    roots: list[Path] = []
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is not None:
        roots.append(Path(bundle))
    roots.append(Path(__file__).resolve().parent.parent)
    return [root / "assets" / name for root in roots]


def _window_icon() -> str | None:
    """Return the first existing window-icon path.

    Returns:
        The icon path, or ``None`` when no icon is available.
    """
    for candidate in _icon_candidates():
        if candidate.exists():
            return str(candidate)
    return None


class DesktopApp:
    """A native window wrapping the Flask audition desk.

    Attributes:
        flask_app: The WSGI application to serve.
        webview_module: The pywebview module (or a test double).
        host: Bind address.
        port: Bind port; ``0`` selects a free port.
    """

    def __init__(
        self,
        flask_app: Callable[..., Any],
        webview_module: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        """Initialise the desktop shell.

        Args:
            flask_app: The Flask application to serve.
            webview_module: The pywebview module.
            host: Bind address.
            port: Bind port; ``0`` selects a free port.
        """
        self.flask_app = flask_app
        self.webview_module = webview_module
        self.host = host
        self.port = port

    def _mark_interrupted(self) -> None:
        """Best-effort: mark still-running generations as interrupted.

        When the window is closed the daemon job threads are killed without
        running their cleanup, so the current run never gets a completion tag.
        Flushing the interrupted marker here means the work is already recorded
        as interrupted the moment the app closes, not only after the next start.
        """
        extensions = getattr(self.flask_app, "extensions", None)
        if not isinstance(extensions, dict):
            return
        service = extensions.get("harmonia_service")
        if service is not None:
            service.interrupt_stale_generations()

    def run(self) -> None:
        """Serve the application and open the native window."""
        server: BaseWSGIServer = make_server(self.host, self.port, self.flask_app, threaded=True)
        url = f"http://{self.host}:{server.server_port}/"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.webview_module.create_window(
                DEFAULT_TITLE,
                url,
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
            )
            self.webview_module.start(icon=_window_icon())
        finally:
            self._mark_interrupted()
            server.shutdown()
            thread.join(timeout=5)


def _ensure_audio_backend(flask_app: Any) -> None:
    """Best-effort first-launch download of the audio backend.

    Args:
        flask_app: The Flask application that carries the export service.
    """
    extensions = getattr(flask_app, "extensions", None)
    if not isinstance(extensions, dict):
        return
    export = extensions.get("harmonia_export")
    if export is not None:
        ensure_async(export.vendor_root)


def main() -> None:
    """Launch the desktop application."""
    flask_app = create_app()
    _ensure_audio_backend(flask_app)
    DesktopApp(flask_app, _load_webview()).run()
