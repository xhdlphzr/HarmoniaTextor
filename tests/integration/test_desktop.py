# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the pywebview desktop shell."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from werkzeug.serving import make_server as real_make_server

from app import desktop

_ICON_CANDIDATE_COUNT = 2


class FakeWebview:
    """A minimal stand-in for the pywebview module."""

    def __init__(self) -> None:
        """Initialise an empty fake."""
        self.windows: list[tuple[str, str, dict[str, Any]]] = []
        self.started = False
        self.icon: str | None = None

    def create_window(self, title: str, url: str, **kwargs: Any) -> object:
        """Record a window creation request.

        Args:
            title: The title.
            url: The request URL.
            kwargs: Forwarded keyword arguments.

        Returns:
            The create window result.
        """
        self.windows.append((title, url, kwargs))
        return object()

    def start(self, **kwargs: Any) -> None:
        """Record that the event loop started and the requested icon.

        Args:
            kwargs: Forwarded keyword arguments.
        """
        self.icon = kwargs.get("icon")
        self.started = True


def wsgi_app(_environ: dict[str, Any], start_response: Any) -> list[bytes]:
    """Return a minimal WSGI application.

    Args:
        _environ: The environ.
        start_response: The start response.

    Returns:
        The wsgi app result.
    """
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"ok"]


class FakeService:
    """A service double that counts interrupt calls."""

    def __init__(self) -> None:
        """Initialise the counter."""
        self.calls = 0

    def interrupt_stale_generations(self) -> int:
        """Record one interrupt call.

        Returns:
            The resulting number.
        """
        self.calls += 1
        return 0


class FakeExport:
    """An export service double exposing a vendor root."""

    vendor_root = "vendir"


class FakeFlaskApp:
    """A callable WSGI app carrying Flask-style extensions."""

    def __init__(self, service: object) -> None:
        """Store a single service extension.

        Args:
            service: The composition service.
        """
        self.extensions: dict[str, object] = {"harmonia_service": service}

    def __call__(self, environ: dict[str, Any], start_response: Any) -> list[bytes]:
        """Delegate to the minimal WSGI app.

        Args:
            environ: The environ.
            start_response: The start response.

        Returns:
            The call result.
        """
        return wsgi_app(environ, start_response)


class TestDesktop:
    """Desktop shell behaviour."""

    def test_load_webview(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The webview module is imported lazily.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        fake = types.ModuleType("webview")
        monkeypatch.setitem(sys.modules, "webview", fake)
        assert desktop._load_webview() is fake

    def test_run(self) -> None:
        """Running the shell serves the app and opens a window."""
        webview = FakeWebview()
        app = desktop.DesktopApp(wsgi_app, webview)
        app.run()
        assert webview.started
        assert webview.windows[0][0] == "HarmoniaTextor"
        assert webview.windows[0][1].startswith("http://127.0.0.1:")

    def test_run_uses_threaded_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The server is threaded so SSE never blocks score requests.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        captured: dict[str, Any] = {}

        def spy(host: str, port: int, app: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return real_make_server(host, port, app, **kwargs)

        monkeypatch.setattr("app.desktop.make_server", spy)
        webview = FakeWebview()
        desktop.DesktopApp(wsgi_app, webview).run()
        assert captured.get("threaded") is True

    def test_ensure_audio_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The first launch kicks off the audio backend download.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        calls: list[object] = []
        monkeypatch.setattr(desktop, "ensure_async", calls.append)
        app = FakeFlaskApp(None)
        app.extensions["harmonia_export"] = FakeExport()
        desktop._ensure_audio_backend(app)
        assert calls == ["vendir"]

    def test_ensure_audio_backend_without_export(self) -> None:
        """An app without the export extension is ignored."""
        desktop._ensure_audio_backend(FakeFlaskApp(None))

    def test_mark_interrupted(self) -> None:
        """Closing the window marks running generations as interrupted."""
        service = FakeService()
        desktop.DesktopApp(FakeFlaskApp(service), FakeWebview())._mark_interrupted()
        assert service.calls == 1

    def test_mark_interrupted_without_service(self) -> None:
        """An app whose service extension is missing is ignored."""
        desktop.DesktopApp(FakeFlaskApp(None), FakeWebview())._mark_interrupted()

    def test_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The entry point wires the app and webview together.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        webview = FakeWebview()
        monkeypatch.setattr(desktop, "create_app", lambda: wsgi_app)
        monkeypatch.setattr(desktop, "_load_webview", lambda: webview)
        desktop.main()
        assert webview.started

    def test_icon_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The icon name follows the platform.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(sys, "platform", "win32")
        assert desktop._icon_name() == "Franx.ico"
        monkeypatch.setattr(sys, "platform", "darwin")
        assert desktop._icon_name() == "Franx.icns"
        monkeypatch.setattr(sys, "platform", "linux")
        assert desktop._icon_name() is None

    def test_icon_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bundled and repository roots are searched.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys, "_MEIPASS", "C:/bundle", raising=False)
        assert len(desktop._icon_candidates()) == _ICON_CANDIDATE_COUNT

    def test_icon_candidates_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Linux has no window icon.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(sys, "platform", "linux")
        assert desktop._icon_candidates() == []

    def test_window_icon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The repository icon is found, and missing icons return None.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(sys, "platform", "win32")
        assert desktop._window_icon() is not None
        monkeypatch.setattr(desktop, "_icon_candidates", list)
        assert desktop._window_icon() is None
