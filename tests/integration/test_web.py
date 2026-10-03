# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the Flask audition desk."""

from __future__ import annotations

from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient
from langchain_core.messages import AIMessage

from app.app import (
    _asset_version,
    create_app,
    get_export_service,
    get_service,
    local_time,
)
from app.jobs import Job
from app.routes import _stream
from harmoniatextor import __version__
from harmoniatextor.config import save_config
from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.score.io import new_score, to_musicxml
from harmoniatextor.score.streamops import ScoreEditor

_HTTP_OK = 200
_HTTP_REDIRECT = 302
_HTTP_BAD_REQUEST = 400
_HTTP_NOT_FOUND = 404


def _call(name: str, args: dict[str, object]) -> AIMessage:
    """Build an assistant message requesting one tool."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"{name}-1", "type": "tool_call"}
        ],
    )


_PLAN_STEPS: tuple[tuple[str, dict[str, object]], ...] = (
    ("add_movement", {}),
    ("set_movement_prompt", {"movement": 1, "prompt": "写一乐章"}),
)


class FakeChatModel:
    """A chat model that plans, composes and approves every review."""

    def __init__(self, **_kwargs: object) -> None:
        """Initialise an empty tool list.

        Args:
            _kwargs: Factory keyword arguments (ignored).
        """
        self.tools: list[str] = []
        self.calls = 0

    def bind_tools(self, tools: list[object], **_kwargs: object) -> FakeChatModel:
        """Remember the bound tool names.

        Args:
            tools: Bound tools.
            _kwargs: Extra keyword arguments (ignored).

        Returns:
            This model.
        """
        self.tools = [str(getattr(tool, "name", "")) for tool in tools]
        return self

    def invoke(self, _messages: object, **_kwargs: object) -> AIMessage:
        """Return the next scripted planning, composing or review message.

        Args:
            _messages: Conversation messages (ignored).
            _kwargs: Extra keyword arguments (ignored).

        Returns:
            The scripted assistant message.
        """
        self.calls += 1
        if "submit_review" in self.tools:
            return _call("submit_review", {"passed": True, "suggestions": ""})
        if "add_movement" in self.tools:
            index = self.calls - 1
            if index < len(_PLAN_STEPS):
                name, args = _PLAN_STEPS[index]
                return _call(name, args)
            return AIMessage(content="规划完成")
        if "edit" in self.tools and self.calls % 2 == 1:
            return _call(
                "edit", {"measure": 1, "voice": "soprano", "musicxml": theme_xml()}
            )
        return AIMessage(content="done")


def theme_xml() -> str:
    """Build a simple theme melody."""
    score = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["soprano"])
    ScoreEditor(score).write_line(
        "soprano", 1, [ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)]
    )
    return to_musicxml(score)


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Flask:
    """Return a testing Flask app with an isolated configuration directory."""
    monkeypatch.setenv("HARMONIA_DOWNLOADS", str(tmp_path / "downloads"))
    monkeypatch.setattr("harmoniatextor.config.config_dir", lambda: tmp_path / "config")
    return create_app(tmp_path / "data", tmp_path / "vendor", testing=True)


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Return a test client."""
    return app.test_client()


def make_work(app: Flask) -> tuple[str, str]:
    """Create a work directly through the service."""
    service = get_service(app)
    work = service.create_work("Demo", "plain", "C")
    return work.id, work.movements[0].id


class TestPages:
    """HTML pages."""

    def test_index_empty(self, client: FlaskClient) -> None:
        """The index renders with no works."""
        response = client.get("/")
        assert response.status_code == _HTTP_OK
        assert "HarmoniaTextor" in response.get_data(as_text=True)

    def test_language_defaults_to_english(self, client: FlaskClient) -> None:
        """The interface is English by default."""
        page = client.get("/").get_data(as_text=True)
        assert 'lang="en"' in page
        assert "Create" in page

    def test_language_switch_to_chinese(self, client: FlaskClient) -> None:
        """The configured language is rendered."""
        save_config({"language": "zh"})
        page = client.get("/").get_data(as_text=True)
        assert 'lang="zh-CN"' in page
        assert "创作规划" in page

    def test_index_no_store(self, client: FlaskClient) -> None:
        """Pages are not cached by the desktop webview."""
        response = client.get("/")
        assert "no-store" in response.headers.get("Cache-Control", "")
        response.close()

    def test_history_persists_across_restarts(self, tmp_path: Path) -> None:
        """Works survive an application restart on the same data directory."""
        first = create_app(
            data_dir=tmp_path, vendor_dir=tmp_path / "vendor", testing=True
        )
        work = get_service(first).create_work("持久作品", "plain", "C")
        second = create_app(
            data_dir=tmp_path, vendor_dir=tmp_path / "vendor", testing=True
        )
        assert work.id in get_service(second).list_works()
        page = second.test_client().get("/").get_data(as_text=True)
        assert "持久作品" in page

    def test_create_and_view(self, client: FlaskClient) -> None:
        """Creating a work redirects to its workspace."""
        response = client.post("/works", data={"title": "Demo", "genre": "plain"})
        assert response.status_code == _HTTP_REDIRECT
        location = response.headers["Location"]
        page = client.get(location)
        assert page.status_code == _HTTP_OK
        assert "Demo" in page.get_data(as_text=True)

    def test_index_with_work(self, app: Flask, client: FlaskClient) -> None:
        """The index lists existing works."""
        make_work(app)
        assert "Demo" in client.get("/").get_data(as_text=True)

    def test_index_with_empty_work(self, app: Flask, client: FlaskClient) -> None:
        """A work without movements still renders the index."""
        work = get_service(app).create_work(
            "空作品", "plain", "C", with_movements=False
        )
        response = client.get("/")
        assert response.status_code == _HTTP_OK
        assert work.title in response.get_data(as_text=True)

    def test_index_shows_style_chip(self, app: Flask, client: FlaskClient) -> None:
        """The history list shows the work's style."""
        get_service(app).create_work("Demo", "plain", "C", style="impressionist")
        page = client.get("/").get_data(as_text=True)
        assert 'chip soft">Impressionist</span>' in page

    def test_index_legacy_work_defaults_to_baroque(
        self, app: Flask, client: FlaskClient
    ) -> None:
        """A work without a stored style is labelled Baroque."""
        service = get_service(app)
        work = service.create_work("Legacy", "plain", "C")
        work.style = None
        service.store.save_work(work)
        page = client.get("/").get_data(as_text=True)
        assert 'chip soft">Baroque</span>' in page

    def test_index_marks_interrupted_generation(
        self, app: Flask, client: FlaskClient
    ) -> None:
        """An interrupted generation is marked in the index."""
        service = get_service(app)
        work = service.create_work("中断作品", "plain", "C", with_movements=False)
        service.start_generation(work.id, "写一段")
        service.interrupt_stale_generations()
        page = client.get("/").get_data(as_text=True)
        assert "Interrupted" in page

    def test_index_marks_failed_generation(
        self, app: Flask, client: FlaskClient
    ) -> None:
        """A generation stopped by an error is marked in the index."""
        service = get_service(app)
        work = service.create_work("失败作品", "plain", "C", with_movements=False)
        service.start_generation(work.id, "写一段")
        service.finish_generation(work.id, False)
        page = client.get("/").get_data(as_text=True)
        assert "Failed" in page

    def test_index_marks_running_generation(
        self, app: Flask, client: FlaskClient
    ) -> None:
        """A generation still running is marked in the index."""
        service = get_service(app)
        work = service.create_work("生成中作品", "plain", "C", with_movements=False)
        service.start_generation(work.id, "写一段")
        page = client.get("/").get_data(as_text=True)
        assert "Running" in page

    def test_work_without_movements_redirects(
        self, app: Flask, client: FlaskClient
    ) -> None:
        """A work without movements redirects to the index."""
        work = get_service(app).create_work(
            "空作品", "plain", "C", with_movements=False
        )
        assert client.get(f"/works/{work.id}").status_code == _HTTP_REDIRECT

    def test_work_with_review(self, app: Flask, client: FlaskClient) -> None:
        """A stored reviewer verdict is shown on the work page."""
        work_id, movement_id = make_work(app)
        get_service(app).record_review(work_id, movement_id, False, "节奏太整齐")
        page = client.get(f"/works/{work_id}").get_data(as_text=True)
        assert "Reviewer" in page
        assert "节奏太整齐" in page

    def test_work_with_plan(self, app: Flask, client: FlaskClient) -> None:
        """A stored creation plan is shown on the work page."""
        work_id, movement_id = make_work(app)
        get_service(app).record_plan(work_id, movement_id, "钢琴:平静到激昂")
        page = client.get(f"/works/{work_id}").get_data(as_text=True)
        assert "Plan" in page
        assert "钢琴:平静到激昂" in page

    def test_favicon(self, client: FlaskClient) -> None:
        """The favicon route serves the bundled PNG."""
        response = client.get("/favicon.ico")
        assert response.status_code == _HTTP_OK
        assert response.content_type == "image/png"
        assert response.get_data()
        response.close()

    def test_favicon_missing(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing favicon yields a 404."""
        monkeypatch.setattr("app.routes.favicon_icon", lambda: None)
        assert client.get("/favicon.ico").status_code == _HTTP_NOT_FOUND


class TestApi:
    """JSON and score endpoints."""

    def test_score(self, app: Flask, client: FlaskClient) -> None:
        """The score endpoint returns MusicXML."""
        work_id, movement_id = make_work(app)
        get_service(app).submit_theme(work_id, movement_id, theme_xml(), check=False)
        response = client.get(f"/api/works/{work_id}/movements/{movement_id}/score")
        assert response.status_code == _HTTP_OK
        assert "score-partwise" in response.get_data(as_text=True)

    def test_score_merged_movements(self, app: Flask, client: FlaskClient) -> None:
        """A work with composed movements serves its live merged score."""
        service = get_service(app)
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.submit_theme(work.id, "m01", theme_xml(), check=False)
        response = client.get(f"/api/works/{work.id}/movements/m01/score")
        assert "score-partwise" in response.get_data(as_text=True)

    def test_score_empty(self, app: Flask, client: FlaskClient) -> None:
        """A work without movements returns an empty score."""
        service = get_service(app)
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        response = client.get(f"/api/works/{work.id}/movements/m01/score")
        assert response.status_code == _HTTP_OK
        assert response.get_data(as_text=True) == ""

    def test_score_fallback_revision(self, app: Flask, client: FlaskClient) -> None:
        """A movement with a voice but no notes falls back to its revision."""
        service = get_service(app)
        work = service.create_work("Demo", "plain", "C")
        movement_id = work.movements[0].id
        service.add_part(work.id, movement_id, "flute", "Flute", check=False)
        response = client.get(f"/api/works/{work.id}/movements/{movement_id}/score")
        assert response.status_code == _HTTP_OK
        assert "score-partwise" in response.get_data(as_text=True)

    def test_score_failure(
        self, app: Flask, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing score read returns an empty document, not an HTML page."""
        work_id, movement_id = make_work(app)
        service = get_service(app)

        def boom(*_args: object, **_kwargs: object) -> str:
            raise RuntimeError("boom")

        monkeypatch.setattr(service, "merged_musicxml", boom)
        response = client.get(f"/api/works/{work_id}/movements/{movement_id}/score")
        assert response.status_code == _HTTP_OK
        assert response.get_data(as_text=True) == ""

    def test_check(self, app: Flask, client: FlaskClient) -> None:
        """The check endpoint reports the empty score."""
        work_id, movement_id = make_work(app)
        data = client.post(
            f"/api/works/{work_id}/movements/{movement_id}/check"
        ).get_json()
        assert data["ok"] is False
        assert any(item["rule_id"] == "empty" for item in data["violations"])

    def test_finalize(self, app: Flask, client: FlaskClient) -> None:
        """Finalisation returns a result."""
        work_id, movement_id = make_work(app)
        data = client.post(
            f"/api/works/{work_id}/movements/{movement_id}/finalize"
        ).get_json()
        assert "ok" in data

    def test_rollback_json(self, app: Flask, client: FlaskClient) -> None:
        """Rollback accepts JSON."""
        service = get_service(app)
        work_id, movement_id = make_work(app)
        service.edit_measure(
            work_id, movement_id, 1, "soprano", theme_xml(), check=False
        )
        data = client.post(
            f"/api/works/{work_id}/movements/{movement_id}/rollback",
            json={"seq": 0},
        ).get_json()
        assert data["ok"] is True

    def test_rollback_form(self, app: Flask, client: FlaskClient) -> None:
        """Rollback accepts form data."""
        work_id, movement_id = make_work(app)
        get_service(app).submit_theme(work_id, movement_id, theme_xml(), check=False)
        data = client.post(
            f"/api/works/{work_id}/movements/{movement_id}/rollback",
            data={"seq": "0"},
        ).get_json()
        assert data["ok"] is True

    def test_audit_json(self, app: Flask, client: FlaskClient) -> None:
        """Audit notes accept JSON."""
        work_id, movement_id = make_work(app)
        data = client.post(
            f"/api/works/{work_id}/movements/{movement_id}/audit",
            json={"note": "很好"},
        ).get_json()
        assert data["ok"] is True

    def test_audit_form(self, app: Flask, client: FlaskClient) -> None:
        """Audit notes accept form data."""
        work_id, movement_id = make_work(app)
        data = client.post(
            f"/api/works/{work_id}/movements/{movement_id}/audit",
            data={"note": "很好"},
        ).get_json()
        assert data["ok"] is True

    def test_agent(
        self, app: Flask, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The agent endpoint runs a session."""
        monkeypatch.setattr("app.routes.create_chat_model", FakeChatModel)
        work_id, movement_id = make_work(app)
        data = client.post(
            f"/api/works/{work_id}/movements/{movement_id}/agent",
            json={"goal": "写一个主题"},
        ).get_json()
        assert data["completed"] is True
        assert data["final_text"] == "done"

    def test_export_musicxml(self, app: Flask, client: FlaskClient) -> None:
        """MusicXML export returns a file."""
        work_id, movement_id = make_work(app)
        response = client.get(
            f"/works/{work_id}/movements/{movement_id}/export/musicxml"
        )
        assert response.status_code == _HTTP_OK
        assert response.get_data()
        response.close()

    def test_export_musicxml_merged_movements(
        self, app: Flask, client: FlaskClient
    ) -> None:
        """A composed work exports its merged score."""
        service = get_service(app)
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.submit_theme(work.id, "m01", theme_xml(), check=False)
        response = client.get(f"/works/{work.id}/movements/m01/export/musicxml")
        assert response.status_code == _HTTP_OK
        assert b"score-partwise" in response.get_data()
        response.close()

    def test_export_all_movements_single(self, app: Flask, client: FlaskClient) -> None:
        """A composed work exports one merged file."""
        service = get_service(app)
        work = service.create_work("Demo", "plain", "C", with_movements=False)
        service.add_movement(work.id)
        service.submit_theme(work.id, "m01", theme_xml(), check=False)
        response = client.get(f"/works/{work.id}/export/musicxml")
        assert response.status_code == _HTTP_OK
        assert b"score-partwise" in response.get_data()
        response.close()

    def test_export_midi(self, app: Flask, client: FlaskClient) -> None:
        """MIDI export returns a file."""
        work_id, movement_id = make_work(app)
        response = client.get(f"/works/{work_id}/movements/{movement_id}/export/midi")
        assert response.status_code == _HTTP_OK
        assert response.get_data()
        response.close()

    def test_export_m4a_unavailable(self, app: Flask, client: FlaskClient) -> None:
        """M4A export reports missing binaries."""
        work_id, movement_id = make_work(app)
        response = client.get(f"/works/{work_id}/movements/{movement_id}/export/m4a")
        assert response.status_code == _HTTP_BAD_REQUEST

    def test_export_mp3_unavailable(self, app: Flask, client: FlaskClient) -> None:
        """MP3 export reports missing binaries."""
        work_id, movement_id = make_work(app)
        response = client.get(f"/works/{work_id}/movements/{movement_id}/export/mp3")
        assert response.status_code == _HTTP_BAD_REQUEST

    def test_export_unsupported(self, app: Flask, client: FlaskClient) -> None:
        """Unknown formats are rejected."""
        work_id, movement_id = make_work(app)
        response = client.get(f"/works/{work_id}/movements/{movement_id}/export/ogg")
        assert response.status_code == _HTTP_BAD_REQUEST

    def test_export_save_json(self, app: Flask, client: FlaskClient) -> None:
        """Saving reports the written path instead of downloading."""
        work_id, movement_id = make_work(app)
        data = client.get(
            f"/works/{work_id}/movements/{movement_id}/export/musicxml?save=1"
        ).get_json()
        assert data["ok"] is True
        assert data["path"]

    def test_export_all_single(self, app: Flask, client: FlaskClient) -> None:
        """A single-movement work exports a single file."""
        work_id, _movement_id = make_work(app)
        response = client.get(f"/works/{work_id}/export/musicxml")
        assert response.status_code == _HTTP_OK
        assert response.get_data()
        response.close()

    def test_export_all_zip(self, app: Flask, client: FlaskClient) -> None:
        """A multi-movement work exports a zip archive."""
        work = get_service(app).create_work("Demo", "sonata", "C")
        response = client.get(f"/works/{work.id}/export/musicxml")
        assert response.status_code == _HTTP_OK
        assert response.get_data()[:2] == b"PK"
        response.close()

    def test_export_all_save_json(self, app: Flask, client: FlaskClient) -> None:
        """Saving a whole work reports the written path."""
        work_id, _movement_id = make_work(app)
        data = client.get(f"/works/{work_id}/export/musicxml?save=1").get_json()
        assert data["ok"] is True
        assert data["path"]

    def test_export_all_unsupported(self, app: Flask, client: FlaskClient) -> None:
        """Unknown work-level formats are rejected."""
        work_id, _movement_id = make_work(app)
        assert (
            client.get(f"/works/{work_id}/export/ogg").status_code == _HTTP_BAD_REQUEST
        )


class TestConfigApi:
    """LLM configuration endpoints."""

    def test_get_config(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The config endpoint returns the effective configuration."""
        monkeypatch.setattr("app.routes.current_config", lambda: {"model": "m"})
        assert client.get("/api/config").get_json() == {"model": "m"}

    def test_set_config(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Posting configuration values persists them."""
        saved: dict[str, object] = {}

        def fake_save(values: dict[str, object]) -> None:
            saved.update(values)

        monkeypatch.setattr("app.routes.save_config", fake_save)
        monkeypatch.setattr("app.routes.current_config", lambda: {"model": "m"})
        data = client.post("/api/config", json={"model": "m"}).get_json()
        assert data["ok"] is True
        assert saved == {"model": "m"}

    def test_set_config_invalid(self, client: FlaskClient) -> None:
        """A non-object payload is rejected."""
        assert client.post("/api/config", json=[1, 2]).status_code == _HTTP_BAD_REQUEST

    def test_llm_status_probes_once(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The status endpoint probes while unknown and then caches."""
        monkeypatch.setattr("app.routes.check_connection", lambda: "ok")
        assert client.get("/api/llm-status").get_json() == {"status": "ok"}
        assert client.get("/api/llm-status").get_json() == {"status": "ok"}

    def test_llm_ping(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ping endpoint forces a fresh probe."""
        monkeypatch.setattr("app.routes.check_connection", lambda: "error")
        assert client.post("/api/llm-ping").get_json() == {"status": "error"}


class TestKitsApi:
    """Style-kit management endpoints."""

    def test_list_kits(self, client: FlaskClient) -> None:
        """Built-in kits are listed."""
        data = client.get("/api/kits").get_json()
        assert any(kit["id"] == "baroque" for kit in data["kits"])

    def test_create_kit(self, client: FlaskClient) -> None:
        """A custom kit can be created."""
        data = client.post(
            "/api/kits",
            json={"name": "Mine", "rules": ["empty"], "techniques": ["imitation"]},
        ).get_json()
        assert data["ok"] is True
        assert data["kit"]["id"].startswith("s-")

    def test_create_kit_invalid(self, client: FlaskClient) -> None:
        """An unknown rule is rejected."""
        response = client.post(
            "/api/kits",
            json={"name": "Mine", "rules": ["nope"], "techniques": ["imitation"]},
        )
        assert response.status_code == _HTTP_BAD_REQUEST

    def test_create_kit_non_list(self, client: FlaskClient) -> None:
        """A non-list selection is treated as empty."""
        data = client.post(
            "/api/kits",
            json={"name": "Mine", "rules": "nope", "techniques": "nope"},
        ).get_json()
        assert data["ok"] is True
        assert data["kit"]["rules"] == []

    def test_rename_kit(self, client: FlaskClient) -> None:
        """A custom kit can be renamed."""
        created = client.post(
            "/api/kits",
            json={"name": "Mine", "rules": ["empty"], "techniques": ["imitation"]},
        ).get_json()
        kit_id = created["kit"]["id"]
        data = client.put(f"/api/kits/{kit_id}", json={"name": "Yours"}).get_json()
        assert data["ok"] is True
        assert data["kit"]["name"] == "Yours"

    def test_rename_builtin(self, client: FlaskClient) -> None:
        """A built-in kit cannot be renamed."""
        response = client.put("/api/kits/baroque", json={"name": "X"})
        assert response.status_code == _HTTP_BAD_REQUEST

    def test_rename_unknown(self, client: FlaskClient) -> None:
        """Renaming a missing kit is a 404."""
        response = client.put("/api/kits/s-missing", json={"name": "X"})
        assert response.status_code == _HTTP_NOT_FOUND

    def test_delete_kit(self, client: FlaskClient) -> None:
        """A custom kit can be deleted."""
        created = client.post(
            "/api/kits",
            json={"name": "Mine", "rules": ["empty"], "techniques": ["imitation"]},
        ).get_json()
        kit_id = created["kit"]["id"]
        assert client.delete(f"/api/kits/{kit_id}").get_json()["ok"] is True

    def test_delete_builtin(self, client: FlaskClient) -> None:
        """A built-in kit cannot be deleted."""
        assert client.delete("/api/kits/baroque").status_code == _HTTP_BAD_REQUEST

    def test_delete_unknown(self, client: FlaskClient) -> None:
        """Deleting a missing kit is a 404."""
        assert client.delete("/api/kits/s-missing").status_code == _HTTP_NOT_FOUND


class TestJobs:
    """Background generation and progress streaming."""

    def test_generate_requires_prompt(self, client: FlaskClient) -> None:
        """An empty prompt is rejected."""
        assert client.post("/api/generate", json={}).status_code == _HTTP_BAD_REQUEST

    def test_generate_and_stream(
        self, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generation starts a job whose events stream to completion."""
        monkeypatch.setattr("app.routes.create_chat_model", FakeChatModel)
        response = client.post(
            "/api/generate",
            json={"prompt": "写一首赋格", "genre": "plain", "tonic": "C"},
        )
        data = response.get_json()
        assert data["ok"] is True
        events = client.get(f"/api/jobs/{data['job_id']}/events").get_data(as_text=True)
        assert "event: done" in events
        assert '"ok": true' in events
        assert '"plan_tree"' in events
        assert "写一乐章" in events

    def test_run_unknown_work(self, client: FlaskClient) -> None:
        """Continuing a missing work is a 404."""
        assert (
            client.post("/api/works/nope/movements/m01/run", json={}).status_code
            == _HTTP_NOT_FOUND
        )

    def test_run_existing_work(
        self, app: Flask, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Continuing an existing work with feedback starts a job."""
        monkeypatch.setattr("app.routes.create_chat_model", FakeChatModel)
        work_id, movement_id = make_work(app)
        data = client.post(
            f"/api/works/{work_id}/movements/{movement_id}/run",
            json={"prompt": "继续", "feedback": "多些模仿"},
        ).get_json()
        assert data["ok"] is True
        events = client.get(f"/api/jobs/{data['job_id']}/events").get_data(as_text=True)
        assert "event: done" in events

    def test_unknown_job(self, client: FlaskClient) -> None:
        """Streaming an unknown job is a 404."""
        assert client.get("/api/jobs/nope/events").status_code == _HTTP_NOT_FOUND

    def test_stream_keepalive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A running job with no events emits a keepalive."""
        monkeypatch.setattr("app.routes._SSE_TIMEOUT", 0.01)
        job = Job("j")
        stream = _stream(job)
        assert next(stream) == ": keepalive\n\n"
        job.emit({"kind": "a"})
        job.finish("done", {"ok": True})
        rest = "".join(stream)
        assert '"kind": "a"' in rest
        assert "event: done" in rest


class TestContext:
    """Service accessors."""

    def test_accessors(self, app: Flask) -> None:
        """The service accessors return bound services."""
        assert get_service(app) is get_export_service(app).service

    def test_asset_version(self, tmp_path: Path) -> None:
        """The asset version falls back to the package version without a folder."""
        assert _asset_version(None) == __version__
        (tmp_path / "js").mkdir()
        (tmp_path / "js" / "app.js").write_text("x", encoding="utf-8")
        assert _asset_version(str(tmp_path)).startswith(__version__)


class TestLocalTime:
    """Local-time rendering of stored UTC timestamps."""

    def test_empty(self) -> None:
        """An empty timestamp stays empty."""
        assert local_time("") == ""

    def test_invalid(self) -> None:
        """Unparseable timestamps are returned unchanged."""
        assert local_time("not-a-date") == "not-a-date"

    def test_aware(self) -> None:
        """A UTC timestamp converts to the local timezone."""
        rendered = local_time("2026-01-02T03:04:05+00:00")
        assert len(rendered) == len("2026-01-02 03:04:05")
        assert rendered[4] == "-" and rendered[10] == " "

    def test_naive(self) -> None:
        """A timestamp without a timezone is assumed to be UTC."""
        rendered = local_time("2026-01-02T03:04:05")
        assert rendered.endswith(":05")
