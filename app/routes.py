# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""HTTP routes for the audition desk."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, url_for

from app.assets import favicon_icon
from app.context import get_export_service, get_service
from app.jobs import EventCallback, Job, JobManager
from harmoniatextor.agent.architect import Architect
from harmoniatextor.agent.health import UNKNOWN, check_connection
from harmoniatextor.agent.llm_factory import create_chat_model
from harmoniatextor.agent.loop import AgentLoop
from harmoniatextor.agent.movements import MovementComposer
from harmoniatextor.agent.reviewer import ReviewerAI
from harmoniatextor.checker.rules import BUILTIN_RULES
from harmoniatextor.config import context_window_tokens, current_config, save_config
from harmoniatextor.domain.enums import (
    TOOL_KIND_LABELS,
    VOICE_LABELS,
    WORK_STATUS_LABELS,
)
from harmoniatextor.service.service import DEFAULT_TITLE

__all__ = ["register_routes"]

_SSE_TIMEOUT = 15


def register_routes(app: Flask) -> None:
    """Register all routes on the application.

    Args:
        app: Flask application.
    """
    app.extensions["harmonia_jobs"] = JobManager()
    app.extensions["harmonia_llm_status"] = {"status": UNKNOWN}
    _register_pages(app)
    _register_config(app)
    _register_kits(app)
    _register_jobs(app)
    _register_api(app)
    _register_export(app)


def _register_pages(app: Flask) -> None:
    """Register the HTML page routes.

    Args:
        app: Flask application.
    """

    @app.get("/")
    def index() -> str:
        """Render the works index."""
        service = get_service(app)
        work_ids = service.list_works()
        works = [service.get_work(work_id) for work_id in work_ids]
        genres = service.genres.all()
        generation_states = {
            work_id: service.latest_generation_state(work_id) for work_id in work_ids
        }
        return render_template(
            "index.html",
            works=works,
            genres=genres,
            generation_states=generation_states,
            genre_labels={genre.id: f"genre.{genre.id}" for genre in genres},
            status_labels=WORK_STATUS_LABELS,
            style_kits=service.styles.all(),
            default_style=service.styles.resolve(None).id,
            rule_options=[{"id": rule.rule_id, "name": rule.name} for rule in BUILTIN_RULES],
            technique_options=[
                {"id": technique.id, "name": technique.name, "category": technique.category.value}
                for technique in service.techniques.all()
            ],
        )

    @app.get("/favicon.ico")
    def favicon() -> Any:
        """Serve the application favicon."""
        icon = favicon_icon()
        if icon is None:
            return Response(status=404)
        return send_file(icon, mimetype="image/png")

    @app.post("/works")
    def create_work() -> Any:
        """Create a work from form data and redirect to it."""
        service = get_service(app)
        title = request.form.get("title") or DEFAULT_TITLE
        genre = request.form.get("genre") or "plain"
        style = request.form.get("style") or None
        work = service.create_work(title, genre, style=style)
        return redirect(url_for("work", work_id=work.id))

    @app.get("/works/<work_id>")
    def work(work_id: str) -> Any:
        """Render a work workspace."""
        service = get_service(app)
        loaded = service.get_work(work_id)
        if not loaded.movements:
            return redirect(url_for("index"))
        movement_id = request.args.get("movement") or loaded.movements[0].id
        movement = service.get_movement(loaded, movement_id)
        revisions = service.store.load_revision_meta(work_id, movement_id)
        themes = service.store.load_themes(work_id, movement_id)
        genres = service.genres.all()
        return render_template(
            "work.html",
            work=loaded,
            movement=movement,
            revisions=revisions,
            themes=themes,
            review=service.latest_review(work_id),
            plan=service.latest_plan(work_id),
            genre_labels={genre.id: f"genre.{genre.id}" for genre in genres},
            status_labels=WORK_STATUS_LABELS,
            voice_labels=VOICE_LABELS,
            tool_labels=TOOL_KIND_LABELS,
            technique_labels={item.id: item.name for item in service.techniques.all()},
        )


def _register_config(app: Flask) -> None:
    """Register the configuration API.

    Args:
        app: Flask application.
    """

    @app.get("/api/config")
    def get_config() -> Response:
        """Return the effective LLM configuration."""
        return jsonify(current_config())

    @app.post("/api/config")
    def set_config() -> Any:
        """Persist LLM configuration values."""
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "invalid payload"}), 400
        save_config({str(key): value for key, value in payload.items()})
        _llm_state(app)["status"] = UNKNOWN
        return jsonify({"ok": True, "config": current_config()})

    @app.get("/api/llm-status")
    def llm_status() -> Response:
        """Return endpoint connectivity, probing once while unknown."""
        state = _llm_state(app)
        if state["status"] == UNKNOWN:
            refresh_llm_status(app)
        return jsonify({"status": state["status"]})

    @app.post("/api/llm-ping")
    def llm_ping() -> Response:
        """Force a connectivity probe of the configured endpoint."""
        return jsonify({"status": refresh_llm_status(app)})


def _as_list(value: Any) -> list[str]:
    """Coerce a JSON value into a list of strings.

    Args:
        value: Raw JSON value.

    Returns:
        A list of strings; empty when the value is not a list.
    """
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _register_kits(app: Flask) -> None:
    """Register the style-kit management API.

    Args:
        app: Flask application.
    """

    @app.get("/api/kits")
    def list_kits() -> Response:
        """Return every style kit."""
        return jsonify({"kits": [kit.to_dict() for kit in get_service(app).styles.all()]})

    @app.post("/api/kits")
    def create_kit() -> Any:
        """Create a custom style kit."""
        payload = request.get_json(silent=True) or {}
        try:
            kit = get_service(app).styles.create(
                str(payload.get("name", "")),
                _as_list(payload.get("rules")),
                _as_list(payload.get("techniques")),
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "kit": kit.to_dict()})

    @app.put("/api/kits/<kit_id>")
    def rename_kit(kit_id: str) -> Any:
        """Rename a custom style kit."""
        payload = request.get_json(silent=True) or {}
        try:
            kit = get_service(app).styles.rename(kit_id, str(payload.get("name", "")))
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except KeyError:
            return jsonify({"ok": False, "error": "unknown kit"}), 404
        return jsonify({"ok": True, "kit": kit.to_dict()})

    @app.delete("/api/kits/<kit_id>")
    def delete_kit(kit_id: str) -> Any:
        """Delete a custom style kit."""
        try:
            get_service(app).styles.delete(kit_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except KeyError:
            return jsonify({"ok": False, "error": "unknown kit"}), 404
        return jsonify({"ok": True})


def _register_jobs(app: Flask) -> None:
    """Register the background generation and progress routes.

    Args:
        app: Flask application.
    """

    @app.post("/api/generate")
    def generate() -> Any:
        """Create a work and start a fully automatic composer session."""
        payload = request.get_json(silent=True) or {}
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            return jsonify({"ok": False, "error": "提示词不能为空。"}), 400
        genre = str(payload.get("genre") or "plain")
        title = str(payload.get("title") or DEFAULT_TITLE)
        style = str(payload.get("style") or "") or None
        service = get_service(app)
        work = service.create_work(title, genre, style=style, with_movements=False)
        job = _start_architecture_job(app, service, work.id, prompt)
        return jsonify({"ok": True, "job_id": job.id, "work_id": work.id, "movement_id": "m01"})

    @app.post("/api/works/<work_id>/movements/<movement_id>/run")
    def run_work(work_id: str, movement_id: str) -> Any:
        """Continue an existing work with a fully automatic composer session."""
        payload = request.get_json(silent=True) or {}
        prompt = str(payload.get("prompt") or "继续完善这首作品。")
        feedback = payload.get("feedback")
        service = get_service(app)
        try:
            service.get_work(work_id)
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "unknown work"}), 404
        job = _start_agent_job(
            app, service, work_id, movement_id, prompt, feedback=str(feedback) if feedback else None
        )
        return jsonify(
            {"ok": True, "job_id": job.id, "work_id": work_id, "movement_id": movement_id}
        )

    @app.get("/api/jobs/<job_id>/events")
    def job_events(job_id: str) -> Any:
        """Stream job progress as server-sent events."""
        job = _jobs(app).get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "unknown job"}), 404
        return Response(_stream(job), mimetype="text/event-stream")


def _jobs(app: Flask) -> JobManager:
    """Return the job manager bound to an app.

    Args:
        app: Flask application.

    Returns:
        The job manager.
    """
    manager: JobManager = app.extensions["harmonia_jobs"]
    return manager


def _llm_state(app: Flask) -> dict[str, str]:
    """Return the mutable endpoint-connectivity state.

    Args:
        app: Flask application.

    Returns:
        The status mapping bound to the application.
    """
    state: dict[str, str] = app.extensions["harmonia_llm_status"]
    return state


def refresh_llm_status(app: Flask) -> str:
    """Probe the configured endpoint and cache the result.

    Args:
        app: Flask application.

    Returns:
        The freshly probed status code.
    """
    state = _llm_state(app)
    state["status"] = check_connection()
    return state["status"]


def _start_agent_job(  # noqa: PLR0913, PLR0917
    app: Flask,
    service: Any,
    work_id: str,
    movement_id: str,
    prompt: str,
    feedback: str | None = None,
    *,
    plan: bool = False,
) -> Job:
    """Create and start an automatic composer job.

    Args:
        app: Flask application.
        service: Composition service.
        work_id: Active work.
        movement_id: Active movement.
        prompt: Composition goal.
        feedback: Optional human audition feedback.
        plan: Whether to run the Step 1 planning phase first.

    Returns:
        The started job.
    """
    job = _jobs(app).create()
    service.start_generation(work_id, prompt)

    def target(emit: EventCallback) -> dict[str, Any]:
        finished = False
        try:
            window = context_window_tokens()
            loop = AgentLoop(
                service,
                create_chat_model(),
                reviewer=ReviewerAI(create_chat_model(), context_window=window),
                context_window=window,
            )
            result = loop.run(
                work_id, movement_id, prompt, feedback=feedback, on_event=emit, plan=plan
            )
            service.ensure_title(work_id)
            if result.review_passed is not None:
                service.record_review(
                    work_id, movement_id, result.review_passed, result.review_suggestions
                )
            report = service.check(work_id, movement_id)
            finished = True
            return {
                "work_id": work_id,
                "movement_id": movement_id,
                "completed": result.completed,
                "steps": result.steps,
                "final_text": result.final_text,
                "plan": result.plan or service.latest_plan(work_id) or "",
                "review_passed": result.review_passed,
                "review_suggestions": result.review_suggestions,
                "ok": report.ok and result.completed,
                "violations": report.to_dict()["violations"],
            }
        finally:
            service.finish_generation(work_id, finished)

    _jobs(app).start(job, target)
    return job


def _start_architecture_job(app: Flask, service: Any, work_id: str, prompt: str) -> Job:
    """Create and start a per-movement architect + composer job.

    Args:
        app: Flask application.
        service: Composition service.
        work_id: Active work.
        prompt: Composition goal.

    Returns:
        The started job.
    """
    job = _jobs(app).create()
    service.start_generation(work_id, prompt)

    def target(emit: EventCallback) -> dict[str, Any]:
        finished = False
        try:
            plan = Architect(service, create_chat_model()).plan(work_id, prompt, on_event=emit)
            window = context_window_tokens()
            composer = MovementComposer(
                service,
                create_chat_model(),
                reviewer=ReviewerAI(create_chat_model(), context_window=window),
                context_window=window,
            )
            outcome = composer.compose(work_id, prompt, on_event=emit)
            service.ensure_title(work_id)
            movements = service.get_work(work_id).movements
            movement_id = movements[0].id if movements else "m01"
            if outcome.review_passed is not None:
                service.record_review(
                    work_id, movement_id, outcome.review_passed, outcome.review_suggestions
                )
            reports = [service.check(work_id, movement.id) for movement in movements]
            violations = [item for report in reports for item in report.to_dict()["violations"]]
            ok = bool(reports) and all(report.ok for report in reports) and outcome.completed
            finished = True
            return {
                "work_id": work_id,
                "movement_id": movement_id,
                "completed": outcome.completed,
                "steps": len(movements),
                "final_text": plan.plan,
                "plan": plan.plan,
                "plan_tree": plan.tree,
                "review_passed": outcome.review_passed,
                "review_suggestions": outcome.review_suggestions,
                "ok": ok,
                "violations": violations,
            }
        finally:
            service.finish_generation(work_id, finished)

    _jobs(app).start(job, target)
    return job


def _stream(job: Job) -> Iterator[str]:
    """Yield server-sent events for a job until it finishes.

    Args:
        job: The job to stream.

    Yields:
        SSE-formatted chunks.
    """
    index = 0
    while True:
        with job.condition:
            if index >= len(job.events) and job.status == "running":
                job.condition.wait(timeout=_SSE_TIMEOUT)
            pending = job.events[index:]
            index = len(job.events)
            status = job.status
        if not pending and status == "running":
            yield ": keepalive\n\n"
            continue
        for event in pending:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        if status != "running":
            break
    payload = {**job.result, "status": job.status}
    yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _register_api(app: Flask) -> None:
    """Register the JSON and score API routes.

    Args:
        app: Flask application.
    """

    @app.get("/api/works/<work_id>/movements/<movement_id>/score")
    def score(work_id: str, movement_id: str) -> Response:
        """Return the current score as MusicXML.

        Works with composed movements expose the live merged movements; an empty
        work falls back to the movement's latest revision.  A transient failure
        returns an empty MusicXML document so the browser renderer never receives
        an HTML error page.
        """
        service = get_service(app)
        mimetype = "application/vnd.recordare.musicxml+xml"
        try:
            merged = service.merged_musicxml(work_id)
            if merged:
                return Response(merged, mimetype=mimetype)
            fallback = service.current_musicxml(work_id, movement_id)
            return Response(fallback, mimetype=mimetype)
        except Exception:
            app.logger.exception("score rendering failed for %s/%s", work_id, movement_id)
            return Response("", mimetype=mimetype)

    @app.post("/api/works/<work_id>/movements/<movement_id>/check")
    def check(work_id: str, movement_id: str) -> Response:
        """Return the symbolic check report."""
        report = get_service(app).check(work_id, movement_id)
        return jsonify(report.to_dict())

    @app.post("/api/works/<work_id>/movements/<movement_id>/finalize")
    def finalize(work_id: str, movement_id: str) -> Response:
        """Finalise a movement."""
        result = get_service(app).finalize(work_id, movement_id)
        return jsonify(_result_payload(result))

    @app.post("/api/works/<work_id>/movements/<movement_id>/rollback")
    def rollback(work_id: str, movement_id: str) -> Response:
        """Roll a movement back to an earlier revision."""
        raw = request.json.get("seq", 0) if request.is_json else request.form.get("seq", 0)
        result = get_service(app).rollback(work_id, movement_id, int(raw))
        return jsonify(_result_payload(result))

    @app.post("/api/works/<work_id>/movements/<movement_id>/audit")
    def audit(work_id: str, movement_id: str) -> Response:
        """Record a human audition note."""
        note = request.form.get("note") or (request.json or {}).get("note", "")
        get_service(app).record_audit(work_id, movement_id, str(note))
        return jsonify({"ok": True})

    @app.post("/api/works/<work_id>/movements/<movement_id>/agent")
    def run_agent(work_id: str, movement_id: str) -> Response:
        """Run one composer-agent session."""
        payload: dict[str, Any] = request.get_json(silent=True) or {}
        goal = str(payload.get("goal", "创作一首古典音乐作品。"))
        feedback = payload.get("feedback")
        service = get_service(app)
        window = context_window_tokens()
        loop = AgentLoop(
            service,
            create_chat_model(),
            reviewer=ReviewerAI(create_chat_model(), context_window=window),
            context_window=window,
        )
        outcome = loop.run(work_id, movement_id, goal, feedback=feedback)
        service.ensure_title(work_id)
        return jsonify(
            {
                "completed": outcome.completed,
                "steps": outcome.steps,
                "final_text": outcome.final_text,
                "review_passed": outcome.review_passed,
            }
        )


def _downloads_dir() -> Path:
    """Return the directory exports are written to.

    Returns:
        ``~/Downloads`` unless ``HARMONIA_DOWNLOADS`` overrides it.
    """
    return Path(os.environ.get("HARMONIA_DOWNLOADS") or (Path.home() / "Downloads"))


def _register_export(app: Flask) -> None:
    """Register the export routes.

    Args:
        app: Flask application.
    """

    @app.get("/works/<work_id>/movements/<movement_id>/export/<fmt>")
    def export(work_id: str, movement_id: str, fmt: str) -> Any:
        """Export a movement artifact into the downloads folder."""
        exporter = get_export_service(app)
        result = exporter.export(work_id, movement_id, _downloads_dir(), fmt)
        if not result.ok or result.path is None:
            return jsonify({"ok": False, "error": result.error}), 400
        if request.args.get("save"):
            return jsonify({"ok": True, "path": str(result.path)})
        return send_file(result.path, as_attachment=True)

    @app.get("/works/<work_id>/export/<fmt>")
    def export_all(work_id: str, fmt: str) -> Any:
        """Export every movement, zipping when there are several."""
        exporter = get_export_service(app)
        result = exporter.export_work(work_id, fmt, _downloads_dir())
        if not result.ok or result.path is None:
            return jsonify({"ok": False, "error": result.error}), 400
        if request.args.get("save"):
            return jsonify({"ok": True, "path": str(result.path)})
        return send_file(result.path, as_attachment=True)


def _result_payload(result: Any) -> dict[str, Any]:
    """Serialise a tool result for JSON responses.

    Args:
        result: A service tool result.

    Returns:
        A JSON-friendly dictionary.
    """
    return {
        "ok": result.ok,
        "message": result.message,
        "error_code": result.error_code,
        "theme_id": result.theme_id,
        "violations": result.report.to_dict()["violations"] if result.report else [],
    }
