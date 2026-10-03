# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for background agent jobs."""

from __future__ import annotations

from typing import Any

from app.jobs import _MAX_JOBS, EventCallback, Job, JobManager

_TIMEOUT = 5.0


def wait(job: Job) -> None:
    """Block until a job finishes.

    Args:
        job: The job.
    """
    job.condition.wait_for(lambda: job.status != "running", timeout=_TIMEOUT)


class TestJob:
    """A single job's event buffer."""

    def test_emit_and_finish(self) -> None:
        """Events accumulate and finish stores the result."""
        job = Job("j1")
        assert job.status == "running"
        job.emit({"kind": "a"})
        job.finish("done", {"ok": True})
        assert job.events == [{"kind": "a"}]
        assert job.status == "done"
        assert job.result == {"ok": True}

    def test_finish_without_result(self) -> None:
        """Finishing without a result stores an empty mapping."""
        job = Job("j2")
        job.finish("error")
        assert job.result == {}


class TestJobManager:
    """Job creation and execution."""

    def test_create_and_get(self) -> None:
        """Created jobs are retrievable by id."""
        manager = JobManager()
        job = manager.create()
        assert manager.get(job.id) is job
        assert manager.get("missing") is None

    def test_run_success(self) -> None:
        """A successful target stores its result and emits events."""
        manager = JobManager()
        job = manager.create()

        def target(emit: EventCallback) -> dict[str, Any]:
            emit({"kind": "a"})
            return {"ok": True}

        manager.start(job, target)
        wait(job)
        assert job.status == "done"
        assert job.result == {"ok": True}
        assert job.events == [{"kind": "a"}]

    def test_run_error(self) -> None:
        """A raising target records an error event and status.

        Raises:
            RuntimeError: When the operation cannot proceed.
        """
        manager = JobManager()
        job = manager.create()

        def target(_emit: EventCallback) -> dict[str, Any]:
            raise RuntimeError("boom")

        manager.start(job, target)
        wait(job)
        assert job.status == "error"
        assert job.result == {"message": "boom"}
        assert any(event["kind"] == "error" for event in job.events)

    def test_run_base_exception(self) -> None:
        """A BaseException still leaves the job in a terminal state.

        Raises:
            KeyboardInterrupt: When the operation cannot proceed.
        """
        manager = JobManager()
        job = manager.create()

        def target(_emit: EventCallback) -> dict[str, Any]:
            raise KeyboardInterrupt

        manager.start(job, target)
        wait(job)
        assert job.status == "error"

    def test_prune_finished_jobs(self) -> None:
        """Old finished jobs are dropped once the registry grows too large."""
        manager = JobManager()
        jobs = []
        for _ in range(_MAX_JOBS + 5):
            job = manager.create()
            job.finish("done", {})
            jobs.append(job)
        assert manager.get(jobs[0].id) is None
        assert manager.get(jobs[-1].id) is jobs[-1]

    def test_prune_keeps_running_jobs(self) -> None:
        """Running jobs are never pruned."""
        manager = JobManager()
        running = [manager.create() for _ in range(_MAX_JOBS + 1)]
        assert manager.get(running[0].id) is running[0]
