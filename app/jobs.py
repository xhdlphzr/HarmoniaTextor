# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Background agent jobs with server-sent-event progress.

A job runs one fully automatic composer session on a daemon thread and streams
structured progress events to the browser.  Server-sent events need no client
library, so the UI stays fully offline.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import Any

__all__ = ["Job", "JobManager"]

EventCallback = Callable[[dict[str, Any]], None]
JobTarget = Callable[[EventCallback], dict[str, Any]]

#: Finished jobs kept before the oldest are pruned, bounding memory in a
#: long-lived desktop session.
_MAX_JOBS = 64


class Job:
    """A single background agent run.

    Attributes:
        id: Job identifier.
        status: One of ``running``, ``done`` or ``error``.
        events: Progress events in emission order.
        result: Final result payload once the job finishes.
    """

    def __init__(self, job_id: str) -> None:
        """Initialise a running job.

        Args:
            job_id: Job identifier.
        """
        self.id = job_id
        self.status = "running"
        self.events: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}
        self.condition = threading.Condition()

    def emit(self, event: dict[str, Any]) -> None:
        """Append a progress event and wake listeners.

        Args:
            event: Event payload.
        """
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    def finish(self, status: str, result: dict[str, Any] | None = None) -> None:
        """Mark the job finished.

        Args:
            status: Terminal status.
            result: Final result payload.
        """
        with self.condition:
            self.status = status
            self.result = result or {}
            self.condition.notify_all()


class JobManager:
    """Registry and thread launcher for background agent jobs."""

    def __init__(self) -> None:
        """Initialise an empty manager."""
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        """Create and register a new job.

        Returns:
            The new job.
        """
        job = Job(uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        return job

    def _prune(self) -> None:
        """Drop the oldest finished jobs once the registry grows too large."""
        while len(self._jobs) > _MAX_JOBS:
            for job_id, job in list(self._jobs.items()):
                if job.status != "running":
                    del self._jobs[job_id]
                    break
            else:
                return

    def get(self, job_id: str) -> Job | None:
        """Look up a job by identifier.

        Args:
            job_id: Job identifier.

        Returns:
            The job, or ``None`` when unknown.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job: Job, target: JobTarget) -> None:
        """Run a job target on a daemon thread.

        Args:
            job: Job to run.
            target: Callable receiving an emit function and returning a result.
        """
        thread = threading.Thread(target=self._run, args=(job, target), daemon=True)
        thread.start()

    @staticmethod
    def _run(job: Job, target: JobTarget) -> None:
        """Execute a job target, recording success or failure.

        Args:
            job: Job to run.
            target: Callable receiving an emit function.
        """
        try:
            result = target(job.emit)
        except Exception as exc:  # noqa: BLE001 - surface any job failure to the client
            job.emit({"kind": "error", "message": str(exc)})
            job.finish("error", {"message": str(exc)})
            return
        except BaseException:  # noqa: BLE001 - KeyboardInterrupt/SystemExit must
            # never leave a job stuck in the running state.
            job.finish("error", {"message": "job aborted"})
        else:
            job.finish("done", result)
