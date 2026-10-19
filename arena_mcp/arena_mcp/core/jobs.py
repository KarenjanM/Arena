from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import secrets
import socket
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from .errors import ToolError
from .proc import alive, stop_group

RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
LOST = "lost"
TERMINAL = (SUCCEEDED, FAILED, CANCELLED, LOST)


def jobs_root() -> Path:
    """$ARENA_DATA_DIR/mcp/<hostname>/jobs, so forks sharing one data dir keep separate job tables."""
    base = Path(os.environ.get("ARENA_DATA_DIR") or Path.home() / "arena_ws" / "data")
    return base / "mcp" / socket.gethostname() / "jobs"


@dataclasses.dataclass
class Job:
    id: str
    kind: str
    title: str
    started: float
    state: str = RUNNING
    ended: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] = dataclasses.field(default_factory=dict)
    argv: list[str] | None = None
    cwd: str | None = None
    pid: int | None = None
    exit_code: int | None = None

    @property
    def log_path(self) -> Path:
        return jobs_root() / f"{self.id}.log"

    @property
    def record_path(self) -> Path:
        return jobs_root() / f"{self.id}.json"

    def as_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["log_path"] = str(self.log_path)
        out["elapsed_s"] = round((self.ended or time.time()) - self.started, 1)
        return {key: value for key, value in out.items() if value not in (None, {}, [])}

    def save(self) -> None:
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.record_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self)))
        tmp.replace(self.record_path)


class JobHandle:
    """What a task job's body uses to report while it runs."""

    def __init__(self, job: Job) -> None:
        self.job = job

    def log(self, line: str) -> None:
        with self.job.log_path.open("a") as log:
            log.write(line.rstrip("\n") + "\n")

    def progress(self, **fields: Any) -> None:
        self.job.progress.update(fields)
        self.job.save()


class Jobs:
    """Every long operation the server starts: in-process tasks and detached subprocesses."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancels: dict[str, Callable[[], Awaitable[None]]] = {}
        self._cancelling: set[str] = set()
        self._loaded = False

    def _new(self, kind: str, title: str) -> Job:
        jobs_root().mkdir(parents=True, exist_ok=True)
        job = Job(id=f"{kind}-{time.strftime('%m%d%H%M%S')}-{secrets.token_hex(2)}", kind=kind, title=title, started=time.time())
        self._jobs[job.id] = job
        job.save()
        return job

    def _finish(self, job: Job, state: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        if job.state in TERMINAL:
            return
        job.state = state
        job.ended = time.time()
        job.result = result
        job.error = error
        job.save()

    def start_task(
        self,
        kind: str,
        title: str,
        body: Callable[[JobHandle], Awaitable[dict[str, Any]]],
        *,
        on_cancel: Callable[[], Awaitable[None]] | None = None,
    ) -> Job:
        """Run body in the server's event loop as a job, its return value becomes the job result."""
        job = self._new(kind, title)
        handle = JobHandle(job)

        async def run() -> None:
            try:
                result = await body(handle)
            except asyncio.CancelledError:
                self._finish(job, CANCELLED)
                raise
            except ToolError as exc:
                self._finish(job, FAILED, error=exc.message if exc.hint is None else f"{exc.message} ({exc.hint})")
            except Exception as exc:
                handle.log(f"{type(exc).__name__}: {exc}")
                self._finish(job, FAILED, error=f"{type(exc).__name__}: {exc}")
            else:
                self._finish(job, SUCCEEDED, result=result)

        self._tasks[job.id] = asyncio.get_running_loop().create_task(run(), name=job.id)
        if on_cancel is not None:
            self._cancels[job.id] = on_cancel
        return job

    async def start_process(
        self,
        kind: str,
        title: str,
        argv: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        result: Callable[[Job], dict[str, Any]] | None = None,
    ) -> Job:
        """Start argv in its own session with output in the job log; it outlives the server."""
        job = self._new(kind, title)
        job.argv = list(argv)
        job.cwd = str(cwd) if cwd is not None else os.getcwd()
        with job.log_path.open("ab") as log:
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=job.cwd,
                    env={**os.environ, **(env or {})},
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                self._finish(job, FAILED, error=f"could not start {argv[0]}: {exc}")
                return job
        job.pid = process.pid
        job.save()

        async def watch() -> None:
            code = await process.wait()
            job.exit_code = code
            if job.state in TERMINAL:
                job.save()
                return
            if job.id in self._cancelling:
                self._finish(job, CANCELLED, error=f"stopped, exit code {code}")
            elif code == 0:
                self._finish(job, SUCCEEDED, result=result(job) if result is not None else None)
            else:
                self._finish(job, FAILED, error=f"exit code {code}, see the log tail", result=result(job) if result is not None else None)

        self._tasks[job.id] = asyncio.get_running_loop().create_task(watch(), name=job.id)
        return job

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        root = jobs_root()
        if not root.is_dir():
            return
        for record in root.glob("*.json"):
            if record.stem in self._jobs:
                continue
            try:
                job = Job(**json.loads(record.read_text()))
            except (OSError, ValueError, TypeError):
                continue
            if job.state == RUNNING and (job.pid is None or not alive(job.pid)):
                job.state = LOST
                job.error = "the server that started it exited first, the outcome was not observed"
                job.save()
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job:
        self._load()
        job = self._jobs.get(job_id)
        if job is None:
            raise ToolError(f"no job {job_id!r}", hint="jobs_list shows the known jobs")
        if job.state == RUNNING and job.id not in self._tasks and job.pid is not None and not alive(job.pid):
            self._finish(job, LOST, error="exited while no server watched it")
        return job

    def list(self, *, kind: str | None = None, state: str | None = None, limit: int = 50) -> list[Job]:
        self._load()
        jobs = [job for job in self._jobs.values() if (kind is None or job.kind == kind) and (state is None or job.state == state)]
        return sorted(jobs, key=lambda job: job.started, reverse=True)[:limit]

    def tail(self, job_id: str, lines: int = 100) -> str:
        job = self.get(job_id)
        if not job.log_path.is_file():
            return ""
        with job.log_path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - 256 * 1024))
            text = log.read().decode(errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    async def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state in TERMINAL:
            return job
        hook = self._cancels.pop(job_id, None)
        if hook is not None:
            with contextlib.suppress(Exception):
                await hook()
        task = self._tasks.get(job_id)
        self._cancelling.add(job_id)
        if job.pid is not None:
            outcome = await stop_group(job.pid)
            if job.state in TERMINAL:
                job.error = outcome
                job.save()
            else:
                self._finish(job, CANCELLED, error=outcome)
        elif task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        return job

    async def wait(self, job_id: str, timeout: float) -> Job:
        job = self.get(job_id)
        task = self._tasks.get(job_id)
        if job.state in TERMINAL:
            return job
        if task is not None:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout)
            return job
        deadline = time.monotonic() + timeout
        while job.state == RUNNING and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            job = self.get(job_id)
        return job

    async def shutdown(self) -> None:
        """Cancel in-process task jobs; detached subprocesses keep running and stay listed."""
        for job_id, task in list(self._tasks.items()):
            job = self._jobs[job_id]
            if job.pid is None and not task.done():
                hook = self._cancels.pop(job_id, None)
                if hook is not None:
                    with contextlib.suppress(Exception):
                        await hook()
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task


JOBS = Jobs()
