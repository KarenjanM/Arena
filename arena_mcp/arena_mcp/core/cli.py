from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import ToolError
from .jobs import JOBS, Job


def arena_dir() -> Path:
    value = os.environ.get("ARENA_DIR")
    if not value:
        raise ToolError("ARENA_DIR is not set", hint="serve from a sourced Arena container")
    return Path(value)


def argv(*args: str) -> list[str]:
    """The arena CLI invocation for args, the way the shell function runs it."""
    return [sys.executable, "-m", "arena_cli", *args]


def _env() -> dict[str, str]:
    meta = str(arena_dir() / "_meta")
    path = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONPATH": f"{meta}:{path}" if path else meta}


def run(*args: str, timeout: float = 600.0, check: bool = True, cwd: str | Path | None = None) -> dict[str, Any]:
    """Run an arena CLI verb to completion and return its exit code and output."""
    try:
        done = subprocess.run(argv(*args), capture_output=True, text=True, timeout=timeout, env=_env(), cwd=cwd or arena_dir().parent.parent, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"arena {' '.join(args)} did not finish within {timeout:.0f}s", hint="long verbs run as jobs") from exc
    out = {"command": f"arena {' '.join(args)}", "exit_code": done.returncode, "stdout": done.stdout[-20000:], "stderr": done.stderr[-20000:]}
    if check and done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-15:]
        raise ToolError(f"arena {' '.join(args)} failed with exit code {done.returncode}:\n" + "\n".join(tail))
    return out


async def job(kind: str, title: str, *args: str, result: Callable[[Job], dict[str, Any]] | None = None, cwd: str | Path | None = None) -> Job:
    """Start an arena CLI verb as a detached job."""
    return await JOBS.start_process(kind, title, argv(*args), cwd=cwd or arena_dir().parent.parent, env=_env(), result=result)
