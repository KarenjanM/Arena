from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import ToolError
from .jobs import JOBS, Job

SHIM = "_arena_cli"
UNSOURCED = "serve from a sourced Arena container"


def arena_dir() -> Path:
    """The Arena checkout: ARENA_DIR, else the checkout this source file lives in."""
    if value := os.environ.get("ARENA_DIR"):
        return Path(value)
    for parent in Path(__file__).resolve().parents:
        if (parent / "_meta" / "arena_cli").is_dir():
            return parent
    raise ToolError("ARENA_DIR is not set and arena_mcp runs outside an Arena checkout", hint=UNSOURCED)


def ws_dir() -> Path:
    """$ARENA_WS_DIR, the directory the arena CLI runs its verbs in."""
    if value := os.environ.get("ARENA_WS_DIR"):
        return Path(value)
    raise ToolError("ARENA_WS_DIR is not set", hint=UNSOURCED)


def sourced() -> bool:
    """Whether this process inherited the _arena_cli function a sourced shell exports."""
    return f"BASH_FUNC_{SHIM}%%" in os.environ


def argv(*args: str) -> list[str]:
    """The arena CLI invocation for args through the _arena_cli function a sourced shell exports."""
    if not sourced():
        raise ToolError(f"the {SHIM} shell function is not exported here", hint=UNSOURCED)
    return ["bash", "-c", f'{SHIM} "$@"', "arena", *args]


def run(*args: str, timeout: float = 600.0, check: bool = True, cwd: str | Path | None = None, clip: int | None = 20000) -> dict[str, Any]:
    """Run an arena CLI verb to completion and return its exit code and output, each stream cut to its last clip characters."""
    try:
        done = subprocess.run(argv(*args), capture_output=True, text=True, timeout=timeout, cwd=cwd or ws_dir(), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"arena {' '.join(args)} did not finish within {timeout:.0f}s", hint="long verbs run as jobs") from exc
    out = {"command": f"arena {' '.join(args)}", "exit_code": done.returncode, "stdout": done.stdout[-clip:] if clip else done.stdout, "stderr": done.stderr[-clip:] if clip else done.stderr}
    if check and done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-15:]
        raise ToolError(f"arena {' '.join(args)} failed with exit code {done.returncode}:\n" + "\n".join(tail))
    return out


async def job(kind: str, title: str, *args: str, result: Callable[[Job], dict[str, Any]] | None = None, cwd: str | Path | None = None) -> Job:
    """Start an arena CLI verb as a detached job."""
    return await JOBS.start_process(kind, title, argv(*args), cwd=cwd or ws_dir(), result=result)
