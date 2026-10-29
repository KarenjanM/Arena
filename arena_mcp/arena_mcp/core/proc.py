from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from arena_bringup.reaper import GRACE_INT_S, GRACE_KILL_S, GRACE_TERM_S

from .errors import ToolError


def _stat(pid: int) -> list[str] | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return None


def parent_of(pid: int) -> int | None:
    fields = _stat(pid)
    return int(fields[1]) if fields is not None else None


def start_time(pid: int) -> int | None:
    """The process start time in clock ticks after boot, field 22 of /proc/<pid>/stat."""
    fields = _stat(pid)
    return int(fields[19]) if fields is not None else None


def ancestry(pid: int | None = None) -> set[int]:
    """The process and every ancestor up to init."""
    current: int | None = os.getpid() if pid is None else pid
    chain: set[int] = set()
    while current is not None and current > 1 and current not in chain:
        chain.add(current)
        current = parent_of(current)
    return chain


def alive(pid: int, started: int | None = None) -> bool:
    """Whether pid runs and is not a zombie, and when started is given, whether it is still the process that started then."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    fields = _stat(pid)
    if fields is None or fields[0] == "Z":
        return False
    return started is None or int(fields[19]) == started


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def group_members(pgid: int) -> list[int]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fields = _stat(int(entry.name))
        if fields is not None and int(fields[2]) == pgid and fields[0] != "Z":
            members.append(int(entry.name))
    return members


class ProtectedProcess(ToolError):
    pass


def check_group(pgid: int) -> None:
    """Refuse a process group that holds this server or one of its ancestors."""
    if pgid in ancestry() or os.getpgid(os.getpid()) == pgid:
        raise ProtectedProcess(f"process group {pgid} contains this server, not signalling it")


def signal_group(pid: int, sig: signal.Signals) -> bool:
    """Send sig to the process group of pid, False when it is gone."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    check_group(pgid)
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


async def stop_group(pgid: int, *, grace: float = GRACE_INT_S) -> str:
    """SIGINT a process group, then SIGTERM and SIGKILL after the arena_bringup reaper's grace windows, each only to what survives."""
    check_group(pgid)
    loop = asyncio.get_running_loop()
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, GRACE_TERM_S), (signal.SIGKILL, GRACE_KILL_S)):
        if not group_members(pgid):
            return "not running"
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return "not running"
        deadline = loop.time() + wait
        while group_members(pgid) and loop.time() < deadline:
            await asyncio.sleep(0.2)
        if not group_members(pgid):
            return f"stopped by {sig.name}"
    return f"survivors {group_members(pgid)}"
