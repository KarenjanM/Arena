from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path


def parent_of(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[1])


def ancestry(pid: int | None = None) -> set[int]:
    """The process and every ancestor up to init."""
    current: int | None = os.getpid() if pid is None else pid
    chain: set[int] = set()
    while current is not None and current > 1 and current not in chain:
        chain.add(current)
        current = parent_of(current)
    return chain


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


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
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            members.append(int(entry.name))
    return members


class ProtectedProcess(RuntimeError):
    pass


async def stop_group(pgid: int, *, grace: float = 15.0, term_grace: float = 5.0) -> str:
    """SIGINT a process group, then SIGTERM and SIGKILL only to what survives each grace window."""
    protected = ancestry()
    if pgid in protected or os.getpgid(os.getpid()) == pgid:
        raise ProtectedProcess(f"process group {pgid} contains this server")
    loop = asyncio.get_running_loop()
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, term_grace), (signal.SIGKILL, 2.0)):
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
