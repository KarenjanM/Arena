"""Stops the `arena launch` children's process groups when the supervisor dies without cleaning up."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from typing import IO

GRACE_INT_S = float(os.environ.get('ARENA_SHUTDOWN_TIMEOUT', '30'))
GRACE_TERM_S = 3.0
GRACE_KILL_S = 5.0


class Reaper:
    """Reaper process fed one process group id per line, and `done` on a clean exit."""

    def __init__(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, '-m', 'arena_bringup.reaper'],
            stdin=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert self._proc.stdin is not None
        self._stdin: IO[str] = self._proc.stdin

    def watch(self, pgid: int) -> None:
        self._send(f'{pgid}\n')

    def release(self) -> None:
        self._send('done\n')
        with contextlib.suppress(BrokenPipeError, OSError):
            self._stdin.close()
        self._proc.wait()

    def _send(self, line: str) -> None:
        with contextlib.suppress(BrokenPipeError, OSError):
            self._stdin.write(line)
            self._stdin.flush()


def _exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal(pgids: Iterable[int], sig: int) -> None:
    for pgid in pgids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)


def _wait(pgids: list[int], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while any(_exists(pgid) for pgid in pgids):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)
    return True


def stop(pgids: list[int]) -> None:
    for sig, grace in ((signal.SIGINT, GRACE_INT_S), (signal.SIGTERM, GRACE_TERM_S), (signal.SIGKILL, GRACE_KILL_S)):
        live = [pgid for pgid in pgids if _exists(pgid)]
        if not live:
            return
        _signal(live, sig)
        if _wait(live, grace):
            return


def main() -> int:
    pgids: list[int] = []
    for line in sys.stdin:
        if line.strip() == 'done':
            return 0
        pgids.append(int(line))
    stop(pgids)
    return 0


if __name__ == '__main__':
    sys.exit(main())
