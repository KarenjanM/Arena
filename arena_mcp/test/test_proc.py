from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import pytest

from arena_mcp.core.proc import ProtectedProcess, alive, ancestry, group_members, stop_group


def test_stop_group_escalates_only_for_survivors():
    stubborn = subprocess.Popen([sys.executable, "-c", "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nprint('ready', flush=True)\ntime.sleep(600)"], start_new_session=True, stdout=subprocess.PIPE, text=True)
    assert stubborn.stdout.readline().strip() == "ready"
    outcome = asyncio.run(stop_group(stubborn.pid, grace=0.5, term_grace=2.0))
    assert outcome == "stopped by SIGTERM"
    stubborn.wait(timeout=5)
    assert not group_members(stubborn.pid)


def test_stop_group_refuses_the_servers_own_group():
    with pytest.raises(ProtectedProcess):
        asyncio.run(stop_group(os.getpgid(os.getpid())))


def test_ancestry_and_liveness():
    assert os.getpid() in ancestry()
    assert os.getppid() in ancestry()
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    time.sleep(0.1)
    assert not alive(child.pid)
    assert alive(os.getpid())
