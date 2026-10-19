from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Trigger

from arena_mcp.core import JOBS, ROS, proc
from arena_mcp.core.errors import ToolError
from arena_mcp.domains import system

GIB = 2**20


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path):
    previous = os.environ.get("ARENA_DATA_DIR")
    os.environ["ARENA_DATA_DIR"] = str(tmp_path / "data")
    yield tmp_path
    if previous is None:
        del os.environ["ARENA_DATA_DIR"]
    else:
        os.environ["ARENA_DATA_DIR"] = previous


def write_meminfo(path, available_gib, swap_total_gib=8.0, swap_free_gib=8.0):
    path.write_text(f"MemTotal:       {64 * GIB} kB\nMemFree:        {GIB} kB\nMemAvailable:   {int(available_gib * GIB)} kB\nSwapTotal:      {int(swap_total_gib * GIB)} kB\nSwapFree:       {int(swap_free_gib * GIB)} kB\nHugePages_Total:       0\n")
    return path


def test_meminfo_reads_the_live_proc_file():
    info = system.meminfo()
    assert info["MemAvailable"] > 0
    assert "SwapTotal" in info


def test_memory_guard_floors_per_sim_and_swap(tmp_path):
    roomy = system.meminfo(write_meminfo(tmp_path / "roomy", 16.0))
    assert system.memory_guard("isaac", roomy)["required_gib"] == 15.0
    assert system.memory_guard("gazebo", roomy)["available_gib"] == 16.0
    middle = system.meminfo(write_meminfo(tmp_path / "middle", 8.0))
    assert system.memory_guard("dummy", middle)["required_gib"] == 6.0
    with pytest.raises(ToolError, match="needs 15 GiB"):
        system.memory_guard("isaac", middle)
    with pytest.raises(ToolError, match="needs 6 GiB"):
        system.memory_guard("flatsim", system.meminfo(write_meminfo(tmp_path / "short", 5.5)))
    with pytest.raises(ToolError, match="swap in use"):
        system.memory_guard("dummy", system.meminfo(write_meminfo(tmp_path / "swapping", 10.0, swap_free_gib=6.5)))
    assert system.memory_guard("dummy", system.meminfo(write_meminfo(tmp_path / "shared_box", 80.0, swap_total_gib=16.0, swap_free_gib=5.0)))["swap_used_gib"] == 11.0


def test_launch_tokens_never_start_rviz_and_refuse_owned_keys():
    tokens = system.launch_tokens("flatsim", env_n=2, world="map_empty", robot="jackal", args={"task.robots": "random", "record.auto": False, "lockstep.rtf": 2.5}, headless=True)
    assert tokens == ["sim:=flatsim", "env.n:=2", "viz:=false", "headless:=true", "world:=map_empty", "robot:=jackal", "task.robots:=random", "record.auto:=false", "lockstep.rtf:=2.5"]
    assert system.launch_tokens("gazebo", env_n=None, world=None, robot=None, args=None, headless=False) == ["sim:=gazebo", "headless:=false"]
    for key in ("viz", "env.n", "sim"):
        with pytest.raises(ToolError, match="its own parameter"):
            system.launch_tokens("dummy", env_n=1, world=None, robot=None, args={key: "x"}, headless=True)
    with pytest.raises(ToolError, match="not a launch argument"):
        system.launch_tokens("dummy", env_n=1, world=None, robot=None, args={"a b": 1}, headless=True)
    with pytest.raises(ToolError, match="at least 1"):
        system.launch_tokens("dummy", env_n=0, world=None, robot=None, args=None, headless=True)


def _registry_peer():
    node = rclpy.create_node("arena_mcp_system_peer", namespace="/arena")
    node.create_service(Trigger, "/arena/register_env", lambda request, response: response)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    return node, executor


def test_launch_is_refused_while_a_runtime_is_up_and_unknown_sims_are_listed():
    async def scenario():
        await ROS.node()
        try:
            assert not await system.runtime_up(discovery_s=0.5)
            with pytest.raises(ToolError) as unknown:
                await system.system_launch("no_such_sim")
            node, executor = _registry_peer()
            try:
                assert await system.runtime_up(discovery_s=10.0)
                with pytest.raises(ToolError, match="already up") as refused:
                    await system.system_launch("dummy")
                with pytest.raises(ToolError, match="already up"):
                    await system.system_runtime("dummy")
            finally:
                executor.shutdown()
                node.destroy_node()
        finally:
            await ROS.shutdown()
        return unknown.value, refused.value

    unknown, refused = asyncio.run(scenario())
    assert "dummy" in unknown.hint and "flatsim" in unknown.hint
    assert "attach=True" in refused.hint


GRACEFUL = """
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time\\ntry:\\n    time.sleep(600)\\nexcept KeyboardInterrupt:\\n    pass"], start_new_session=True)
print(child.pid, flush=True)
def forward(signum, frame):
    import os
    os.killpg(child.pid, signal.SIGINT)
    child.wait()
    sys.exit(0)
signal.signal(signal.SIGINT, forward)
time.sleep(600)
"""

ORPHANING = """
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import signal, time\\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\\ntime.sleep(600)"], start_new_session=True)
print(child.pid, flush=True)
time.sleep(600)
"""


async def _started(kind, script):
    job = await JOBS.start_process(kind, f"{kind} stand-in", [sys.executable, "-c", script])
    for _ in range(100):
        if JOBS.tail(job.id).strip():
            break
        await asyncio.sleep(0.1)
    return job, int(JOBS.tail(job.id).split()[0])


def test_stop_sigints_the_launch_group_and_waits_for_the_whole_tree():
    async def scenario():
        job, child = await _started("launch", GRACEFUL)
        stacks = system.system_stacks()
        stopped = await system.system_stop(grace_s=10)
        return job, child, stacks, stopped

    job, child, stacks, stopped = asyncio.run(scenario())
    assert [entry["id"] for entry in stacks["jobs"]] == [job.id]
    assert stacks["jobs"][0]["processes"] == 2
    assert not os.path.exists(f"/proc/{child}") or open(f"/proc/{child}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
    assert stopped == {"job": job.id, "signal": "SIGINT", "stopped": True, "survivors": []}


def test_stop_reports_orphans_and_escalates_on_the_next_call():
    async def scenario():
        job, child = await _started("launch", ORPHANING)
        first = await system.system_stop(job.id, grace_s=2)
        second = await system.system_stop(job.id, grace_s=10)
        return child, first, second

    child, first, second = asyncio.run(scenario())
    assert first["signal"] == "SIGINT" and not first["stopped"]
    assert [entry["pid"] for entry in first["survivors"]] == [child]
    assert "SIG_IGN" in first["survivors"][0]["cmdline"]
    assert second["signal"] == "SIGTERM" and second["stopped"]


def test_stop_without_a_launch_job_and_on_a_non_launch_job():
    async def scenario():
        with pytest.raises(ToolError, match="no launch job"):
            await system.system_stop()
        job = await JOBS.start_process("echo", "not a launch", [sys.executable, "-c", "print(1)"])
        await JOBS.wait(job.id, 20)
        with pytest.raises(ToolError, match="not a launch"):
            await system.system_stop(job.id)
        with pytest.raises(ToolError, match="grace_s"):
            await system.system_stop(job.id, grace_s=0)

    asyncio.run(scenario())


def test_stacks_list_foreign_launch_processes_outermost_only():
    script = "import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', 'ros2', 'run', 'inner', 'x'])\ntime.sleep(60)"
    outer = subprocess.Popen([sys.executable, "-c", script, "ros2", "run", "outer", "x"])
    try:
        for _ in range(100):
            inner = [pid for pid in system.tree(outer.pid) if pid != outer.pid and "ros2 run inner" in proc.cmdline(pid)]
            if inner:
                break
            time.sleep(0.1)
        assert inner
        pids = {entry["pid"]: entry["cmdline"] for entry in system.system_stacks()["other"]}
        assert outer.pid in pids
        assert not set(inner) & set(pids)
    finally:
        for pid in reversed(system.tree(outer.pid)):
            os.kill(pid, signal.SIGTERM)
        outer.wait()


def test_colcon_jobs_are_serialized():
    async def scenario():
        job = await JOBS.start_process("build", "held build", [sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with pytest.raises(ToolError, match="still running"):
                await system.system_build(["arena_mcp"])
            with pytest.raises(ToolError, match="still running"):
                await system.system_test("arena_mcp")
        finally:
            await JOBS.cancel(job.id)

    asyncio.run(scenario())


def test_node_logs_by_file_name_then_by_logger(tmp_path):
    root = tmp_path / "log"
    run = root / "2026-10-19-17-00-00-000000-host-1"
    run.mkdir(parents=True)
    named = root / "map_server_42_1792401415689.log"
    named.write_text("[INFO] [1.0] [map_server]: loaded\n")
    other = root / "map_server_manager_43_1792401415689.log"
    other.write_text("noise\n")
    python = root / "python3_44_1792401415690.log"
    python.write_text("[INFO] [1.0] [arena.env_0.task_generator_node]: reset done\n")
    (run / "launch.log").write_text("[INFO] [launch]: All log files can be found below\n")
    os.utime(other, (time.time() + 5, time.time() + 5))
    assert system.node_log(root, "/map_server") == named
    assert system.node_log(root, "/arena/env_0/task_generator_node") == python
    assert system.node_log(root, "task_generator_node") == python
    assert system.node_log(root, "launch") == run / "launch.log"
    with pytest.raises(ToolError, match="no log of"):
        system.node_log(root, "/arena/env_1/task_generator_node")
    with pytest.raises(ToolError, match="no ROS log directory"):
        system.node_log(tmp_path / "missing", "x")
    assert system.tail(python, 1) == "[INFO] [1.0] [arena.env_0.task_generator_node]: reset done"


def test_logs_of_a_job_and_argument_checks(tmp_path):
    async def scenario():
        job = await JOBS.start_process("echo", "two lines", [sys.executable, "-c", "print('first'); print('second')"])
        await JOBS.wait(job.id, 20)
        return job

    job = asyncio.run(scenario())
    logs = system.system_logs(job_id=job.id, lines=1)
    assert logs["log"] == "second" and logs["state"] == "succeeded"
    with pytest.raises(ToolError, match="not both"):
        system.system_logs(job_id=job.id, node="x")
    with pytest.raises(ToolError, match="lines"):
        system.system_logs(job_id=job.id, lines=0)
    previous = os.environ.get("ROS_LOG_DIR")
    os.environ["ROS_LOG_DIR"] = str(tmp_path / "ros_log")
    try:
        (tmp_path / "ros_log").mkdir()
        (tmp_path / "ros_log" / "talker_1_2.log").write_text("hello\n")
        assert system.system_logs(node="/talker")["log"] == "hello"
    finally:
        if previous is None:
            del os.environ["ROS_LOG_DIR"]
        else:
            os.environ["ROS_LOG_DIR"] = previous
