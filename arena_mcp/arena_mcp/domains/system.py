from __future__ import annotations

import asyncio
import os
import re
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from arena_mcp.core import JOBS, Domain, Job, ToolError, cli, proc

domain = Domain(
    "system",
    """Stack lifecycle through the arena CLI: launch a simulation or a bare runtime, stop gracefully, list stacks, read logs, build, test, install deps, preload worlds.
    Launches, builds and tests are process jobs (jobs_status, jobs_log), they keep running when the server exits.
    A launch is always without rviz and headless by default, and like arena launch it adds envs to a runtime that is already up.""",
)

LAUNCH_KINDS = ("launch", "runtime")
COLCON_KINDS = ("build", "rebuild", "test")
RESERVED_ARGS = ("sim", "env.n", "world", "robot", "headless", "viz")
LAUNCH_RE = re.compile(r"arena_bringup[./]supervisor|(^|[/\s])ros2 (launch|run) ")
ARG_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")

ArgValue = str | int | float | bool


def launch_tokens(sim: str, *, env_n: int | None, world: str | None, robot: str | None, args: dict[str, ArgValue] | None, headless: bool) -> list[str]:
    """The KEY:=VALUE tokens of an arena launch or arena runtime call, refusing keys the parameters own."""
    tokens = [f"sim:={sim}"]
    if env_n is not None:
        if env_n < 1:
            raise ToolError("env_n must be at least 1")
        tokens.append(f"env.n:={env_n}")
        tokens.append("viz:=false")
    tokens.append(f"headless:={'true' if headless else 'false'}")
    if world:
        tokens.append(f"world:={world}")
    if robot:
        tokens.append(f"robot:={robot}")
    for key, value in (args or {}).items():
        if key in RESERVED_ARGS:
            raise ToolError(f"{key} is set by its own parameter, not through args", hint=f"args may not hold {', '.join(RESERVED_ARGS)}")
        if not ARG_KEY_RE.fullmatch(key):
            raise ToolError(f"{key!r} is not a launch argument name", hint="args maps launch argument names such as task.robots to values")
        tokens.append(f"{key}:={_arg_value(value)}")
    return tokens


def _arg_value(value: ArgValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _sims() -> list[str]:
    try:
        from arena_runtime import sim as runtime_sim
    except ImportError as exc:
        raise ToolError(f"arena_runtime is not importable ({exc})", hint="system_build arena_runtime") from exc
    return sorted(key.value for key in runtime_sim.SimulatorRegistry.keys())


async def _check_sim(sim: str) -> None:
    sims = await asyncio.to_thread(_sims)
    if sim not in sims:
        raise ToolError(f"unknown sim {sim!r}", hint=f"sims: {', '.join(sims)}")


def _running(kinds: tuple[str, ...]) -> list[Job]:
    return [job for kind in kinds for job in JOBS.list(kind=kind, state="running")]


def _tail_result(lines: int = 40) -> Callable[[Job], dict[str, Any]]:
    def result(job: Job) -> dict[str, Any]:
        return {"exit_code": job.exit_code, "tail": JOBS.tail(job.id, lines)}

    return result


@domain.tool(destructive=True)
async def system_launch(
    sim: Annotated[str, Field(description="simulator backend, one of the registered sims (dummy, flatsim, gazebo, isaac, mujoco, ...)")],
    env_n: Annotated[int, Field(description="envs to start, or to add to a runtime that is already up")] = 1,
    world: str | None = None,
    robot: str | None = None,
    args: Annotated[dict[str, ArgValue] | None, Field(description="further launch arguments as {name: value}, e.g. {'task.robots': 'random'}")] = None,
    headless: Annotated[bool, Field(description="hide the simulator GUI, rviz is never started")] = True,
) -> dict[str, Any]:
    """Start `arena launch` as a job, which adds env_n envs to a runtime that is already up."""
    await _check_sim(sim)
    tokens = launch_tokens(sim, env_n=env_n, world=world, robot=robot, args=args, headless=headless)
    job = await cli.job("launch", f"arena launch {' '.join(tokens)}", "launch", *tokens)
    return job.as_dict()


@domain.tool(destructive=True)
async def system_runtime(
    sim: Annotated[str, Field(description="simulator backend, one of the registered sims")],
    world: str | None = None,
    args: Annotated[dict[str, ArgValue] | None, Field(description="further runtime launch arguments as {name: value}")] = None,
    headless: bool = True,
) -> dict[str, Any]:
    """Start `arena runtime` (sim and arena_node, no envs) as a job."""
    await _check_sim(sim)
    tokens = launch_tokens(sim, env_n=None, world=world, robot=None, args=args, headless=headless)
    job = await cli.job("runtime", f"arena runtime {' '.join(tokens)}", "runtime", *tokens)
    return job.as_dict()


def _children() -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            parent = proc.parent_of(int(entry.name))
            if parent is not None:
                out.setdefault(parent, []).append(int(entry.name))
    return out


def tree(pid: int) -> list[int]:
    """pid and every live descendant."""
    children = _children()
    found: list[int] = []
    stack = [pid]
    while stack:
        current = stack.pop()
        if current in found or not proc.alive(current):
            continue
        found.append(current)
        stack.extend(children.get(current, []))
    return found


def _signal_group(pid: int, sig: signal.Signals) -> bool:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    if pgid in proc.ancestry() or pgid == os.getpgid(os.getpid()):
        raise ToolError(f"process group {pgid} contains this server, not signalling it")
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


def _survivors(job: Job) -> list[int]:
    return [pid for pid in job.progress.get("survivors", []) if proc.alive(pid)]


def _stoppable() -> list[Job]:
    return [job for kind in LAUNCH_KINDS for job in JOBS.list(kind=kind) if job.state == "running" or _survivors(job)]


def _launch_job(job_id: str | None) -> Job:
    if job_id is not None:
        job = JOBS.get(job_id)
        if job.kind not in LAUNCH_KINDS or job.pid is None:
            raise ToolError(f"job {job_id} is a {job.kind} job, not a launch", hint="system_stacks lists the launch jobs")
        return job
    running = _stoppable()
    if len(running) == 1:
        return running[0]
    if not running:
        raise ToolError("no launch job is running", hint="system_stacks also lists Arena processes this server did not start")
    raise ToolError(f"{len(running)} launch jobs are running, pass job_id", hint=", ".join(f"{job.id} ({job.title})" for job in running))


@domain.tool(destructive=True)
async def system_stop(
    job_id: Annotated[str | None, Field(description="the launch job, None for the only running one")] = None,
    grace_s: Annotated[float, Field(description="seconds to wait for every process of the stack to exit, at most 300")] = 40.0,
) -> dict[str, Any]:
    """SIGINT a launch job's process group and wait for its whole process tree, each further call escalates to SIGTERM, then SIGKILL."""
    if not 0 < grace_s <= 300:
        raise ToolError("grace_s must be in (0, 300]")
    job = _launch_job(job_id)
    assert job.pid is not None
    attempt = int(job.progress.get("stop_attempts", 0)) + 1
    if proc.alive(job.pid):
        targets = tree(job.pid)
        sent = signal.SIGINT if _signal_group(job.pid, signal.SIGINT) else None
    else:
        targets = _survivors(job)
        sent = signal.SIGTERM if attempt <= 2 else signal.SIGKILL
        delivered = [pid for pid in targets if _signal_group(pid, sent)]
        if not delivered:
            sent = None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace_s
    while any(proc.alive(pid) for pid in targets) and loop.time() < deadline:
        await asyncio.sleep(0.5)
    survivors = [pid for pid in targets if proc.alive(pid)]
    job.progress.update(stop_attempts=attempt, survivors=survivors)
    job.save()
    out: dict[str, Any] = {
        "job": job.id,
        "signal": sent.name if sent is not None else None,
        "stopped": not survivors,
        "survivors": [{"pid": pid, "cmdline": proc.cmdline(pid)} for pid in survivors],
    }
    if survivors:
        out["hint"] = "call system_stop again to escalate"
    return out


def _foreign(exclude: set[int]) -> list[dict[str, Any]]:
    matched: dict[int, str] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in exclude:
            continue
        line = proc.cmdline(int(entry.name))
        if LAUNCH_RE.search(line):
            matched[int(entry.name)] = line
    return [{"pid": pid, "cmdline": line} for pid, line in sorted(matched.items()) if proc.parent_of(pid) not in matched]


@domain.tool(read_only=True)
def system_stacks() -> dict[str, Any]:
    """Launch jobs of this server that still have processes, with their counts, plus outermost Arena launch processes (ros2 launch/run, the arena launch supervisor) it did not start."""
    jobs = []
    owned: set[int] = set(proc.ancestry())
    for job in _stoppable():
        roots = ([job.pid] if job.state == "running" and job.pid is not None else []) + _survivors(job)
        members = list(dict.fromkeys(pid for root in roots for pid in tree(root)))
        owned.update(members)
        jobs.append({**job.as_dict(), "processes": len(members)})
    return {"jobs": jobs, "other": _foreign(owned)}


def ros_log_dir() -> Path:
    """$ROS_LOG_DIR, else $ROS_HOME/log, else ~/.ros/log."""
    if os.environ.get("ROS_LOG_DIR"):
        return Path(os.environ["ROS_LOG_DIR"])
    if os.environ.get("ROS_HOME"):
        return Path(os.environ["ROS_HOME"]) / "log"
    return Path.home() / ".ros" / "log"


def node_log(root: Path, node: str, *, scan: int = 300) -> Path:
    """The newest log under root (two levels deep) named after the node's base name, or else carrying lines of its logger."""
    base = node.rstrip("/").rsplit("/", 1)[-1]
    named = re.compile(re.escape(base) + r"(_\d+_\d+)?\.log")
    logger = re.compile(rb"[\[.]" + re.escape(node.strip("/").replace("/", ".").encode()) + rb"\]")
    if not root.is_dir():
        raise ToolError(f"no ROS log directory at {root}", hint="set ROS_LOG_DIR, or the node has not logged yet")
    files = sorted((path for pattern in ("*.log", "*/*.log") for path in root.glob(pattern) if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)[:scan]
    for path in files:
        if named.fullmatch(path.name):
            return path
    for path in files:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            if logger.search(handle.read()):
                return path
    raise ToolError(f"no log of {node} among the {len(files)} newest files under {root}", hint="pass the node's base name, e.g. task_generator_node")


def tail(path: Path, lines: int) -> str:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 512 * 1024))
        text = handle.read().decode(errors="replace")
    return "\n".join(text.splitlines()[-lines:])


@domain.tool(read_only=True)
def system_logs(
    job_id: Annotated[str | None, Field(description="a job whose log to read")] = None,
    node: Annotated[str | None, Field(description="a ROS node name or base name whose newest ROS log file to read")] = None,
    lines: int = 200,
) -> dict[str, Any]:
    """The tail of a job's log, of a node's newest ROS log file, or, with neither, of the only running launch job."""
    if not 0 < lines <= 5000:
        raise ToolError("lines must be in (0, 5000]")
    if job_id is not None and node is not None:
        raise ToolError("pass job_id or node, not both")
    if node is not None:
        path = node_log(ros_log_dir(), node)
        return {"node": node, "path": str(path), "log": tail(path, lines)}
    job = JOBS.get(job_id) if job_id is not None else _launch_job(None)
    return {"job": job.id, "state": job.state, "path": str(job.log_path), "log": JOBS.tail(job.id, lines)}


def _colcon_guard() -> None:
    running = _running(COLCON_KINDS)
    if running:
        raise ToolError(f"{running[0].kind} job {running[0].id} is still running in this workspace", hint=f"jobs_wait {running[0].id}, two colcon runs on one tree collide")


@domain.tool(destructive=True)
async def system_build(packages: Annotated[list[str] | None, Field(description="package names (each built with every package above it) or colcon selection flags, None for the whole workspace")] = None) -> dict[str, Any]:
    """Run `arena build` as a job."""
    _colcon_guard()
    names = list(packages or [])
    return (await cli.job("build", f"arena build {' '.join(names)}".strip(), "build", *names, result=_tail_result())).as_dict()


@domain.tool(destructive=True)
async def system_rebuild(package: Annotated[str, Field(description="package whose build and install trees, and those of every package above it, are deleted and rebuilt")]) -> dict[str, Any]:
    """Run `arena rebuild <package>` as a job."""
    _colcon_guard()
    return (await cli.job("rebuild", f"arena rebuild {package}", "rebuild", package, result=_tail_result())).as_dict()


@domain.tool()
async def system_test(
    package: str,
    pytest_args: Annotated[list[str] | None, Field(description="arguments for pytest, passed as colcon --pytest-args, e.g. ['-k', 'reset']")] = None,
) -> dict[str, Any]:
    """Run `arena test <package>` (colcon test plus the failure summary) as a job."""
    _colcon_guard()
    extra = ["--pytest-args", *pytest_args] if pytest_args else []
    return (await cli.job("test", f"arena test {package}", "test", package, *extra, result=_tail_result(80))).as_dict()


@domain.tool(destructive=True, open_world=True)
async def system_deps() -> dict[str, Any]:
    """Run `arena deps` (rosdep install of the workspace's system dependencies) as a job."""
    return (await cli.job("deps", "arena deps", "deps", result=_tail_result())).as_dict()


@domain.tool(open_world=True, idempotent=True)
async def system_preload(world: str, dry_run: Annotated[bool, Field(description="report what is missing without downloading")] = False) -> dict[str, Any]:
    """Run `arena preload <world>` as a job, fetching the world's and its scenarios' assets into the local cache."""
    extra = ["--dry-run"] if dry_run else []
    return (await cli.job("preload", f"arena preload {world}", "preload", world, *extra, result=_tail_result())).as_dict()
