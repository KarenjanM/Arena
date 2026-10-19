from __future__ import annotations

import asyncio
import dataclasses
import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field

from arena_mcp.core import LATCHED, RELIABLE, ROS, Domain, ToolError, cli, fleet, media, msgs

if TYPE_CHECKING:
    from arena_rclpy_mixins.Time import Time
    from sensor_msgs.msg import Image

    from arena_cam.curves import Quat, Vec3
    from arena_mcp.core.ros import ServiceType

domain = Domain(
    "view",
    """Viewport cameras: aim and capture the sim GUI camera (/arena/viewport: MuJoCo, Isaac, Gazebo while its GUI runs) or an env's rviz camera.
    Poses are in the env frame (the world frame when several envs run and env_id is omitted), relative to the robot while following one.
    view_frame_exact holds the sim and steps it per frame, view_record films an arena cam take in lockstep as a job.""",
    requires=("viewport_control_msgs", "arena_runtime_msgs"),
)

SIM = "/arena"
HOLD = "/arena/sim_lifecycle/hold"
STEP = "/arena/sim_lifecycle/step"
LOCKSTEP = "/arena/state/lockstep"
NO_VIEWPORT = "MuJoCo and Isaac serve /arena/viewport, Gazebo only while its GUI runs (launch without headless:=true), an env's rviz camera needs arena viz, view_available lists what is up"
MAX_FRAMES = 8
FPS = 30.0
CHASE_EYE = (-3.0, 0.0, 2.0)
Surface = Literal["sim", "rviz"]


class View(BaseModel):
    """Where the camera looks: from eye at target, or straight down from top_down metres, optionally following a robot."""

    eye: tuple[float, float, float] | None = Field(default=None, description="camera position, default 3 m behind and 2 m above a followed robot")
    target: tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="look-at point")
    top_down: float | None = Field(default=None, description="height above target, looks straight down and ignores eye")
    follow: str | None = Field(default=None, description="robot name, eye and target become relative to its base frame")
    mode: Literal["full", "yaw", "position"] = Field(default="yaw", description="how much of the followed robot's rotation the camera inherits")
    fov: float = Field(default=0.0, description="horizontal field of view in rad, 0 keeps the current one")
    projection: Literal["perspective", "orthographic"] | None = None


@dataclasses.dataclass(frozen=True)
class _Shot:
    eye: Vec3
    target: Vec3
    quat: Quat
    fov: float
    following: str | None


@dataclasses.dataclass(frozen=True)
class _Endpoint:
    ns: str
    offset: tuple[float, float]
    reference: tuple[float, float]


class _Reply(Protocol):
    @property
    def success(self) -> object: ...

    @property
    def message(self) -> object: ...


_last: dict[str, _Shot] = {}


async def _endpoint(surface: Surface, env_id: int | None) -> _Endpoint:
    from arena_runtime_msgs.msg import EnvRegistry

    from arena_cam import surfaces

    try:
        refs = surfaces.env_refs(await ROS.latest(EnvRegistry, surfaces.ENVS_TOPIC, qos=LATCHED, timeout=1.0))
    except ToolError:
        refs = {}
    if env_id is not None and env_id not in refs:
        raise ToolError(f"no env {env_id}", hint=f"env_ids: {sorted(refs)}" if refs else "no Arena runtime is up")
    env = env_id if env_id is not None else next(iter(refs)) if len(refs) == 1 else None
    if surface == "rviz" and env is None:
        raise ToolError("an rviz camera belongs to one env, pass env_id", hint=f"env_ids: {sorted(refs)}")
    selection = surfaces.TargetSelection(include_sim=surface == "sim", viz_all=False, viz_env=env if surface == "rviz" else None)
    found = surfaces.find_targets((await ROS.graph())["services"], selection, refs)
    if not found:
        raise ToolError(f"no {surface} viewport is up", hint=NO_VIEWPORT)
    ns, offset = found[0]
    return _Endpoint(ns=ns, offset=offset, reference=refs[env] if env is not None else (0.0, 0.0))


async def _call[RequestT, ResponseT: _Reply](srv_type: ServiceType[RequestT, ResponseT], name: str, request: RequestT, timeout: float = 10.0) -> ResponseT:
    response = await ROS.call(srv_type, name, request, timeout=timeout, hint=NO_VIEWPORT)
    if not response.success:
        raise ToolError(f"{name} rejected: {response.message}")
    return response


async def _base_frame(env_id: int | None, robot: str) -> str:
    robots = await fleet(env_id)
    for entry in robots:
        if robot in (entry["name"], entry["ns"]):
            return entry["base_frame"]
    raise ToolError(f"no robot {robot!r}", hint=f"robots: {[entry['name'] for entry in robots]}")


async def _apply(endpoint: _Endpoint, view: View, env_id: int | None) -> _Shot:
    """Set the projection and reference frame view asks for and return its camera pose in the viewport frame."""
    from viewport_control_msgs.srv import ViewportSetProjection, ViewportSetReferenceFrame

    from arena_cam import curves, surfaces

    ns = endpoint.ns
    if view.projection is not None:
        await _call(ViewportSetProjection, f"{ns}/viewport/set_projection", ViewportSetProjection.Request(projection=view.projection))
    request = ViewportSetReferenceFrame.Request(mode=surfaces.REFERENCE_MODES[view.mode if view.follow else "full"])
    if view.follow:
        request.entity = await _base_frame(env_id, view.follow)
    else:
        request.has_pose = True
        request.pose = surfaces.ros_pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    target: Vec3 = view.target
    if view.top_down is not None:
        eye = curves.vadd(target, (0.0, 0.0, view.top_down))
    elif view.eye is not None:
        eye = view.eye
    elif view.follow:
        eye = curves.vadd(target, CHASE_EYE)
    else:
        raise ToolError("a view needs eye, top_down or follow")
    if not view.follow:
        lift = (endpoint.reference[0], endpoint.reference[1], 0.0)
        eye = surfaces.localize(endpoint.offset, curves.vadd(eye, lift))
        target = surfaces.localize(endpoint.offset, curves.vadd(target, lift))
    await _call(ViewportSetReferenceFrame, f"{ns}/viewport/set_reference_frame", request)
    shot = _Shot(eye=eye, target=target, quat=curves.look_at_quat(eye, target), fov=view.fov, following=view.follow)
    _last[ns] = shot
    return shot


async def _current(endpoint: _Endpoint) -> _Shot:
    from geometry_msgs.msg import PoseStamped

    from arena_cam import curves

    if endpoint.ns in _last:
        return _last[endpoint.ns]
    pose = (await ROS.latest(PoseStamped, f"{endpoint.ns}/viewport/camera_pose", qos=RELIABLE, timeout=3.0, hint=NO_VIEWPORT)).pose
    eye, quat = (pose.position.x, pose.position.y, pose.position.z), (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
    return _Shot(eye=eye, target=curves.vadd(eye, curves.quat_forward(quat)), quat=quat, fov=0.0, following=None)


async def _capture(endpoint: _Endpoint, shot: _Shot, min_sim_time: Time | None = None) -> Image:
    from viewport_control_msgs.srv import ViewportCapture

    from arena_cam import surfaces

    request = ViewportCapture.Request(pose=surfaces.ros_pose(shot.eye, shot.quat), fov=float(shot.fov))
    if min_sim_time is not None:
        request.min_sim_time = min_sim_time.to_msg()
    return (await _call(ViewportCapture, f"{endpoint.ns}/viewport/capture", request, timeout=30.0)).image


def _describe(endpoint: _Endpoint, shot: _Shot) -> dict[str, Any]:
    from arena_cam import surfaces

    return {"viewport": f"{endpoint.ns}/viewport", "camera": msgs.pose_dict(surfaces.ros_pose(shot.eye, shot.quat)), "following": shot.following}


@domain.tool(read_only=True)
async def view_available() -> dict[str, Any]:
    """Which viewport cameras are up right now, the node serving each, whether it captures, and the simulator."""
    from arena_cam import surfaces

    graph = await ROS.graph()
    services = set(graph["services"])
    found = surfaces.find_targets(graph["services"], surfaces.TargetSelection(include_sim=True, viz_all=True, viz_env=None), {})
    node = await ROS.node()
    providers: dict[str, list[str]] = {}
    for name, namespace in node.get_node_names_and_namespaces():
        for service, _ in node.get_service_names_and_types_by_node(name, namespace):
            if service.endswith(surfaces.VIEW_SUFFIX):
                providers.setdefault(service[: -len(surfaces.VIEW_SUFFIX)], []).append(f"{namespace.rstrip('/')}/{name}")
    try:
        sim = (await ROS.get_params(SIM, ["sim"]))["sim"]
    except ToolError:
        sim = None
    viewports = [{"surface": "sim" if ns == SIM else "rviz", "tag": surfaces.tag(ns), "ns": f"{ns}/viewport", "capture": f"{ns}/viewport/capture" in services, "served_by": providers.get(ns, [])} for ns, _ in found]
    out: dict[str, Any] = {"sim": sim, "viewports": viewports}
    if not any(entry["surface"] == "sim" for entry in viewports):
        out["hint"] = "Gazebo serves /arena/viewport only while its GUI runs, relaunch without headless:=true" if sim == "gazebo" else NO_VIEWPORT
    return out


@domain.tool(idempotent=True)
async def view_set(view: View, env_id: int | None = None, surface: Surface = "sim") -> dict[str, Any]:
    """Aim a viewport camera and leave it there: projection, reference frame (a followed robot or the world) and view."""
    from geometry_msgs.msg import Point
    from viewport_control_msgs.srv import ViewportSetView

    endpoint = await _endpoint(surface, env_id)
    shot = await _apply(endpoint, view, env_id)
    eye, target = shot.eye, shot.target
    request = ViewportSetView.Request(eye=Point(x=eye[0], y=eye[1], z=eye[2]), target=Point(x=target[0], y=target[1], z=target[2]), fov=float(shot.fov))
    await _call(ViewportSetView, f"{endpoint.ns}/viewport/set_view", request)
    return _describe(endpoint, shot)


@domain.tool(idempotent=True)
async def view_capture(view: View | None = None, env_id: int | None = None, surface: Surface = "sim") -> list[Any]:
    """One rendered frame of a viewport camera, aimed by view when given, else at the last view_set or the current camera pose."""
    endpoint = await _endpoint(surface, env_id)
    shot = await _apply(endpoint, view, env_id) if view is not None else await _current(endpoint)
    image = await _capture(endpoint, shot)
    return [{**_describe(endpoint, shot), "stamp": msgs.seconds(image.header.stamp), "width": image.width, "height": image.height}, media.image_msg(image)]


async def _held_sim_time() -> Time:
    node = await ROS.node()
    seen = node.sim_time
    for _ in range(25):
        await asyncio.sleep(0.2)
        if node.sim_time == seen:
            break
        seen = node.sim_time
    return seen


@domain.tool()
async def view_frame_exact(
    env_id: int | None = None,
    robot: Annotated[str | None, Field(description="follow this robot, shorthand for view.follow")] = None,
    view: View | None = None,
    frames: Annotated[int, Field(description=f"frames to capture, at most {MAX_FRAMES}")] = 1,
    step_s: Annotated[float, Field(description="sim seconds stepped before each frame, 0 captures the held tick")] = 1.0 / 30.0,
    surface: Surface = "sim",
) -> list[Any]:
    """Frame-exact capture: hold the sim, step it step_s per frame, capture each frame once the sim reached that time, release the hold."""
    from arena_rclpy_mixins.Time import Time
    from arena_runtime_msgs.msg import LockstepStatus
    from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep

    if not 1 <= frames <= MAX_FRAMES:
        raise ToolError(f"frames must be in [1, {MAX_FRAMES}]")
    if step_s < 0.0:
        raise ToolError("step_s must not be negative")
    try:
        status = await ROS.latest(LockstepStatus, LOCKSTEP, qos=LATCHED, timeout=0.5)
    except ToolError:
        status = None
    if status is not None and status.active:
        raise ToolError("a lockstep run owns the sim stepping", hint="view_record rides it frame-exactly, or stop the run first")
    if robot is not None:
        view = (view or View()).model_copy(update={"follow": robot})
    endpoint = await _endpoint(surface, env_id)
    shot = await _apply(endpoint, view, env_id) if view is not None else await _current(endpoint)
    caller = (await ROS.node()).get_fully_qualified_name()
    hold = LifecycleHold.Request(action=LifecycleHold.Request.ACQUIRE, caller_id=caller, reason="view_frame_exact")
    await ROS.call(LifecycleHold, HOLD, hold, hint="no Arena runtime is up, view_capture works without one")
    images: list[Image] = []
    try:
        tick = await _held_sim_time()
        for _ in range(frames):
            if step_s > 0.0:
                stepped = await ROS.call(LifecycleStep, STEP, LifecycleStep.Request(seconds=step_s), timeout=30.0)
                if not stepped.success:
                    raise ToolError(f"step failed: {stepped.error_msg}")
                tick = tick + Time.from_float(stepped.advanced)
            images.append(await _capture(endpoint, shot, min_sim_time=tick))
    finally:
        await ROS.call(LifecycleHold, HOLD, LifecycleHold.Request(action=LifecycleHold.Request.RELEASE, caller_id=caller, reason="view_frame_exact"))
    return [{**_describe(endpoint, shot), "frames": len(images), "sim_times": [msgs.seconds(image.header.stamp) for image in images]}, *(media.image_msg(image) for image in images)]


def _catalog() -> tuple[list[str], list[str]]:
    from arena_cam.registry import PRIMITIVES
    from arena_cam.shots import SHOTS, discover

    importlib.import_module("arena_cam.camera")
    discover()
    return sorted(PRIMITIVES), sorted(SHOTS)


@domain.resource("arena://view/shots", mime_type="application/json", description="arena cam verbs and installed shots, the names view_record films")
def view_shots() -> dict[str, list[str]]:
    verbs, shots = _catalog()
    return {"verbs": verbs, "shots": shots}


@domain.tool(destructive=True)
async def view_record(
    name: Annotated[str, Field(description="arena cam verb, shot or shot .yaml path, arena://view/shots lists them")],
    params: Annotated[dict[str, str | float | bool] | None, Field(description="verb or shot parameters, a tuple as 'x,y,z'")] = None,
    out: Annotated[str | None, Field(description="output file, a bare name lands under $ARENA_DATA_DIR/recordings, default <name>_<timestamp>.mp4")] = None,
    surface: Annotated[Literal["sim", "rviz", "all"], Field(description="which cameras film, all and several rviz cameras write one file per camera")] = "sim",
    env_id: Annotated[int | None, Field(description="the env whose rviz camera films, every env's when omitted")] = None,
    force: Annotated[bool, Field(description="overwrite an existing file")] = False,
) -> dict[str, Any]:
    """Film an arena cam take at 30 fps in lockstep, one 1/30 s sim step per frame, as a job whose result lists the written files."""
    from arena_cam import Camera, TargetSelection, load_shot
    from arena_cam.cli import _parse_params
    from arena_cam.record import default_name, resolve_path

    tokens = [f"{key}={value}" for key, value in (params or {}).items()]
    is_path = name.endswith((".yaml", ".yml")) or "/" in name or os.path.exists(name)
    targets = TargetSelection(include_sim=surface != "rviz", viz_all=surface != "sim" and env_id is None, viz_env=env_id if surface != "sim" else None)
    try:
        if is_path:
            load_shot(name, targets)
        else:
            Camera(targets).add(name, _parse_params(tokens))
    except (ValueError, OSError, SystemExit) as exc:
        verbs, shots = _catalog()
        raise ToolError(f"arena cam cannot play {name!r}: {exc}", hint=f"verbs: {', '.join(verbs)}, shots: {', '.join(shots) or 'none'}") from exc
    target = out or default_name(Path(name).stem)
    path = resolve_path(target)
    args = ["cam", name, *tokens, "--record", target, "--fps", f"{FPS:g}", "--lockstep"]
    if surface == "sim":
        args.append("--sim")
    elif surface == "rviz":
        args += ["--viz", str(env_id)] if env_id is not None else ["--viz"]
    if force:
        args.append("--force")

    def written(_job: object) -> dict[str, Any]:
        return {"files": [str(candidate) for candidate in [path, *sorted(path.parent.glob(f"{path.stem}-*{path.suffix}"))] if candidate.is_file()]}

    job = await cli.job("view", f"arena cam {name} --record {target}", *args, result=written)
    return {**job.as_dict(), "output": str(path)}
