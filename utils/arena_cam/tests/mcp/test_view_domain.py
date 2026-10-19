from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import threading
import uuid

import pytest

pytest.importorskip("arena_mcp")
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("viewport_control_msgs.srv")
pytest.importorskip("arena_runtime_msgs.srv")
pytest.importorskip("task_generator_msgs.msg")

from arena_mcp.core import JOBS, ROS, ToolError

from arena_cam import mcp as view

REFERENCE = (10.0, 0.0)
WIDTH, HEIGHT = 32, 24


class Arena:
    """The /arena node of a runtime with a capturing sim viewport: holds, steps, a sim clock and the viewport services."""

    def __init__(self, tg: str) -> None:
        from arena_rclpy_mixins.qos import latched
        from arena_runtime_msgs.msg import EnvRecord, EnvRegistry, LockstepStatus
        from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep
        from geometry_msgs.msg import PoseStamped
        from rosgraph_msgs.msg import Clock
        from task_generator_msgs.msg import RobotDescriptor, RobotFleet, RobotState
        from viewport_control_msgs.srv import ViewportCapture, ViewportSetProjection, ViewportSetReferenceFrame, ViewportSetView

        self.node = rclpy.create_node("arena")
        self.node.declare_parameter("sim", "mujoco")
        self.requests: dict[str, list] = {"set_view": [], "set_reference_frame": [], "set_projection": [], "capture": []}
        self.steps: list[float] = []
        self.holds = 0
        self.clock_ns = 0
        self.lock = threading.Lock()
        for name, srv_type in (("set_view", ViewportSetView), ("set_reference_frame", ViewportSetReferenceFrame), ("set_projection", ViewportSetProjection)):
            self.node.create_service(srv_type, f"/arena/viewport/{name}", self._recorder(name))
        self.node.create_service(ViewportCapture, "/arena/viewport/capture", self._capture)
        self.node.create_service(LifecycleHold, "/arena/sim_lifecycle/hold", self._hold)
        self.node.create_service(LifecycleStep, "/arena/sim_lifecycle/step", self._step)
        self._clock = self.node.create_publisher(Clock, "/clock", 10)
        self._pose = self.node.create_publisher(PoseStamped, "/arena/viewport/camera_pose", 10)
        self.lockstep = self.node.create_publisher(LockstepStatus, "/arena/state/lockstep", latched(1))
        self.lockstep.publish(LockstepStatus(active=False))
        envs = self.node.create_publisher(EnvRegistry, "/arena/state/envs", latched(1))
        envs.publish(EnvRegistry(envs=[EnvRecord(env_id=0, fqn=tg, reference=list(REFERENCE), ready=True)]))
        self._fleet = self.node.create_publisher(RobotFleet, f"{tg}/state/robots", latched(1))
        self._fleet.publish(RobotFleet(robots=[RobotState(descriptor=RobotDescriptor(name="jackal", model="jackal", ns=f"{tg}/jackal", frame="env_0/jackal", base_frame="env_0/jackal/base_link"))]))
        self.node.create_timer(0.05, self._tick)

    def _stamp(self):
        from builtin_interfaces.msg import Time

        return Time(sec=self.clock_ns // 1_000_000_000, nanosec=self.clock_ns % 1_000_000_000)

    def _tick(self) -> None:
        from geometry_msgs.msg import PoseStamped
        from rosgraph_msgs.msg import Clock

        with self.lock:
            stamp = self._stamp()
        self._clock.publish(Clock(clock=stamp))
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x, pose.pose.position.z = 1.0, 5.0
        pose.pose.orientation.w = 1.0
        self._pose.publish(pose)

    def _recorder(self, name: str):
        def serve(request, response):
            self.requests[name].append(request)
            response.success = True
            response.message = "ok"
            return response

        return serve

    def _hold(self, request, response):
        with self.lock:
            self.holds += 1 if request.action == request.ACQUIRE else -1
            response.paused = self.holds > 0
            response.hold_count = self.holds
        return response

    def _step(self, request, response):
        with self.lock:
            if self.holds <= 0:
                response.success = False
                response.error_msg = "step requires an active hold and no open unpause window"
                return response
            self.clock_ns += round(request.seconds * 1e9)
            self.steps.append(request.seconds)
        response.success = True
        response.advanced = request.seconds
        return response

    def _capture(self, request, response):
        from sensor_msgs.msg import Image

        with self.lock:
            reached = self.clock_ns >= request.min_sim_time.sec * 1_000_000_000 + request.min_sim_time.nanosec
            stamp = self._stamp()
        self.requests["capture"].append(request)
        if not reached:
            response.success = False
            response.message = "capture timed out, min_sim_time not reached"
            return response
        response.image = Image(height=HEIGHT, width=WIDTH, encoding="rgb8", step=WIDTH * 3, data=bytes([len(self.requests["capture"]) % 256]) * (WIDTH * HEIGHT * 3))
        response.image.header.stamp = stamp
        response.success = True
        response.message = "ok"
        return response


@dataclasses.dataclass
class Rig:
    loop: asyncio.AbstractEventLoop
    tg: str
    arena: Arena

    def run(self, coro):
        return self.loop.run_until_complete(coro)


@pytest.fixture(scope="module")
def rig():
    from rclpy.executors import MultiThreadedExecutor

    owned = rclpy.ok()
    if not owned:
        rclpy.init()
    tg = f"/mcp_{uuid.uuid4().hex[:8]}/env_0/task_generator_node"
    arena = Arena(tg)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(arena.node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    loop = asyncio.new_event_loop()
    try:
        yield Rig(loop=loop, tg=tg, arena=arena)
    finally:
        executor.shutdown()
        arena.node.destroy_node()
        loop.run_until_complete(ROS.shutdown())
        loop.close()
        if rclpy.ok():
            rclpy.shutdown()
        if owned:
            rclpy.init()


def test_view_domain_is_served_from_its_entry_point():
    from arena_mcp.server import discover

    (loaded,) = discover(["view"])
    assert loaded.served, loaded.reason
    assert {spec.name for spec in loaded.domain.tools} == {"view_available", "view_set", "view_capture", "view_frame_exact", "view_record"}
    assert [resource.uri for resource in loaded.domain.resources] == ["arena://view/shots"]


def test_shots_resource_lists_the_cam_registry():
    from arena_cam.registry import PRIMITIVES

    catalog = view.view_shots()
    assert set(catalog["verbs"]) == set(PRIMITIVES)
    assert {"look", "orbit", "hold", "track"} <= set(catalog["verbs"])


def test_available_names_the_simulator_and_the_serving_node(rig: Rig):
    out = rig.run(view.view_available())
    assert out["sim"] == "mujoco"
    assert out["viewports"] == [{"surface": "sim", "tag": "sim", "ns": "/arena/viewport", "capture": True, "served_by": ["/arena"]}]
    assert "hint" not in out


def test_set_and_capture_place_env_poses_in_the_world_and_follow_robots(rig: Rig):
    async def scenario():
        placed = await view.view_set(view.View(eye=(1.0, 0.0, 2.0), target=(0.0, 0.0, 0.0), projection="orthographic"))
        captured = await view.view_capture()
        followed = await view.view_set(view.View(follow="jackal"))
        down = await view.view_capture(view.View(top_down=6.0, target=(2.0, 3.0, 0.0)))
        with pytest.raises(ToolError, match="no robot 'ghost'"):
            await view.view_set(view.View(follow="ghost"))
        with pytest.raises(ToolError, match="needs eye, top_down or follow"):
            await view.view_set(view.View())
        with pytest.raises(ToolError, match="no rviz viewport"):
            await view.view_capture(surface="rviz")
        return placed, captured, followed, down

    placed, captured, followed, down = rig.run(scenario())
    requests = rig.arena.requests
    first_view = requests["set_view"][0]
    assert (first_view.eye.x, first_view.eye.y, first_view.eye.z) == (11.0, 0.0, 2.0)
    assert (first_view.target.x, first_view.target.y, first_view.target.z) == (10.0, 0.0, 0.0)
    assert requests["set_projection"][0].projection == "orthographic"
    world, follow, top = requests["set_reference_frame"][:3]
    assert world.has_pose and world.entity == "" and world.mode == world.FULL
    assert follow.entity == "env_0/jackal/base_link" and follow.mode == follow.YAW_ONLY
    assert top.has_pose and top.entity == ""
    assert placed["camera"]["x"] == pytest.approx(11.0)
    summary, image = captured
    assert requests["capture"][0].pose.position.x == pytest.approx(11.0)
    assert (summary["width"], summary["height"]) == (WIDTH, HEIGHT)
    assert image.to_image_content().mime_type == "image/png"
    assert followed["following"] == "jackal"
    assert (requests["set_view"][1].eye.x, requests["set_view"][1].eye.z) == (-3.0, 2.0)
    assert requests["capture"][1].pose.position.x == pytest.approx(12.0)
    assert requests["capture"][1].pose.position.z == pytest.approx(6.0)
    assert down[0]["camera"]["y"] == pytest.approx(3.0)


def test_frame_exact_steps_the_held_sim_per_frame_and_releases_the_hold(rig: Rig):
    from arena_runtime_msgs.msg import LockstepStatus

    async def scenario():
        before = rig.arena.clock_ns
        frames = await view.view_frame_exact(robot="jackal", frames=3, step_s=0.1)
        rig.arena.lockstep.publish(LockstepStatus(active=True))
        await asyncio.sleep(0.5)
        with pytest.raises(ToolError, match="lockstep run owns"):
            await view.view_frame_exact()
        rig.arena.lockstep.publish(LockstepStatus(active=False))
        await asyncio.sleep(0.5)
        return before, frames

    before, frames = rig.run(scenario())
    summary, *images = frames
    assert len(images) == summary["frames"] == 3
    assert rig.arena.steps[-3:] == [0.1, 0.1, 0.1]
    assert summary["sim_times"] == pytest.approx([before * 1e-9 + 0.1 * (index + 1) for index in range(3)])
    assert summary["following"] == "jackal"
    assert rig.arena.holds == 0
    gated = [request.min_sim_time.sec + request.min_sim_time.nanosec * 1e-9 for request in rig.arena.requests["capture"][-3:]]
    assert gated == pytest.approx(summary["sim_times"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None or not os.environ.get("ARENA_DIR"), reason="needs ffmpeg and a sourced Arena checkout")
def test_record_films_a_lockstep_take_through_the_cam_cli(rig: Rig, tmp_path):
    out = tmp_path / "take.mp4"

    async def scenario():
        with pytest.raises(ToolError, match="cannot play 'no_such_shot'"):
            await view.view_record("no_such_shot")
        captures = len(rig.arena.requests["capture"])
        job = await view.view_record("hold", {"duration": 0.2}, out=str(out))
        done = await JOBS.wait(job["id"], 120.0)
        return job, done, len(rig.arena.requests["capture"]) - captures

    job, done, frames = rig.run(scenario())
    assert job["output"] == str(out)
    assert done.state == "succeeded", JOBS.tail(done.id, 40)
    assert done.result == {"files": [str(out)]}
    assert out.stat().st_size > 0
    assert frames == 6
    assert rig.arena.holds == 0
