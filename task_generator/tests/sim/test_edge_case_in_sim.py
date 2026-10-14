"""End-to-end: a generated prompt scenario runs in the real stack.

Boots `arena launch` (gazebo, arena_humansim, nav2, the edge_case modes) on one generated
scenario and checks the whole chain the demos rely on: the case builds, the robot sets off and
the case clock starts, every runtime effect fires, the case's own encounter and formation agents come
within 1.5 m of the robot, the population walks, the robot's lidar sees
the pedestrians and HumanSim is told where the robot is, the score row lands, and nothing
crashes. Minutes per test and it owns the machine's simulator, so it only runs with
`ARENA_SIM_TESTS=1` (see `data/promptgen/tools/sim_test.sh`).
"""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.sim, pytest.mark.slow]

WS = Path("/opt/arena_ws")
TOOLS = WS / "data/promptgen/tools"
WORLD = os.environ.get("ARENA_SIM_TEST_WORLD", "hospital_1")
SCENARIO = os.environ.get("ARENA_SIM_TEST_SCENARIO", "normal__hospital_1_001")
ROBOT = os.environ.get("ARENA_SIM_TEST_ROBOT", "jackal")
LOG = Path(f"/tmp/sim_test_{SCENARIO}.log")
BOOT_S = 420.0
DEPART_S = 40.0
EFFECTS_S = 90.0


def _skip_unless_enabled() -> None:
    if os.environ.get("ARENA_SIM_TESTS") != "1":
        pytest.skip("set ARENA_SIM_TESTS=1 to boot the simulator")
    pytest.importorskip("rclpy")
    if not (WS / "source").exists():
        pytest.skip("runs inside the project container")


def _log() -> str:
    return LOG.read_text(errors="replace") if LOG.exists() else ""


def _wait(pattern: str, timeout: float, *, since: int = 0) -> re.Match[str] | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        m = re.search(pattern, _log()[since:])
        if m:
            return m
        time.sleep(2.0)
    return None


def _topic(node: object, suffix: str, kind: str = "") -> str:
    import rclpy

    for _ in range(40):
        names = sorted(n for n, types in node.get_topic_names_and_types() if n.endswith(suffix) and (not kind or kind in types))
        if names:
            return names[0]
        rclpy.spin_once(node, timeout_sec=0.5)
    raise AssertionError(f"no topic ending in {suffix}")


def _tf_buffer(node: object) -> object:
    """A tf2 buffer fed from the env's own tf topics: `TransformListener` reads the global `/tf`, which the env does not publish on."""
    import tf2_ros
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from tf2_msgs.msg import TFMessage

    buf = tf2_ros.Buffer()
    static = _topic(node, "/tf_static")

    def feed(msg: TFMessage, is_static: bool) -> None:
        for t in msg.transforms:
            (buf.set_transform_static if is_static else buf.set_transform)(t, "sim_test")

    node.create_subscription(TFMessage, static, lambda m: feed(m, True), QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(TFMessage, static.removesuffix("_static"), lambda m: feed(m, False), 100)
    return buf


@pytest.fixture(scope="module")
def stack():
    _skip_unless_enabled()
    subprocess.run(["bash", str(TOOLS / "killsim.sh")], capture_output=True, timeout=60)
    LOG.unlink(missing_ok=True)
    cmd = (
        f"source {WS}/source >/dev/null 2>&1; export GZ_IP=127.0.0.1 GZ_RELAY=127.0.0.1; cd {WS} && "
        f"arena launch sim:=gazebo world:={WORLD} robot:={ROBOT} human:=arena task.robots:=edge_case task.obstacles:=edge_case "
        f"task.edge_case_robot.retries:=3 task.edge_case_robot.on_blocked:=continue headless:=true task.episode.auto_reset:=true "
        f"task.edge_case.record_dir:=/tmp/sim_test_rec/{SCENARIO} task.scenario.file:={SCENARIO} > {LOG} 2>&1"
    )
    proc = subprocess.Popen(["bash", "-lc", cmd], start_new_session=True)
    try:
        yield proc
    finally:
        # The launch's own process group first; killsim.sh then sweeps what outlives it. Note
        # that killsim.sh matches on command lines ("task_generator", "arena launch", ...): the
        # process running this test must not carry those words (see sim_test.sh).
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=20)
        except Exception:  # noqa: BLE001 - already gone, or stuck: the sweep below handles it
            pass
        subprocess.run(["bash", str(TOOLS / "killsim.sh")], capture_output=True, timeout=120)


@pytest.fixture(scope="module")
def episode(stack) -> dict:
    """The first episode the case is built for (episode 1 runs before the robot is placed)."""
    started = _wait(r"EPISODE STARTED #2", BOOT_S)
    assert started, f"no second episode within {BOOT_S:.0f}s; log tail:\n" + _log()[-3000:]
    log = _log()
    first = log.index("EPISODE STARTED #1")
    block = log[first : started.start()]
    built = re.search(rf"edge_case: {re.escape(SCENARIO)} built - (\d+) injected, (\d+) plan\(s\), (\d+) base agent\(s\)", block)
    # an effect-free case is read at the reset, before episode 1 starts
    no_effects = re.search(rf"-> {re.escape(SCENARIO)}: 0 effect\(s\)", log[: started.start()])
    assert built or no_effects, "the obstacles mode never reported the case built:\n" + block[-3000:]
    return {"at": started.start(), "plans": int(built.group(2)) if built else 0, "agents": int(built.group(3)) if built else 0}


def test_the_case_builds_without_abort(episode) -> None:
    block = _log()[: episode["at"]]
    assert "Traceback" not in block.split("EPISODE STARTED #1")[-1], "a traceback during the reset"
    assert not re.search(r"CaseAborted|aborting the case", block.split("EPISODE STARTED #1")[-1])


def test_the_robot_sets_off_and_the_case_clock_starts(episode) -> None:
    m = _wait(r"under way \(([0-9.]+) m from its start[^)]*\) ([0-9.]+)s after", DEPART_S, since=episode["at"])
    assert m, f"the robot did not set off within {DEPART_S:.0f}s of the episode start"
    assert float(m.group(2)) < DEPART_S


def test_every_runtime_effect_fires(episode) -> None:
    if episode["plans"] == 0:
        pytest.skip("this case has no runtime effect")
    armed = re.findall(r"edge_case: armed (\S+) over \d+ agent\(s\), first rewrite at t=([0-9.]+)s( once [^\n]{0,80})?", _log()[: episode["at"]])
    assert armed, "no plans armed"
    ungated = [(label, float(at)) for label, at, gate in armed if not gate]
    for label, at in ungated:
        if at > EFFECTS_S:
            continue  # beyond what this test waits for
        assert _wait(rf"edge_case: {re.escape(label)}(?: at t=| fired at t=| trigger fired)", EFFECTS_S + at, since=episode["at"]), f"{label} (due at {at:.0f}s) never fired"


def _case(root: Path) -> dict:
    # the generator's reader: format 2 humans under `dynamic`, the block from edge_case.yaml
    from arena_evaluation.generation.formats import read_scenario

    return read_scenario(root / "scenario.yaml")


def _await_stimulus(effect: dict, episode: dict) -> None:
    fired = _wait(rf"edge_case: {re.escape(effect['label'])} at t=[0-9.]+s - (\d+) of (\d+) agent\(s\) stimulated", EFFECTS_S + float(effect["when"].get("at", 0.0)), since=episode["at"])
    assert fired and fired.group(1) == fired.group(2), f"{effect['label']}: the stimulus reached {fired.groups() if fired else 'nobody'}"


def _handed_to(type_doc: dict, need: str) -> dict:
    to = next((t["goto"] for seq in type_doc["sequences"].values() for t in seq.get("transitions", []) if need in t["when"]), None)
    return type_doc["sequences"].get(to, {})


def _speed_polys(path: Path, need: str) -> list[tuple[float, ...]]:
    """The `desired_velocity` transforms raising `need` holds on the type at `path`, in HumanSim's fold order: its
    type-level `tunes:` on the need, then the sequence the need preempts into."""
    from arena_humansim.core.agents.loader import load_agent_type_from_file

    agent_type = load_agent_type_from_file(path)
    tunes = [entry.tune for entry in agent_type.tunes if need in entry.when]
    to = next((t["goto"] for seq in yaml.safe_load(path.read_text())["sequences"].values() for t in seq.get("transitions", []) if need in t["when"]), None)
    if to is not None and agent_type.sequences[to].tune is not None:
        tunes.append(agent_type.sequences[to].tune)
    return [tune.fields["desired_velocity"] for tune in tunes if "desired_velocity" in tune.fields]


def _destination(type_doc: dict, need: str) -> tuple[float, float] | None:
    # the first go_to of the sequence the need preempts into
    step = next((st for st in _handed_to(type_doc, need).get("steps", {}).values() if st.get("kind") == "go_to"), None)
    return None if step is None else (float(step["target_pose"]["x"]), float(step["target_pose"]["y"]))


def test_stimulated_trees_head_for_their_destination(episode) -> None:
    import rclpy
    from arena_people_msgs.msg import Pedestrians
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from tf2_msgs.msg import TFMessage

    root = WS / "src/Arena/arena_simulation_setup/worlds" / WORLD / "scenarios" / SCENARIO
    case = _case(root)
    stimuli = [e for e in (case.get("edge_case") or {}).get("effects", []) if e["type"] == "stimulus"]
    if not stimuli:
        pytest.skip("this case raises no need")
    types = {a["name"]: a["agent"]["agent_type"] for a in case["dynamic"]}
    goals: dict[str, tuple[float, float]] = {}
    for effect in stimuli:
        _await_stimulus(effect, episode)
        for name in effect["scope"]:
            goal = _destination(yaml.safe_load((root / types[name]).read_text()), effect["need"])
            if goal is not None:
                goals[name] = goal
    if not goals:
        pytest.skip("no stimulated sequence walks anywhere")

    if not rclpy.ok():
        rclpy.init()
    node = Node("sim_test_stimulus_probe")
    first: dict[str, float] = {}
    nearest: dict[str, float] = {}
    last: dict[str, tuple[float, float]] = {}
    offset: list[tuple[float, float]] = []

    def on_peds(msg: Pedestrians) -> None:
        if not offset:
            return
        for p in msg.pedestrians:
            name = next((n for n in goals if p.name == n or p.name.endswith(f"/{n}")), None)
            if name is None:
                continue
            gx, gy = goals[name][0] + offset[0][0], goals[name][1] + offset[0][1]
            d = math.hypot(p.pose.position.x - gx, p.pose.position.y - gy)
            first.setdefault(name, d)
            nearest[name] = min(nearest.get(name, math.inf), d)
            last[name] = (round(p.pose.position.x - offset[0][0], 1), round(p.pose.position.y - offset[0][1], 1))

    def on_static(msg: TFMessage) -> None:
        # the world frame the generator wrote in sits in `map` at the env's `<prefix>/map` anchor
        for t in msg.transforms:
            if not offset and t.header.frame_id.strip("/") == "map" and t.child_frame_id.strip("/").endswith("/map"):
                offset.append((t.transform.translation.x, t.transform.translation.y))

    # tf is per env under gazebo (`<env ns>/tf_static`), global otherwise
    node.create_subscription(TFMessage, _topic(node, "/tf_static"), on_static, QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(Pedestrians, _topic(node, "/arena_peds"), on_peds, 10)
    t0 = time.time()
    while time.time() - t0 < 60.0 and not (len(nearest) == len(goals) and all(d < 2.5 for d in nearest.values())):
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()

    assert offset, "no env anchor (map -> <prefix>/map) on tf_static"
    assert set(nearest) == set(goals), f"stimulated agents not on /arena_peds: {sorted(set(goals) - set(nearest))}"
    stuck = {n: (round(first[n], 1), round(nearest[n], 1), last[n], goals[n]) for n in goals if not (nearest[n] < 2.5 or first[n] - nearest[n] > 3.0)}
    assert not stuck, f"stimulated agents neither reached nor closed on their destination (first, nearest m, last, goal xy): {stuck}"


def test_stimulated_trees_take_their_tune(episode) -> None:
    import rclpy
    from arena_humansim.core.agents.tune import polyval
    from arena_humansim_msgs.msg import AgentFrame, AgentMeta
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile

    root = WS / "src/Arena/arena_simulation_setup/worlds" / WORLD / "scenarios" / SCENARIO
    case = _case(root)
    agents = {a["name"]: a for a in case["dynamic"]}
    expected: dict[str, float] = {}
    for effect in (e for e in (case.get("edge_case") or {}).get("effects", []) if e["type"] == "stimulus"):
        _await_stimulus(effect, episode)
        for name in effect["scope"]:
            block = agents[name]["agent"]
            polys = _speed_polys(root / block["agent_type"], effect["need"])
            # the spawn speed is the agent block's; without one it was sampled and cannot be predicted
            if polys and block.get("desired_velocity"):
                speed = float(block["desired_velocity"])
                for poly in polys:
                    speed = polyval(poly, speed)
                expected[name] = speed
    if not expected:
        pytest.skip("no stimulated agent holds a speed tune")

    if not rclpy.ok():
        rclpy.init()
    node = Node("sim_test_tune_probe")
    names: dict[int, str] = {}
    speeds: dict[str, list[float]] = {}

    def on_meta(msg: AgentMeta) -> None:
        names.update((aid, n.rsplit("/", 1)[-1]) for aid, n in zip(msg.agent_id, msg.name, strict=True))

    def on_frame(msg: AgentFrame) -> None:
        for aid, speed in zip(msg.agent_id, msg.desired_velocity, strict=True):
            if names.get(aid) in expected:
                speeds.setdefault(names[aid], []).append(speed)

    latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(AgentMeta, _topic(node, "/agent_meta", "arena_humansim_msgs/msg/AgentMeta"), on_meta, latched)
    node.create_subscription(AgentFrame, _topic(node, "/agent_states", "arena_humansim_msgs/msg/AgentFrame"), on_frame, 10)
    t0 = time.time()
    while time.time() - t0 < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()

    assert set(speeds) == set(expected), f"tuned agents not on the frame: {sorted(set(expected) - set(speeds))}"
    off = {n: (round(expected[n], 2), round(sorted(v)[len(v) // 2], 2)) for n, v in speeds.items() if abs(sorted(v)[len(v) // 2] - expected[n]) > 0.05}
    assert not off, f"desired speed after the stimulus (expected, median seen): {off}"


def test_the_population_walks_and_the_robot_is_seen_and_sees(episode) -> None:
    import rclpy
    from arena_humansim_msgs.msg import AgentStates
    from arena_people_msgs.msg import Pedestrians
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import LaserScan

    if not rclpy.ok():
        rclpy.init()  # the ros conftest may already have done it
    node = Node("sim_test_probe")
    tf = _tf_buffer(node)

    first: dict[str, tuple[float, float]] = {}
    path: dict[str, float] = {}
    last: dict[str, tuple[float, float]] = {}
    peds: dict[str, tuple[float, float]] = {}
    world_state = [0]
    scans: list[LaserScan] = []

    def on_peds(msg):
        for p in msg.pedestrians:
            xy = (p.pose.position.x, p.pose.position.y)
            peds[p.name] = xy
            if p.name in last:
                path[p.name] = path.get(p.name, 0.0) + math.dist(last[p.name], xy)
            else:
                first[p.name] = xy
            last[p.name] = xy

    node.create_subscription(Pedestrians, _topic(node, "/arena_peds"), on_peds, 10)
    node.create_subscription(AgentStates, _topic(node, "/world_state"), lambda m: world_state.__setitem__(0, world_state[0] + 1), 10)
    node.create_subscription(LaserScan, _topic(node, f"/{ROBOT}/lidar"), lambda m: (scans.clear(), scans.append(m)), QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))

    seen: list[tuple[str, float, int]] = []
    t0 = time.time()
    while time.time() - t0 < 30.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if scans and peds and int((time.time() - t0) * 2) % 2 == 0:
            s = scans[-1]
            try:
                t = tf.lookup_transform("map", s.header.frame_id, rclpy.time.Time())
            except Exception:
                continue
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            rx, ry = t.transform.translation.x, t.transform.translation.y
            pts = [(rx + d * math.cos(yaw + s.angle_min + i * s.angle_increment), ry + d * math.sin(yaw + s.angle_min + i * s.angle_increment)) for i, d in enumerate(s.ranges) if s.range_min < d < s.range_max]
            for name, (px, py) in peds.items():
                dist = math.hypot(px - rx, py - ry)
                if 0.8 < dist < 4.0:
                    seen.append((name, dist, sum(1 for x, y in pts if math.hypot(x - px, y - py) < 0.45)))
    node.destroy_node()

    assert len(peds) >= 2, "no pedestrians published"
    # talkers (behaviour trees without a baked `walk` route) and designed standers may stand
    root = WS / "src/Arena/arena_simulation_setup/worlds" / WORLD / "scenarios" / SCENARIO
    case = _case(root)

    def stands(agent: dict) -> bool:
        ref = str((agent.get("agent") or {}).get("agent_type", ""))
        return ref.startswith("./") and "walk" not in ((yaml.safe_load((root / ref).read_text()) or {}).get("sequences") or {})

    trees = {str(a["name"]) for a in case["dynamic"] if stands(a)}
    walkers = [n for n in peds if n.rsplit("/", 1)[-1] not in trees and not re.search(r"edge_\d+$|d_\d+$", n)]
    moving = [n for n in walkers if path.get(n, 0.0) > 2.0]
    assert len(moving) >= max(1, len(walkers) // 2), f"population stands: {[(n, round(path.get(n, 0), 1)) for n in walkers]}"
    assert world_state[0] >= 20, f"HumanSim got {world_state[0]} world_state messages in 30 s: the robot's pose is not fed back"
    assert seen, "no pedestrian passed within 0.8-4 m of the lidar during the sample"
    hit = [h for _, _, h in seen if h >= 3]
    assert len(hit) >= len(seen) // 3, f"the lidar returns nothing from pedestrians: {seen[:12]}"


def test_the_episode_ends_with_a_score_row(episode) -> None:
    finished = _wait(r"EPISODE FINISHED #2", 300.0, since=episode["at"])
    assert finished, "the single leg did not finish within 300 s"
    rows = Path(f"/tmp/sim_test_rec/{SCENARIO}/scores.jsonl")
    deadline = time.time() + 20
    while time.time() < deadline and not rows.exists():
        time.sleep(1)
    assert rows.exists() and rows.read_text().strip(), "no score row written for the finished episode"
    tail = _log()[episode["at"] :]
    assert "Traceback" not in tail.replace("KeyboardInterrupt", ""), "a traceback during the episode"


#: Metres the agents an encounter or a formation owns must come to the robot, as in the hand-baked demo scenes.
CASE_REACH_M = 1.5


# last: it may watch for two minutes, and the tests before it time themselves on the first episode's onsets
def test_the_cases_own_agents_come_to_the_robot(episode) -> None:
    import json

    import rclpy
    from arena_humansim_msgs.msg import AgentFrame, AgentMeta, AgentStates
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile

    root = WS / "src/Arena/arena_simulation_setup/worlds" / WORLD / "scenarios" / SCENARIO
    record = json.loads((root / "record.json").read_text()) if (root / "record.json").is_file() else {}
    by_step = {step: {str(n) for n in (record.get("owned") or {}).get(step) or []} for step in ("C", "D")}
    by_step = {step: names for step, names in by_step.items() if names}
    owned = set().union(*by_step.values()) if by_step else set()
    if not owned:
        pytest.skip("no encounter or formation in this case")
    if not rclpy.ok():
        rclpy.init()
    node = Node("sim_test_reach_probe")
    # the robot as Arena feeds it to HumanSim and the pedestrians HumanSim reports, both in the engine's frame
    names: dict[int, str] = {}
    robot: list[tuple[float, float]] = []
    closest: dict[str, float] = {}

    def on_world(msg: AgentStates) -> None:
        robot[:] = [(a.pose.x, a.pose.y) for a in msg.agents if a.name.rsplit("/", 1)[-1] == ROBOT][:1]

    def on_frame(msg: AgentFrame) -> None:
        for aid, x, y in zip(msg.agent_id, msg.x, msg.y, strict=True):
            name = names.get(aid, "")
            if robot and name in owned:
                closest[name] = min(closest.get(name, math.inf), math.dist((x, y), robot[0]))

    latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(AgentMeta, _topic(node, "/agent_meta", "arena_humansim_msgs/msg/AgentMeta"), lambda m: names.update((aid, n.rsplit("/", 1)[-1]) for aid, n in zip(m.agent_id, m.name, strict=True)), latched)
    node.create_subscription(AgentFrame, _topic(node, "/agent_states", "arena_humansim_msgs/msg/AgentFrame"), on_frame, 10)
    node.create_subscription(AgentStates, _topic(node, "/world_state"), on_world, 10)
    t0 = time.time()

    def reach() -> dict[str, float]:
        return {step: min((closest[n] for n in names if n in closest), default=math.inf) for step, names in by_step.items()}

    while time.time() - t0 < 120.0 and not all(d < CASE_REACH_M for d in reach().values()):
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    (Path("/tmp/sim_test_rec") / SCENARIO).mkdir(parents=True, exist_ok=True)
    (Path("/tmp/sim_test_rec") / SCENARIO / "reach.json").write_text(json.dumps({"reach": reach(), "closest": closest}))
    assert closest, f"none of the case's agents {sorted(owned)} appeared beside the robot on /agent_states"
    # each encounter (C) and each formation (D) on its own: one close agent must not stand in for another step
    assert all(d < CASE_REACH_M for d in reach().values()), f"a step's agents stayed away from the robot: {reach()} ({closest})"
