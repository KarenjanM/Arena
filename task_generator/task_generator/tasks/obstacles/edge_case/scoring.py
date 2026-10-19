"""Live episode sampling for criticality; a sampling failure degrades to no score, never to an aborted episode."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import arena_people_msgs.msg
import rclpy.node
from arena_rclpy_mixins.shared import Namespace
from rclpy.parameter import Parameter

from .criticality import Criticality, Point, Sample, score

if TYPE_CHECKING:
    from task_generator.manager.robot_manager.robot_manager import RobotManager

    from .provenance import CaseRecord

DEFAULT_RATE_HZ = 10.0
STALE_POSE_S = 0.5


def differenced_velocity(
    prev: tuple[float, float],
    prev_vel: tuple[float, float],
    here: tuple[float, float],
    now: float,
    moved_t: float,
) -> tuple[tuple[float, float], float]:
    if here == prev:
        return ((0.0, 0.0) if now - moved_t >= STALE_POSE_S else prev_vel), moved_t
    dt = now - moved_t
    if dt <= 1e-6:
        return prev_vel, now
    return ((here[0] - prev[0]) / dt, (here[1] - prev[1]) / dt), now


# arena_people_msgs/Pedestrian has no radius field, so every pedestrian is scored as this nominal body
DEFAULT_PED_RADIUS = 0.3

INJECT_PREFIX = "edge_"


class EpisodeScorer(rclpy.node.Node):
    def __init__(
        self,
        robot_manager: RobotManager,
        *,
        rate_hz: float = DEFAULT_RATE_HZ,
        peds_topic: str | None = None,
    ) -> None:
        super().__init__(
            "edge_case_scorer",
            namespace=str(robot_manager.namespace),
            # sim time gates the tick while paused; no global args so the task generator's remaps skip this node
            use_global_arguments=False,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self._rm = robot_manager
        self._samples: list[Sample] = []
        self._saw_peds = False
        self._peds_warned = False
        self._moved_t = 0.0
        self._peds: dict[str, tuple[tuple[float, float], tuple[float, float], float]] = {}
        self._last: tuple[float, float] | None = None

        # pedestrians are published on the env namespace, one level above the robot manager's
        env_ns = Namespace(robot_manager.node.get_namespace())
        self._peds_topic = peds_topic or str(env_ns("arena_peds"))
        self._peds_sub = self.create_subscription(arena_people_msgs.msg.Pedestrians, self._peds_topic, self._on_peds, 10)
        self._peds_retry_t: float | None = None
        self._timer = self.create_timer(1.0 / max(rate_hz, 1e-3), self._tick)

    @staticmethod
    def _roster_from_msg(msg: arena_people_msgs.msg.Pedestrians) -> dict[str, tuple[tuple[float, float], tuple[float, float], float]]:
        peds = {}
        for p in msg.pedestrians:
            peds[p.name] = (
                (p.pose.position.x, p.pose.position.y),
                (p.twist.linear.x, p.twist.linear.y),
                DEFAULT_PED_RADIUS,
            )
        return peds

    def _on_peds(self, msg: arena_people_msgs.msg.Pedestrians) -> None:
        self._peds = self._roster_from_msg(msg)

    def _tick(self) -> None:
        pose = self._rm.pose
        if pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        here = (pose.position.x, pose.position.y)

        vel = (0.0, 0.0)
        if self._samples:
            prev = self._samples[-1]
            vel, self._moved_t = differenced_velocity(prev.robot, prev.robot_vel, here, now, self._moved_t)
        else:
            self._moved_t = now

        self._samples.append(
            Sample(
                t=now,
                robot=here,
                robot_vel=vel,
                robot_radius=float(self._rm.radius),
                peds=dict(self._peds),
            )
        )
        if not self._saw_peds and self._peds:
            self._saw_peds = True
            if self._peds_warned:
                self.get_logger().warning(f"edge_case scorer: arena_peds healed after {now - self._samples[0].t:.1f} s of silence ({len(self._peds)} peds)")
        if not self._saw_peds and now - self._samples[0].t > 10.0 and (self._peds_retry_t is None or now - self._peds_retry_t >= 10.0):
            self._peds_retry_t = now
            self.destroy_subscription(self._peds_sub)
            self._peds_sub = self.create_subscription(arena_people_msgs.msg.Pedestrians, self._peds_topic, self._on_peds, 10)
            self.get_logger().warning(f"edge_case scorer: arena_peds silent for {now - self._samples[0].t:.0f} s - recreating the subscription")
        if not self._saw_peds and not self._peds_warned and now - self._samples[0].t > 15.0:
            self._peds_warned = True
            self.get_logger().warning("edge_case scorer: no pedestrians on arena_peds after 15 s of sampling - this episode will be unscorable")

    @property
    def samples(self) -> list[Sample]:
        return self._samples

    def reset(self) -> None:
        self._samples.clear()
        self._peds.clear()
        self._saw_peds = False
        self._peds_warned = False

    def shutdown(self) -> None:
        self._timer.cancel()
        self.destroy_node()


SCOPE_AUTO = "auto"
SCOPE_ALL = "all"
SCOPE_INJECTED = "injected"


def injected_seen(samples: Sequence[Sample], own: Collection[str] = ()) -> list[str]:
    seen: set[str] = set()
    for sample in samples:
        seen.update(name for name in sample.peds if name.rsplit("/", 1)[-1].startswith(INJECT_PREFIX) or name.rsplit("/", 1)[-1] in own)
    return sorted(seen)


def score_scope(setting: str, case: CaseRecord, own: Collection[str], samples: Sequence[Sample]) -> tuple[str, list[str], str]:
    """Which agents an episode is scored against: `all`, or the case's own agents once they appeared; the third item says why it fell back."""
    scope = (setting or SCOPE_AUTO).strip().lower()
    if scope == SCOPE_ALL or (scope == SCOPE_AUTO and not case.injected and not own):
        return SCOPE_ALL, [], ""
    seen = injected_seen(samples, own)
    if not seen:
        return SCOPE_ALL, [], "score_scope wanted the injected agents but none appeared on arena_peds; scoring all"
    return SCOPE_INJECTED, seen, ""


def score_episode(samples: Sequence[Sample], *, encounter: Point | None, agents: Sequence[str], goal: Point | None, route_length: float) -> tuple[Criticality, Criticality | None, dict[str, Any]]:
    """The episode's score, ambient score when it is scoped, and how far the robot got; with a goal, the window ends 2 s after arrival."""
    window = list(samples)
    away = None
    extra: dict[str, Any] = {}
    if goal is not None and route_length > 0 and window:

        def goal_dist(s: Sample) -> float:
            return math.hypot(s.robot[0] - goal[0], s.robot[1] - goal[1])

        # samples can predate the teleport back to the start; arrival counts only once clear of the goal
        away = next((i for i, s in enumerate(window) if goal_dist(s) > 1.2), None)
        hit = None if away is None else next((i for i in range(away, len(window)) if goal_dist(window[i]) <= 0.6), None)
        if hit is not None:
            keep_t = window[hit].t + 2.0
            window = window[: hit + 1] + [s for s in window[hit + 1 :] if s.t <= keep_t]
        if away is not None:
            nearest = min(goal_dist(s) for s in window[away:])
            extra["progress"] = round(max(0.0, min(1.0, 1.0 - nearest / route_length)), 4)
            extra["arrived"] = nearest <= 0.6
            extra["ended"] = "arrived" if nearest <= 0.6 else "cut"
    result = score(window, encounter=encounter, only=agents or None)
    ambient = score(window, encounter=encounter) if agents else None
    return result, ambient, extra


def write_score(
    record_dir: Path | str,
    case: CaseRecord,
    result: Criticality,
    *,
    scope: str = "all",
    scored_agents: Sequence[str] = (),
    ambient: Criticality | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Append one row to scores.jsonl, joined to cases.jsonl on (run_seed, episode_id)."""
    out = Path(record_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scores.jsonl"

    row = {
        "run_seed": case.run_seed,
        "episode_id": case.episode_id,
        "world": case.world,
        "seed": case.seed,
        "scenario": case.scenario,
        "case_id": case.case_id,
        "prompt_id": case.prompt_id,
        "steps": case.steps,
        "designed_ttc": case.designed_ttc,
        "min_ttc_s": result.min_ttc_s,
        "min_ttc_with": result.min_ttc_with,
        "min_ttc_at": result.min_ttc_at,
        "min_clearance_m": result.min_clearance_m,
        "min_clearance_with": result.min_clearance_with,
        "min_clearance_at": result.min_clearance_at,
        "pet_s": result.pet_s,
        "designed_pet": case.designed_pet,
        "freeze_duration_s": result.freeze_duration_s,
        "intrusion_time_s": result.intrusion_time_s,
        "collided": result.collided,
        "collision_fault": result.collision_fault or None,
        "collision_with": result.collision_with or None,
        "collision_at": result.collision_at,
        "collision_robot_closing": result.collision_robot_closing,
        "collision_ped_closing": result.collision_ped_closing,
        "observed": result.observed,
        "samples": result.samples,
        "duration_s": result.duration_s,
        "scope": scope,
        "scored_agents": list(scored_agents),
        "min_ttc_s_ambient": None if ambient is None else ambient.min_ttc_s,
        "min_clearance_m_ambient": None if ambient is None else ambient.min_clearance_m,
        "collided_ambient": None if ambient is None else ambient.collided,
        "collision_fault_ambient": None if ambient is None else (ambient.collision_fault or None),
    }
    if extra:
        # an extra must never overwrite a measured field
        row.update({k: v for k, v in extra.items() if k not in row})
    with path.open("a") as fh:
        fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True, default=str) + "\n")
    return path
