from pathlib import Path

import attrs
import numpy as np
from arena_humansim.core.agents import BUILTIN_AGENTS, AgentType, SampledParams, sample_agent_type
from arena_humansim.core.agents.loader import load_agent_type_from_file
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg

from task_generator.shared import DynamicObstacle


def _is_path_agent_type(name: str) -> bool:
    return "/" in name or name.endswith(".yaml")


def resolve_agent_type_path(agent_type: str, included_from: Path | None) -> str:
    if included_from is not None and _is_path_agent_type(agent_type) and not Path(agent_type).is_absolute():
        return str((Path(included_from) / agent_type).resolve())
    return agent_type


def agent_type_def(agent_type: str) -> AgentType | None:
    """The agent type behind a yaml path or a builtin name, None when neither resolves."""
    if _is_path_agent_type(agent_type):
        path = Path(agent_type)
        return load_agent_type_from_file(path) if path.is_file() else None
    return BUILTIN_AGENTS.get(agent_type)


def path_agent_type_name(agent_type: str) -> str | None:
    """The name the engine reports for a path-based agent type, None for a builtin name or a missing file."""
    if not _is_path_agent_type(agent_type):
        return None
    definition = agent_type_def(agent_type)
    return definition.name if definition is not None else None


_WAYPOINT_MODE_MAP = {
    "repeat": WaypointsMsg.MODE_REPEAT,
    "reverse": WaypointsMsg.MODE_REVERSE,
    "once": WaypointsMsg.MODE_ONCE,
    "random": WaypointsMsg.MODE_RANDOM,
}


@attrs.define()
class ArenaHumanDynamicObstacle:
    agent_type: str = "adult"
    #: Per-agent overrides from the scenario's `agent:` block. `None` means the scenario did not
    #: say, and the agent type's own distribution decides.
    desired_velocity_min: float | None = None
    desired_velocity_max: float | None = None
    agent_radius: float | None = None
    waypoint_mode: int = WaypointsMsg.MODE_REPEAT
    _waypoints: list = attrs.Factory(list)

    @classmethod
    def from_dynamic_obstacle(cls, obs: DynamicObstacle) -> "ArenaHumanDynamicObstacle | None":
        raw = obs.agent
        if raw is None:
            return None
        if isinstance(raw, str):
            return cls(
                agent_type=cls._resolve_agent_type_path(raw, obs),
                waypoints=list(obs.waypoints),
                waypoint_mode=cls._parse_waypoint_mode(obs.waypoint_mode),
            )
        if not isinstance(raw, dict):
            return None

        vel = raw.get("desired_velocity")
        vel_min: float | None
        vel_max: float | None
        if isinstance(vel, dict):
            vel_min = float(vel.get("min", 1.0))
            vel_max = float(vel.get("max", 1.5))
        elif vel is not None:
            vel_min = vel_max = float(vel)
        else:
            vel_min = vel_max = None

        radius = raw.get("radius", raw.get("agent_radius"))

        agent_type = raw.get("agent_type", "adult")
        agent_type = cls._resolve_agent_type_path(agent_type, obs)

        return cls(
            agent_type=agent_type,
            desired_velocity_min=vel_min,
            desired_velocity_max=vel_max,
            agent_radius=None if radius is None else float(radius),
            waypoints=list(obs.waypoints),
            waypoint_mode=cls._parse_waypoint_mode(obs.waypoint_mode),
        )

    @staticmethod
    def _resolve_agent_type_path(agent_type: str, obs: DynamicObstacle) -> str:
        return resolve_agent_type_path(agent_type, obs.included_from)

    @staticmethod
    def _parse_waypoint_mode(raw: str) -> int:
        return _WAYPOINT_MODE_MAP.get(str(raw).lower(), WaypointsMsg.MODE_REPEAT)

    def sample_params(self, rng: np.random.Generator) -> SampledParams | None:
        definition = agent_type_def(self.agent_type)
        if definition is None:
            return None

        params = sample_agent_type(definition, rng)

        # Only what the scenario declared overrides the type's own sample.
        overrides: dict[str, float] = {}
        if self.desired_velocity_min is not None and self.desired_velocity_max is not None:
            overrides["desired_velocity"] = float(rng.uniform(self.desired_velocity_min, self.desired_velocity_max))
        if self.agent_radius is not None:
            overrides["agent_radius"] = float(self.agent_radius)
        return attrs.evolve(params, **overrides) if overrides else params
