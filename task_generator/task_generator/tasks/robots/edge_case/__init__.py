"""``edge_case`` robot task mode; registered at import since, unlike the obstacle mode, it needs no HumanSim."""

import typing

from arena_rclpy_mixins.declarations import declare_double, declare_enum, declare_int
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.constants import Constants
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

# own namespace: a leaf-name collision with the obstacle mode's `task.edge_case` would only surface at node start
_NS = _REGISTRY_NAMESPACE("edge_case_robot")


def declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    declare_int(
        node,
        ns("traversals"),
        1,
        label="Traversals",
        description="Round trips through the population per episode. N traversals is 2N-1 legs (out, back, out, ...). 1 (the default: a single start->goal leg, what the cases are designed against) behaves exactly like tm_robots:=scenario.",
    )
    declare_int(
        node,
        ns("retries"),
        1,
        label="Retries per leg",
        description="How many times each leg is submitted in a row. nav2 gives up on a leg it cannot make progress on; with retries > 1 the same goal is attempted again before the route moves on, which keeps the robot in the crowd for longer. 1 is one attempt per leg.",
    )
    declare_double(
        node,
        ns("hold"),
        0.0,
        label="Hold before the first leg",
        description="Sim seconds the robot waits at its start before the first leg. Lets the population walk and the case switch on (onsets are 8-10 s) before the robot drives into it; recordings use 15-20. 0 = drive at once.",
    )
    declare_double(
        node,
        ns("accept_within"),
        0.6,
        label="Accept a leg within",
        description="Goal tolerance of every leg, when wider than task.episode.goto_pose.tolerance.radius. nav2 refuses to plan into a goal cell inside the robot's inflation; without this the robot stood 0.23 m from such a goal for a whole recording. 0 = the robot's own tolerance.",
    )
    declare_double(
        node,
        ns("blocked_distance"),
        0.25,
        label="Progress threshold",
        description="Metres the robot must get closer to its current goal for the obstruction watchdog to count it as progress.",
    )
    declare_double(
        node,
        ns("blocked_timeout"),
        15.0,
        label="Blocked timeout",
        description="Seconds without progress before the robot counts as obstructed.",
    )
    declare_enum(
        node,
        ns("on_blocked"),
        "abort",
        choices=["abort", "continue"],
        label="On blocked",
        description="'abort' ends the episode as FAILED with the reason; 'continue' logs it and keeps going.",
    )


@ROBOTS_MODES.register(Constants.TaskMode.TM_Robots.EDGE_CASE, namespace=_NS, schema=declare_schema)
def _load_edge_case() -> type["TM_Robots"]:
    from .impl import TM_EdgeCase

    return TM_EdgeCase


__all__ = ["declare_schema"]
