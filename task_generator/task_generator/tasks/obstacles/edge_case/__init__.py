"""Registered by ``ArenaHumanSimulator._register_task_modes``: under ``human:=dummy``/``isaac`` this mode would silently do nothing."""

from arena_rclpy_mixins.declarations import declare_bool, declare_double, declare_enum, declare_string
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.tasks.registry import _REGISTRY_NAMESPACE

NS = _REGISTRY_NAMESPACE("edge_case")

ROBOT_SPEED_MPS = 1.0


def declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    declare_double(
        node,
        ns("robot_speed"),
        ROBOT_SPEED_MPS,
        label="Robot design speed",
        description="Metres per second the robot is assumed to drive when interceptions, holds and object events are timed.",
    )
    declare_bool(
        node,
        ns("read_scenario_block"),
        True,
        label="Read scenario block",
        description="Read the active scenario's `edge_case.yaml` at every reset. Off runs the base population alone, which is the control arm.",
    )
    declare_string(
        node,
        ns("prompt"),
        "",
        label="Prompt",
        description=(
            "A natural-language situation. When set, the scenario is generated from it "
            "at reset - the same generator as `evaluation generate promptgen`, cached by prompt text - written beside the "
            "world's scenarios as `<base>__rviz_<hash>` and run instead of `task.scenario.file`. Empty runs "
            "`task.scenario.file` as it is."
        ),
    )
    declare_string(
        node,
        ns("prompt_base"),
        "",
        label="Prompt base scenario",
        description="Base scenario the prompt is realised on. Empty uses `task.scenario.file` (or its base when that is itself a prompt scenario).",
    )
    declare_string(
        node,
        ns("record_dir"),
        "",
        label="Record dir",
        description="Where cases.jsonl and scores.jsonl are written. Empty uses $ARENA_DATA_DIR/edge_case.",
    )

    declare_string(
        node,
        ns("objects"),
        "",
        label="Object timeline",
        description=("Path to a YAML timeline of static objects to spawn and despawn while the episode runs. Empty runs the block's own `objects:` entry, or none. A hand-authored companion to the pedestrian effects; the prompt pipeline does not write these."),
    )
    declare_double(
        node,
        ns("objects_rate_hz"),
        5.0,
        label="Object timeline rate",
        description="How often the timeline is checked, Hz. Sim time, so it is gated while the simulator is paused.",
    )

    declare_bool(
        node,
        ns("score"),
        True,
        label="Score episodes",
        description="Measure each episode's criticality (min-TTC, clearance, PET, freeze, intrusion) into scores.jsonl. Off creates no node, subscription or timer.",
    )
    declare_double(
        node,
        ns("score_rate_hz"),
        10.0,
        label="Score rate",
        description="Sampling rate for the criticality panel, Hz. Runs on sim time.",
    )
    declare_enum(
        node,
        ns("score_scope"),
        "auto",
        choices=["auto", "all", "injected"],
        label="Score scope",
        description="Which pedestrians the pairwise metrics cover: 'auto' (the case's own agents - injected ones and the encounter and formation agents its block owns - when it has any, everyone otherwise), 'all', or 'injected' (the case's own agents).",
    )
