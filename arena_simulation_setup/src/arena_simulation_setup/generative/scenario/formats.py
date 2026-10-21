"""Scenario and world files: worlds in the working shape, scenarios as format 2 mappings checked by the scenario model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .zones import Role, stated_roles

if TYPE_CHECKING:
    from arena_humansim.core.agents import AgentType

    from arena_simulation_setup.tree.World.model import Level, World

SCENARIO_ORDER = ("objects", "humans", "sounds", "lights", "robots", "regions", "timeline", "conditions")


def read_scenario(path: Path) -> dict[str, Any]:
    """The format 2 mapping of a scenario file without its `format` key; a format 1 file is upgraded on read."""
    from arena_simulation_setup.tree.World.Scenario import upgrade_scenario

    raw = yaml.safe_load(path.read_text()) or {}
    if "format" not in raw:
        raw, _warnings = upgrade_scenario(raw)
    doc = dict(raw)
    doc.pop("format", None)
    return doc


def _agent_type(ref: str, out: Path) -> AgentType | None:
    """The agent type a scenario names, scenario-local files resolved against `out`; None when it names no file or builtin."""
    from arena_humansim.core.agents import BUILTIN_AGENTS
    from arena_humansim.core.agents.loader import load_agent_type_from_file

    candidate = out / ref[2:] if ref.startswith("./") else Path(ref)
    if candidate.is_file():
        return load_agent_type_from_file(candidate)
    if candidate.suffix == ".yaml":
        return None
    return BUILTIN_AGENTS.get(ref)


def _with_radius(agents: list[Any], out: Path) -> list[Any]:
    """Each agent block with its type's mean radius pinned, scenario-local types resolved against `out`."""
    pinned = []
    for agent in agents:
        block = agent.get("agent") if isinstance(agent, dict) else None
        if isinstance(block, dict) and "radius" not in block:
            definition = _agent_type(str(block.get("agent_type") or "adult"), out)
            radius = definition.agent_radius.mean if definition is not None else None
            if radius:
                agent = {**agent, "agent": {**block, "radius": radius}}
        pinned.append(agent)
    return pinned


def check_scenario(scenario: Mapping[str, Any], world_root: Path) -> None:
    """Parse a format 2 scenario mapping against the world's zones as the runtime loads it."""
    import numpy as np

    from arena_simulation_setup.tree.World.Scenario import parse_scenario
    from arena_simulation_setup.tree.World.view import WorldView

    view = WorldView(Path(world_root).resolve())
    world = view.load()
    origins = view.level_origins() or {level_id: (0.0, 0.0) for level_id in world.levels}
    parse_scenario(dict(scenario), world.compact_world(origins).zone_converter(np.random.default_rng(0)))


def scenario_text(doc: Mapping[str, Any], *, world_root: Path, types_dir: Path, header: str = "") -> str:
    """`doc` as format 2 scenario text, scenario-local types resolved against `types_dir`; raises on a key format 2 does not know or a scenario the model rejects."""
    from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT
    from arena_simulation_setup.tree.World.write import to_yaml

    unknown = sorted(set(doc) - set(SCENARIO_ORDER))
    if unknown:
        raise ValueError(f"format 2 does not know {unknown}")
    work = dict(doc)
    if isinstance(work.get("humans"), list):
        work["humans"] = _with_radius(work["humans"], types_dir)
    scenario = {"format": SCENARIO_FORMAT, **{key: work[key] for key in SCENARIO_ORDER if work.get(key)}}
    try:
        check_scenario(scenario, world_root)
    except (ValueError, TypeError) as exc:
        raise ValueError(str(exc)) from exc
    return header + to_yaml(scenario)


def _world(root: Path) -> World:
    from arena_simulation_setup.tree.World.view import WorldView

    return WorldView(root.resolve()).load()


def _working_level(level: Level) -> dict[str, Any]:
    import shapely

    zones = [{"name": zone.name, "corners": [{"x": p.x, "y": p.y} for p in zone.polygon], "entities": {"static": []}} for zone in level.zones]
    shapes = [shapely.Polygon([(c["x"], c["y"]) for c in z["corners"]]) if len(z["corners"]) >= 3 else None for z in zones]
    for obj in level.objects:
        x, y = obj.pose.position.x, obj.pose.position.y
        home = next((z for z, shape in zip(zones, shapes, strict=True) if shape is not None and shape.covers(shapely.Point(x, y))), None)
        if home is not None:
            home["entities"]["static"].append({"name": obj.name, "model": obj.model.name, "pose": [x, y]})
    return {"id": level.id, "zones": zones}


def levels(root: Path, world: World | None = None) -> list[dict[str, Any]]:
    """The world's levels with zones as working dicts; `world` is the loaded world at `root` when given."""
    return [_working_level(level) for level in (world if world is not None else _world(root)).levels.values() if level.zones]


def first_level(root: Path, world: World | None = None) -> dict[str, Any] | None:
    return next(iter(levels(root, world)), None)


def world_roles(root: Path, world: World | None = None) -> dict[str, Role]:
    """Zone name to role as the world's KG states it, linked worlds' facts included; `world` is the loaded world at `root` when given."""
    return stated_roles((world if world is not None else _world(root)).kg)


def is_world(root: Path) -> bool:
    return (root / "world.yaml").is_file() or any(root.glob("*/world.yaml"))
