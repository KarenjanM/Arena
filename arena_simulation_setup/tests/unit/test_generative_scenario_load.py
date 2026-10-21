"""What `populate` writes, `ScenarioView.load` reads: no agent dropped, every zone reference resolved inside its zone."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from arena_simulation_setup.generative.scenario.formats import first_level, read_scenario, write_scenario
from arena_simulation_setup.generative.scenario.populate import MIN_ROUTE_CLEARANCE, bt_type_source, build, robot_zones, route_points
from arena_simulation_setup.generative.scenario.zone_screen import ref_safe_zones, screen_zones
from arena_simulation_setup.tree.World import WorldIdentifier, WorldView
from arena_simulation_setup.tree.World.Scenario import Scenario

WORLD = "hospital_1"
# the tracked `normal` scenario's robot; furniture is left out, this checks the written format, not placement
ROBOT = {"start": [10.0, 0.5, 1.5708], "phases": [{"goto": [15.0, 32.2, 0.0]}]}
NAME = "pytest_room_centric"


@pytest.fixture
def world(tmp_path: Path) -> Path:
    try:
        source = Path(WorldIdentifier(WORLD).resolve_sync().path)
    except FileNotFoundError:
        pytest.skip(f"{WORLD} is not installed here")
    root = tmp_path / WORLD
    root.mkdir()
    for name in ("world.yaml", "kg.yaml"):
        if (source / name).is_file():
            shutil.copy(source / name, root / name)
    return root


@pytest.fixture
def written(world: Path) -> dict:
    zones, robot = first_level(world)["zones"], dict(ROBOT)
    verdicts = screen_zones(zones, world_root=world, route=route_points(robot, zones), min_route_clearance=MIN_ROUTE_CLEARANCE, robot_zones=robot_zones(robot))
    doc = build("normal", zones=zones, robot=robot, seed=0, homes=ref_safe_zones(verdicts))
    out = world / "scenarios" / NAME
    out.mkdir(parents=True)
    shutil.copy(bt_type_source(), out / bt_type_source().name)
    return read_scenario(write_scenario(out, doc, world_root=world))


@pytest.fixture
def loaded(world: Path, written: dict) -> Scenario:
    view = WorldView(world)
    level = view.load()
    origins = view.level_origins() or {level_id: (0.0, 0.0) for level_id in level.levels}
    converter = level.compact_world(origins).zone_converter(np.random.default_rng(0))
    return view.scenario(NAME).resolve_sync().load(converter=converter)


def _polygons(world: Path) -> dict[str, list[tuple[float, float]]]:
    return {z.name: [(c.x, c.y) for c in z.polygon] for level in WorldView(world).load().levels.values() for z in level.zones}


def test_every_generated_agent_survives_the_load(loaded: Scenario, written: dict) -> None:
    assert len(loaded.humans) == len(written["humans"])
    assert {a.name for a in loaded.humans} == {a["name"] for a in written["humans"]}


def test_a_zone_reference_resolves_to_a_point_inside_that_zone(loaded: Scenario, written: dict, world: Path) -> None:
    referenced = {a["name"]: a["pose"] for a in written["humans"] if isinstance(a["pose"], str)}
    if not referenced:
        pytest.skip("no zone screened clean in this world")
    polygons = _polygons(world)
    for agent in loaded.humans:
        zone = referenced.get(agent.name)
        if zone is None:
            continue
        assert _inside((agent.pose.position.x, agent.pose.position.y), polygons[zone]), f"{agent.name} resolved outside {zone}"


def test_every_waypoint_reference_resolves_to_its_own_zone(loaded: Scenario, written: dict, world: Path) -> None:
    by_name = {a["name"]: a for a in written["humans"]}
    polygons = _polygons(world)
    for agent in loaded.humans:
        names = by_name[agent.name]["waypoints"]
        assert len(agent.waypoints) == len(names)
        for waypoint, zone in zip(agent.waypoints, names, strict=True):
            assert isinstance(zone, str), f"{agent.name} carries a coordinate waypoint"
            assert _inside((waypoint.x, waypoint.y), polygons[zone]), f"{agent.name}: {zone} resolved outside"


def _inside(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = pt
    hit = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            hit = not hit
    return hit
