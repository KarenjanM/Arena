"""Per-level directory shape, selective loading, WorldIdentifier.split_filter."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.shared.world import Elevator
from arena_simulation_setup.tree.World import FORMAT, Level, World, WorldIdentifier, WorldView, Zone
from arena_simulation_setup.utils.geometry import Position

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SQUARE = [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]]


def _minimal_zone(name: str = "room") -> Zone:
    return Zone(name=name, polygon=[Position(x, y) for x, y in _SQUARE])


def _write_level_yaml(path: Path, zone_names: list[str]) -> None:
    """Write a minimal format 1 level to *path*."""
    path.write_text(yaml.safe_dump({"zones": [{"name": name, "corners": _SQUARE} for name in zone_names]}))


def _write_world_yaml(world_dir: Path, level_ids: list[str]) -> None:
    """Write a minimal format 2 world with one zone per level to *world_dir*."""
    world_dir.mkdir(parents=True)
    levels = [{"id": level_id, "zones": [{"name": f"room_{level_id}", "polygon": _SQUARE}]} for level_id in level_ids]
    (world_dir / "world.yaml").write_text(yaml.safe_dump({"format": FORMAT, "levels": levels}))


# ---------------------------------------------------------------------------
# WorldIdentifier.split_filter
# ---------------------------------------------------------------------------


def test_parse_no_filter():
    name, filt = WorldIdentifier.split_filter("myworld")
    assert name == "myworld"
    assert filt is None


def test_parse_single_level():
    name, filt = WorldIdentifier.split_filter("myworld[1]")
    assert name == "myworld"
    assert filt == {"1"}


def test_parse_multi_level():
    name, filt = WorldIdentifier.split_filter("myworld[1,3]")
    assert name == "myworld"
    assert filt == {"1", "3"}


def test_parse_with_spaces():
    name, filt = WorldIdentifier.split_filter("myworld[ 1 , 3 ]")
    assert name == "myworld"
    assert filt == {"1", "3"}


def test_parse_empty_brackets_returns_none_filter():
    name, filt = WorldIdentifier.split_filter("myworld[]")
    assert name == "myworld"
    assert filt is None


# ---------------------------------------------------------------------------
# World.validate: elevator destinations
# ---------------------------------------------------------------------------

_BOX = [Position(0, 0), Position(10, 0), Position(10, 10), Position(0, 10)]


def _make_world_two_floors_linked() -> World:
    """Two levels (0, 1) each with one elevator, linked 0->1 and 1->0."""
    level0 = Level(id="0", zones=[Zone(name="z0", polygon=_BOX)], elevators=[Elevator(name="elev_a", position=Position(1.0, 1.0), destinations=["elev_b"])])
    level1 = Level(id="1", zones=[Zone(name="z1", polygon=_BOX)], elevators=[Elevator(name="elev_b", position=Position(1.0, 1.0), destinations=["elev_a"])])
    return World(levels={"0": level0, "1": level1})


def test_validate_linked_two_levels_passes():
    world = _make_world_two_floors_linked()
    assert world.validate() == []
    assert world.elevator_levels() == {"elev_a": "0", "elev_b": "1"}


def test_validate_dangling_dest_hard_error_when_level_loaded():
    level0 = Level(id="0", zones=[Zone(name="z0", polygon=_BOX)], elevators=[Elevator(name="elev_a", position=Position(1.0, 1.0), destinations=["nonexistent_elev"])])
    level1 = Level(id="1", zones=[Zone(name="z1", polygon=_BOX)])
    with pytest.raises(ValueError, match="nonexistent_elev"):
        World(levels={"0": level0, "1": level1}).validate()


def test_validate_dangling_dest_warn_when_level_not_loaded(caplog: pytest.LogCaptureFixture):
    level0 = Level(id="0", zones=[Zone(name="z0", polygon=_BOX)], elevators=[Elevator(name="elev_a", position=Position(1.0, 1.0), destinations=["elev_b"])])
    with caplog.at_level(logging.WARNING):
        warns = World(levels={"0": level0}).validate(loaded_level_ids={"0"})

    assert len(warns) == 1
    assert "not loaded" in warns[0]
    assert "elev_b" in warns[0]


def test_selective_load_keeps_destinations_into_dropped_levels(tmp_path: Path):
    world_dir = tmp_path / "lift"
    _write_world_yaml(world_dir, ["0", "1"])
    data = yaml.safe_load((world_dir / "world.yaml").read_text())
    data["levels"][0]["elevators"] = [{"name": "lift_0", "position": [1.0, 1.0], "destinations": ["lift_1"]}]
    data["levels"][1]["elevators"] = [{"name": "lift_1", "position": [1.0, 1.0], "destinations": ["lift_0"]}]
    (world_dir / "world.yaml").write_text(yaml.safe_dump(data))
    assert set(WorldView(world_dir).load().levels) == {"0", "1"}
    partial = WorldView(world_dir).load(level_filter={"0"})
    assert set(partial.levels) == {"0"}
    assert partial.levels["0"].elevators[0].destinations == ["lift_1"]
    assert partial.elevator_levels() == {"lift_0": "0"}
    with pytest.raises(ValueError, match="lift_1"):
        partial.validate()


# ---------------------------------------------------------------------------
# Auto-discovery: per-level subdirs (format 1)
# ---------------------------------------------------------------------------


def test_auto_discovery_two_levels(tmp_path: Path):
    world_dir = tmp_path / "myworld"
    for fid in ("0", "1"):
        fd = world_dir / fid
        fd.mkdir(parents=True)
        _write_level_yaml(fd / "world.yaml", [f"room_{fid}"])

    with pytest.warns(DeprecationWarning):
        world = WorldView(world_dir).load(validate=False)

    assert set(world.levels.keys()) == {"0", "1"}


def test_auto_discovery_zone_names(tmp_path: Path):
    world_dir = tmp_path / "myworld"
    for fid in ("0", "1"):
        fd = world_dir / fid
        fd.mkdir(parents=True)
        _write_level_yaml(fd / "world.yaml", [f"room_{fid}"])

    with pytest.warns(DeprecationWarning):
        world = WorldView(world_dir).load(validate=False)

    assert [z.name for z in world.levels["0"].zones] == ["room_0"]


# ---------------------------------------------------------------------------
# Auto-wrap legacy: single root world.yaml -> level '0'
# ---------------------------------------------------------------------------


def test_legacy_single_file_loads_as_level_0(tmp_path: Path):
    world_dir = tmp_path / "legacyworld"
    world_dir.mkdir()
    _write_level_yaml(world_dir / "world.yaml", ["hall"])

    with pytest.warns(DeprecationWarning):
        world = WorldView(world_dir).load(validate=False)

    assert list(world.levels) == ["0"]


def test_legacy_single_file_zone_preserved(tmp_path: Path):
    world_dir = tmp_path / "legacyworld"
    world_dir.mkdir()
    _write_level_yaml(world_dir / "world.yaml", ["my_hall"])

    with pytest.warns(DeprecationWarning):
        world = WorldView(world_dir).load(validate=False)

    assert [z.name for z in world.levels["0"].zones] == ["my_hall"]


# ---------------------------------------------------------------------------
# Level filter: selective load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["format1", "format2"])
def test_level_filter_returns_only_requested(tmp_path: Path, layout: str):
    world_dir = tmp_path / "multilevel"
    _write_levels(world_dir, ["1", "2", "3"], layout)
    world = WorldView(world_dir).load(validate=False, level_filter={"1"})
    assert set(world.levels.keys()) == {"1"}


@pytest.mark.parametrize("layout", ["format1", "format2"])
def test_level_filter_excludes_others(tmp_path: Path, layout: str):
    world_dir = tmp_path / "multilevel"
    _write_levels(world_dir, ["1", "2", "3"], layout)
    world = WorldView(world_dir).load(validate=False, level_filter={"1", "3"})
    assert set(world.levels) == {"1", "3"}


def _write_levels(world_dir: Path, level_ids: list[str], layout: str) -> None:
    if layout == "format2":
        _write_world_yaml(world_dir, level_ids)
        return
    for fid in level_ids:
        fd = world_dir / fid
        fd.mkdir(parents=True)
        _write_level_yaml(fd / "world.yaml", [f"room_{fid}"])


# ---------------------------------------------------------------------------
# WorldView.save shape
# ---------------------------------------------------------------------------


def test_save_writes_one_world_file_and_a_map_per_level(tmp_path: Path):
    world = World.from_levels(Level(zones=[_minimal_zone()]), Level(zones=[_minimal_zone("attic")]))
    WorldView(tmp_path / "w").save(world)
    files = sorted(str(path.relative_to(tmp_path / "w")) for path in (tmp_path / "w").rglob("*") if path.is_file())
    assert files == ["map/0.png", "map/0.yaml", "map/1.png", "map/1.yaml", "world.yaml"]
    assert yaml.safe_load((tmp_path / "w" / "world.yaml").read_text())["format"] == FORMAT
    assert yaml.safe_load(WorldView(tmp_path / "w").map_yaml("1").read_text())["image"] == "1.png"


def test_save_writes_no_per_level_dirs(tmp_path: Path):
    WorldView(tmp_path / "w").save(World.from_levels(Level(zones=[_minimal_zone()])))
    assert not (tmp_path / "w" / "0").exists()
    assert not (tmp_path / "w" / "map" / "levels").exists()
