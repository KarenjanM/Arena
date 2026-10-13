"""The world preflight of arena asset push on worlds that link levels, against real world folders."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

from arena_cli.asset import _world_dangling


def _crate(world: Path, name: str) -> None:
    model = world / "assets" / "Common" / "Object" / name
    model.mkdir(parents=True)
    (model / "annotation.yaml").write_text(yaml.safe_dump({"name": name, "path": f"Object/{name}"}), encoding="utf-8")


def _world(directory: Path, *levels: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": list(levels)}), encoding="utf-8")
    return directory


def _floor(level_id: str, model: str) -> dict:
    return {"id": level_id, "objects": [{"name": f"crate_{level_id}", "pose": [1.0, 1.0], "model": f"Common/{model}"}]}


def test_assets_of_a_subfolder_floor_travel_with_the_world(tmp_path: Path) -> None:
    from arena_simulation_setup.tree.World import WorldView

    root = tmp_path / "house"
    _crate(_world(root / "floors" / "ground", _floor("0", "Ground_Floor_Crate")), "Ground_Floor_Crate")
    _world(root, {"from": "./floors/ground"})
    assert _world_dangling(WorldView(root), root) == []


def test_a_link_outside_the_world_dangles(tmp_path: Path) -> None:
    from arena_simulation_setup.tree.World import WorldView

    _crate(_world(tmp_path / "neighbor", _floor("0", "Neighbor_Crate")), "Neighbor_Crate")
    root = _world(tmp_path / "house", {"from": "../neighbor"})
    assert _world_dangling(WorldView(root), root) == ["level 0 (links ../neighbor, outside the world)"]


def test_a_local_only_named_world_dangles_but_not_its_assets(tmp_path: Path) -> None:
    worlds = tmp_path / "worlds"
    _crate(_world(worlds / "preflight_neighbor", _floor("3", "Named_Neighbor_Crate")), "Named_Neighbor_Crate")
    root = _world(tmp_path / "house", {"id": "0", "from": "preflight_neighbor:3"})
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from arena_cli.asset import _world_dangling
        from arena_simulation_setup.tree.World import WorldView
        for line in _world_dangling(WorldView(Path({str(root)!r})), Path({str(root)!r})):
            print(line)
        """
    )
    env = {**os.environ, "ARENA_WORLD_PATH": str(worlds), "PYTHONPATH": os.pathsep.join([str(Path(__file__).parents[2]), os.environ.get("PYTHONPATH", "")])}
    lines = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(lines) == 1
    assert "preflight_neighbor" in lines[0] and f"only at {worlds / 'preflight_neighbor'}" in lines[0]
