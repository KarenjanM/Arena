from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.promptgen.world import WorldContext

np = pytest.importorskip("numpy")

RES = 0.1


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _write_world(root: Path, zones: dict[str, list[list[float]]], free: list[tuple[float, float, float, float]], size: tuple[float, float]) -> None:
    from PIL import Image

    (root / "map").mkdir(parents=True)
    levels = [{"id": "0", "zones": [{"name": n, "polygon": p, "floor": False} for n, p in zones.items()]}]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": levels}))
    w, h = int(size[0] / RES), int(size[1] / RES)
    px = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in free:
        px[h - int(y1 / RES) : h - int(y0 / RES), int(x0 / RES) : int(x1 / RES)] = 255
    Image.fromarray(px, mode="L").save(root / "map" / "0.png")
    (root / "map" / "0.yaml").write_text(yaml.safe_dump({"image": "0.png", "resolution": RES, "origin": [0.0, 0.0, 0.0], "occupied_thresh": 0.65, "negate": 0}))


def _robot(start: tuple[float, float], goal: tuple[float, float]) -> dict:
    return {"robots": [{"start": [start[0], start[1], 0.0], "phases": [{"goto": [goal[0], goal[1], 0.0]}]}]}


def test_homes_are_screened_against_the_planned_route(tmp_path: Path) -> None:
    root = tmp_path / "worlds" / "detour"
    zones = {"south": _box(0, 0, 10, 3), "east": _box(7, 3, 10, 10), "north": _box(0, 10, 10, 13), "middle": _box(0, 5, 5.5, 8)}
    free = [(0.3, 0.3, 9.7, 3.0), (7.3, 3.0, 9.7, 10.0), (0.3, 10.0, 9.7, 12.7), (0.3, 5.3, 5.2, 7.7)]
    _write_world(root, zones, free, (10.0, 13.0))
    world = WorldContext.load(root, _robot((1.5, 1.5), (1.5, 11.5)))
    assert max(x for x, _y in world.route) > 7.0
    assert "east" not in world.homes
    assert "middle" in world.homes


def test_resolve_searches_outward_for_a_valid_point(tmp_path: Path) -> None:
    root = tmp_path / "worlds" / "open"
    _write_world(root, {"room": _box(0, 0, 10, 10)}, [(0.3, 0.3, 9.7, 9.7)], (10.0, 10.0))
    world = WorldContext.load(root, _robot((1.0, 1.0), (9.0, 9.0)))
    assert world.valid((5.0, 5.0)) is not None
    pt, why = world.resolve((5.0, 5.0), within=2.0)
    assert world.valid(pt) is None and why.startswith("moved")
    assert 1.5 <= math.dist(pt, (5.0, 5.0)) <= 1.65
