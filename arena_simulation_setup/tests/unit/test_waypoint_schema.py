"""Scenario waypoint entries: positions, zone/elevator refs, level and dwell."""

from __future__ import annotations

import numpy as np
import pytest
from arena_simulation_setup.shared.entities import DynamicObstacle, Waypoint
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import PointResolver, Position


def _resolver() -> PointResolver:
    polygons = {
        "lobby": [Position(0.0, 0.0), Position(2.0, 0.0), Position(2.0, 2.0), Position(0.0, 2.0)],
        "elev_a": [Position(10.0, 0.0), Position(12.0, 0.0), Position(12.0, 2.0), Position(10.0, 2.0)],
    }
    return PointResolver(lookup=polygons.get, rng=np.random.default_rng(0), candidates=lambda: list(polygons))


def _structure(raw: list) -> list[Waypoint]:
    c = converter.copy()
    c.set_resolver(_resolver())
    return c.structure(raw, list[Waypoint])


def test_plain_forms_keep_defaults() -> None:
    wps = _structure([[1, 2], {"x": 3, "y": 4}, [5, 6, 0.5]])
    assert [(w.x, w.y, w.z) for w in wps] == [(1.0, 2.0, 0.0), (3.0, 4.0, 0.0), (5.0, 6.0, 0.5)]
    assert all(w.dwell == 0.0 and w.elevator is None and w.level == "" for w in wps)


def test_dict_forms_carry_level_and_dwell() -> None:
    a, b, c = _structure(
        [
            {"x": 1, "y": 2, "dwell": 3},
            {"x": 4, "y": 5, "level": "1", "dwell": 2.5},
            {"zone": "lobby", "level": 2},
        ]
    )
    assert (a.x, a.y, a.dwell, a.level) == (1.0, 2.0, 3.0, "")
    assert (b.x, b.y, b.dwell, b.level) == (4.0, 5.0, 2.5, "1")
    assert 0.0 <= c.x <= 2.0 and 0.0 <= c.y <= 2.0 and c.level == "2" and c.zone == "lobby"


def test_zone_waypoint_may_dwell() -> None:
    (wp,) = _structure([{"zone": "lobby", "dwell": 1}])
    assert 0.0 <= wp.x <= 2.0 and wp.dwell == 1.0 and wp.zone == "lobby"


def test_elevator_waypoint_lands_in_the_cabin() -> None:
    plain, prefixed = _structure([{"elevator": "elev_a"}, {"elevator": "1.elev_a", "dwell": 4}])
    for wp in (plain, prefixed):
        assert 10.0 <= wp.x <= 12.0 and 0.0 <= wp.y <= 2.0
    assert plain.elevator == "elev_a" and prefixed.elevator == "1.elev_a" and prefixed.dwell == 4.0


@pytest.mark.parametrize(
    "raw",
    [
        {"x": 1, "y": 2, "pause": 3},
        {"position": [1, 2]},
        {"x": 1, "y": 2, "level_id": "1"},
        {"dwell": 3},
        {"x": 1, "y": 2, "dwell": -1},
    ],
)
def test_bad_dicts_are_rejected(raw: dict) -> None:
    with pytest.raises((ValueError, ExceptionGroup)):
        _structure([raw])


def test_shift_keeps_metadata() -> None:
    wp = Waypoint(x=1.0, y=1.0, z=0.0, level="1", zone="lobby", dwell=2.0, elevator="elev_a")
    for moved in (wp + Position(1.0, 2.0), Position(1.0, 2.0) + wp):
        assert (moved.x, moved.y) == (2.0, 3.0)
        assert (moved.level, moved.zone, moved.dwell, moved.elevator) == ("1", "lobby", 2.0, "elev_a")


def test_dynamic_obstacle_carries_spawn_delay_and_alias() -> None:
    c = converter.copy()
    c.set_resolver(_resolver())
    obs = c.structure(
        {"name": "p", "model": "arenian", "pose": [0, 0, 0], "spawn_delay": 4, "semantic_alias": {"elevator": "elev_a"}, "waypoints": [{"x": 1, "y": 2, "dwell": 3}, {"elevator": "elev_a"}]},
        DynamicObstacle,
    )
    assert obs.waypoints[0].dwell == 3.0
    assert obs.waypoints[1].elevator == "elev_a"
    assert obs.spawn_delay == 4.0
    assert obs.semantic_alias == {"elevator": "elev_a"}


def test_dynamic_obstacle_rejects_negative_spawn_delay() -> None:
    with pytest.raises((ValueError, ExceptionGroup)):
        converter.structure({"name": "p", "model": "arenian", "pose": [0, 0, 0], "spawn_delay": -1}, DynamicObstacle)
