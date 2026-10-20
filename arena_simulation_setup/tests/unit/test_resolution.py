from __future__ import annotations

import numpy as np
import pytest

from arena_simulation_setup.shared.entities import Waypoint
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import PointResolver, Pose, Position
from arena_simulation_setup.utils.resolution import activate_resolver, current_resolver


def _square() -> list[Position]:
    return [Position(0.0, 0.0), Position(2.0, 0.0), Position(2.0, 2.0), Position(0.0, 2.0)]


def _resolver(is_valid=None) -> PointResolver:
    polygons = {"zone_a": _square()}
    return PointResolver(
        lookup=polygons.get,
        rng=np.random.default_rng(0),
        is_valid=is_valid,
        candidates=lambda: ["zone_a", "door_1"],
    )


def _inside(p: Position) -> bool:
    return 0.0 <= p.x <= 2.0 and 0.0 <= p.y <= 2.0


# ---------------------------------------------------------------------------
# PointResolver
# ---------------------------------------------------------------------------


def test_point_resolver_samples_inside_polygon():
    pt = _resolver().resolve("zone_a")
    assert _inside(pt)


def test_point_resolver_unknown_ref_lists_candidates():
    with pytest.raises(ValueError, match="zone_a, door_1|door_1, zone_a"):
        _resolver().resolve("nope")


def test_point_resolver_respects_is_valid():
    pt = _resolver(is_valid=lambda p: p.x >= 1.0).resolve("zone_a")
    assert pt.x >= 1.0


# ---------------------------------------------------------------------------
# Context-scoped resolution in parse / from_any
# ---------------------------------------------------------------------------


def test_no_active_context_by_default():
    assert current_resolver() is None


def test_position_parse_string_requires_context():
    with pytest.raises(ValueError, match="no world resolution context active"):
        Position.parse("zone_a")


def test_position_parse_string_with_context():
    with activate_resolver(_resolver()):
        pt = Position.parse("zone_a")
    assert isinstance(pt, Position)
    assert _inside(pt)


def test_pose_parse_string_with_context():
    with activate_resolver(_resolver()):
        pose = Pose.parse("zone_a")
    assert _inside(pose.position)
    assert pose.orientation.w == pytest.approx(1.0)


def test_waypoint_from_any_string_with_context():
    with activate_resolver(_resolver()):
        wp = Waypoint.from_any("zone_a")
    assert isinstance(wp, Waypoint)
    assert _inside(wp)
    assert (wp.zone, wp.level) == ("zone_a", "")


def test_shifted_zone_waypoint_keeps_its_zone():
    with activate_resolver(_resolver()):
        wp = Waypoint.from_any("zone_a") + Position(10.0, 0.0)
    assert isinstance(wp, Waypoint)
    assert wp.zone == "zone_a"


def test_waypoint_from_any_string_requires_context():
    with pytest.raises(ValueError, match="no world resolution context active"):
        Waypoint.from_any("zone_a")


def test_parsed_entity_remembers_the_zone_its_pose_was_sampled_from():
    from arena_simulation_setup.shared import DynamicObstacle

    c = converter.copy()
    c.set_resolver(_resolver())
    zoned = c.structure({"name": "walker", "model": "human", "pose": "zone_a"}, DynamicObstacle)
    literal = c.structure({"name": "walker", "model": "human", "pose": [1.0, 1.0]}, DynamicObstacle)
    assert (zoned.placement_zone, literal.placement_zone) == ("zone_a", "")
    assert _inside(zoned.pose.position)


def test_waypoint_from_any_mapping_keeps_its_level():
    assert Waypoint.from_any({"x": 1.0, "y": 2.0, "level": "1"}) == Waypoint(x=1.0, y=2.0, z=0.0, level="1")


def test_non_string_inputs_unaffected_without_context():
    assert Position.parse([1.0, 2.0, 3.0]) == Position(1.0, 2.0, 3.0)
    assert Waypoint.from_any(Position(1.0, 2.0)) == Waypoint(x=1.0, y=2.0, z=0.0)


# ---------------------------------------------------------------------------
# cattrs converter integration
# ---------------------------------------------------------------------------


def test_converter_resolves_string_waypoints():
    c = converter.copy()
    c.set_resolver(_resolver())
    wps = c.structure(["zone_a", "zone_a"], list[Waypoint])
    assert len(wps) == 2
    assert all(isinstance(w, Waypoint) and _inside(w) for w in wps)
