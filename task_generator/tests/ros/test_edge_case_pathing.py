"""The Level C route planner that replaced the straight start-to-goal line."""

from __future__ import annotations

import math

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _map(grid):
    from arena_rclpy_mixins.Time import Time
    from task_generator.manager.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from task_generator.shared import Position

    # grids below are written with row r at y = r; a map's row 0 is its top edge
    return WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(np.flipud(grid).copy())), origin=Position(x=0.0, y=0.0), resolution=1.0, time=Time())


def _grid(rows=24, cols=24):
    from task_generator.manager.world_manager.utils import WorldOccupancy

    return np.full((rows, cols), WorldOccupancy.EMPTY, dtype=np.uint8)


def _wall_with_doorway():
    from task_generator.manager.world_manager.utils import WorldOccupancy

    grid = _grid()
    grid[9:12, :] = WorldOccupancy.FULL
    grid[9:12, 14:19] = WorldOccupancy.EMPTY
    return grid


def _is_valid(grid, safe_dist=1.0):
    from task_generator.tasks.obstacles._validation import make_is_valid

    return make_is_valid(_map(grid), safe_dist)


def test_clear_line_is_returned_unchanged() -> None:
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route

    grid = _grid()
    assert plan_route(_map(grid), (2.0, 2.0), (8.0, 2.0), is_valid=_is_valid(grid)) == [(2.0, 2.0), (8.0, 2.0)]


def test_route_bends_through_the_doorway() -> None:
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route, route_length

    grid = _wall_with_doorway()
    route = plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=_is_valid(grid))

    assert route[0] == (5.0, 5.0)
    assert route[-1] == (5.0, 15.0)
    assert len(route) > 2
    assert max(p[0] for p in route) > 12.0, "must detour toward the doorway"
    assert route_length(route) > math.dist((5.0, 5.0), (5.0, 15.0))


def test_every_interior_point_of_a_planned_route_is_navigable() -> None:
    from task_generator.shared import Position
    from task_generator.tasks.obstacles.edge_case.pathing import iter_points, plan_route

    grid = _wall_with_doorway()
    is_valid = _is_valid(grid)
    route = plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=is_valid)

    interior = [p for p in iter_points(route, 0.25) if math.dist(p, route[0]) > 1.5 and math.dist(p, route[-1]) > 1.5]
    assert interior
    assert all(is_valid(Position(x=p[0], y=p[1])) for p in interior)


def test_unreachable_goal_raises_rather_than_returning_a_fake_route() -> None:
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.tasks.obstacles.edge_case.pathing import PathError, plan_route

    grid = _grid()
    grid[9:12, :] = WorldOccupancy.FULL
    with pytest.raises(PathError, match="no free-space route"):
        plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=_is_valid(grid))


def test_off_map_endpoint_is_named() -> None:
    from task_generator.tasks.obstacles.edge_case.pathing import PathError, plan_route

    grid = _wall_with_doorway()
    with pytest.raises(PathError, match="outside the occupancy grid"):
        plan_route(_map(grid), (5.0, 5.0), (500.0, 500.0), is_valid=_is_valid(grid))


def test_no_predicate_means_the_straight_line() -> None:
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route

    grid = _wall_with_doorway()
    assert plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=None) == [(5.0, 5.0), (5.0, 15.0)]


def test_simplify_collapses_the_staircase() -> None:
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route

    grid = _wall_with_doorway()
    route = plan_route(_map(grid), (5.0, 5.0), (5.0, 15.0), is_valid=_is_valid(grid))
    assert len(route) <= 8, f"expected a handful of waypoints, got {len(route)}"


def test_route_length_sums_the_segments() -> None:
    from task_generator.tasks.obstacles.edge_case.pathing import route_length

    assert route_length([(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]) == pytest.approx(7.0)


def test_point_at_arc_walks_the_polyline_and_reports_local_heading() -> None:
    from task_generator.tasks.obstacles.edge_case.pathing import point_at_arc

    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]

    pt, heading = point_at_arc(route, 5.0)
    assert pt == pytest.approx((5.0, 0.0))
    assert heading == pytest.approx((1.0, 0.0))

    pt, heading = point_at_arc(route, 15.0)
    assert pt == pytest.approx((10.0, 5.0))
    assert heading == pytest.approx((0.0, 1.0)), "heading must follow the segment it is on"


def test_point_at_arc_clamps_past_the_end() -> None:
    from task_generator.tasks.obstacles.edge_case.pathing import point_at_arc

    route = [(0.0, 0.0), (10.0, 0.0)]
    assert point_at_arc(route, 999.0)[0] == pytest.approx((10.0, 0.0))


def test_iter_points_covers_the_whole_route() -> None:
    from task_generator.tasks.obstacles.edge_case.pathing import iter_points

    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    pts = list(iter_points(route, 1.0))
    assert pts[0] == pytest.approx((0.0, 0.0))
    assert pts[-1] == pytest.approx((10.0, 10.0))
    assert len(pts) >= 20


def test_encounter_on_a_bent_route_lands_on_the_route() -> None:
    from task_generator.tasks.obstacles.edge_case.geometry import solve_encounter

    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    enc = solve_encounter(route, robot_speed=1.0, ped_speed=1.0, encounter_at=0.75)

    assert enc.encounter == pytest.approx((10.0, 5.0))
    assert enc.t_encounter == pytest.approx(15.0)
    assert math.dist(enc.encounter, (7.5, 7.5)) > 2.0


def test_head_on_follows_the_local_heading_after_a_corner() -> None:
    from task_generator.tasks.obstacles.edge_case.geometry import solve_encounter

    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    enc = solve_encounter(route, robot_speed=1.0, ped_speed=1.0, encounter_at=0.75, approach_angle=180.0)

    assert enc.spawn[1] > enc.encounter[1]
    assert enc.spawn[0] == pytest.approx(enc.encounter[0])


def test_forbidden_endpoint_discs_do_not_trap_the_search() -> None:
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.tasks.obstacles.edge_case.pathing import plan_route

    grid = _grid()
    grid[4:7, 4:7] = WorldOccupancy.FULL
    grid[16:19, 4:7] = WorldOccupancy.FULL

    route = plan_route(_map(grid), (5.0, 5.0), (5.0, 17.0), is_valid=_is_valid(grid))
    assert route[0] == (5.0, 5.0)
    assert route[-1] == (5.0, 17.0)


def test_exemption_does_not_tunnel_through_a_real_wall() -> None:
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.tasks.obstacles.edge_case.pathing import PathError, plan_route

    grid = _grid()
    grid[9:12, :] = WorldOccupancy.FULL
    with pytest.raises(PathError, match="no free-space route"):
        plan_route(_map(grid), (5.0, 2.0), (5.0, 20.0), is_valid=_is_valid(grid))
