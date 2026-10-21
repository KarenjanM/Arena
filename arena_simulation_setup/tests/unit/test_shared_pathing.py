from __future__ import annotations

import math

import numpy as np
import pytest

from arena_simulation_setup.shared.pathing import OccupancyGrid, PathError, iter_points, plan_route, point_at_arc, route_length


def _grid(blocked: np.ndarray) -> OccupancyGrid:
    # grids below are written with row r at y = r; a grid's row 0 is its top edge
    return OccupancyGrid(occupied=np.flipud(blocked).copy(), res=1.0, ox=0.0, oy=0.0)


def _open(rows: int = 24, cols: int = 24) -> np.ndarray:
    return np.zeros((rows, cols), dtype=bool)


def _wall_with_doorway() -> np.ndarray:
    blocked = _open()
    blocked[9:12, :] = True
    blocked[9:12, 14:19] = False
    return blocked


def _plan(blocked: np.ndarray, start: tuple[float, float], goal: tuple[float, float], clearance: float = 1.0) -> list[tuple[float, float]]:
    grid = _grid(blocked)
    return plan_route(grid, start, goal, is_valid=grid.clear_of(clearance))


def test_clear_line_is_returned_unchanged() -> None:
    assert _plan(_open(), (2.0, 2.0), (8.0, 2.0)) == [(2.0, 2.0), (8.0, 2.0)]


def test_route_bends_through_the_doorway() -> None:
    route = _plan(_wall_with_doorway(), (5.0, 5.0), (5.0, 15.0))

    assert route[0] == (5.0, 5.0)
    assert route[-1] == (5.0, 15.0)
    assert len(route) > 2
    assert max(p[0] for p in route) > 12.0, "must detour toward the doorway"
    assert route_length(route) > math.dist((5.0, 5.0), (5.0, 15.0))


def test_every_interior_point_of_a_planned_route_is_navigable() -> None:
    grid = _grid(_wall_with_doorway())
    is_valid = grid.clear_of(1.0)
    route = plan_route(grid, (5.0, 5.0), (5.0, 15.0), is_valid=is_valid)

    interior = [p for p in iter_points(route, 0.25) if math.dist(p, route[0]) > 1.5 and math.dist(p, route[-1]) > 1.5]
    assert interior
    assert all(is_valid(p) for p in interior)


def test_unreachable_goal_raises_rather_than_returning_a_fake_route() -> None:
    blocked = _open()
    blocked[9:12, :] = True
    with pytest.raises(PathError, match="no free-space route"):
        _plan(blocked, (5.0, 5.0), (5.0, 15.0))


def test_off_map_endpoint_is_named() -> None:
    with pytest.raises(PathError, match="outside the occupancy grid"):
        _plan(_wall_with_doorway(), (5.0, 5.0), (500.0, 500.0))


def test_no_predicate_means_the_straight_line() -> None:
    assert plan_route(_grid(_wall_with_doorway()), (5.0, 5.0), (5.0, 15.0), is_valid=None) == [(5.0, 5.0), (5.0, 15.0)]


def test_simplify_collapses_the_staircase() -> None:
    route = _plan(_wall_with_doorway(), (5.0, 5.0), (5.0, 15.0))
    assert len(route) <= 8, f"expected a handful of waypoints, got {len(route)}"


def test_forbidden_endpoint_discs_do_not_trap_the_search() -> None:
    blocked = _open()
    blocked[4:7, 4:7] = True
    blocked[16:19, 4:7] = True

    route = _plan(blocked, (5.0, 5.0), (5.0, 17.0))
    assert route[0] == (5.0, 5.0)
    assert route[-1] == (5.0, 17.0)


def test_exemption_does_not_tunnel_through_a_real_wall() -> None:
    blocked = _open()
    blocked[9:12, :] = True
    with pytest.raises(PathError, match="no free-space route"):
        _plan(blocked, (5.0, 2.0), (5.0, 20.0))


def test_clearance_is_the_runtime_square_window() -> None:
    blocked = _open()
    blocked[10, 10] = True
    grid = _grid(blocked)
    is_valid = grid.clear_of(1.0)
    assert not is_valid((11.5, 11.5)) and not is_valid((9.5, 9.5))
    assert is_valid((12.5, 10.5)) and is_valid((10.5, 12.5))
    assert not is_valid((-1.0, 5.0))
    assert grid.clear_of(0.0) is None


def test_cell_and_centre_round_trip() -> None:
    grid = OccupancyGrid(occupied=np.zeros((40, 60), dtype=bool), res=0.05, ox=-1.0, oy=2.0)
    for cell in ((0, 0), (39, 59), (12, 31)):
        assert grid.cell(grid.xy(cell)) == cell
    assert grid.cell((-1.0 + 0.01, 2.0 + 0.01)) == (39, 0)


def test_route_length_sums_the_segments() -> None:
    assert route_length([(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]) == pytest.approx(7.0)


def test_point_at_arc_walks_the_polyline_and_reports_local_heading() -> None:
    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]

    pt, heading = point_at_arc(route, 5.0)
    assert pt == pytest.approx((5.0, 0.0))
    assert heading == pytest.approx((1.0, 0.0))

    pt, heading = point_at_arc(route, 15.0)
    assert pt == pytest.approx((10.0, 5.0))
    assert heading == pytest.approx((0.0, 1.0)), "heading must follow the segment it is on"


def test_point_at_arc_clamps_past_the_end() -> None:
    assert point_at_arc([(0.0, 0.0), (10.0, 0.0)], 999.0)[0] == pytest.approx((10.0, 0.0))


def test_iter_points_covers_the_whole_route() -> None:
    pts = list(iter_points([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], 1.0))
    assert pts[0] == pytest.approx((0.0, 0.0))
    assert pts[-1] == pytest.approx((10.0, 10.0))
    assert len(pts) >= 20
