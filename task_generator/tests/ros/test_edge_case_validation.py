"""The occupancy predicates the scenario and edge_case modes place agents with."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _map(grid: np.ndarray | None = None):
    from arena_rclpy_mixins.Time import Time
    from task_generator.manager.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from task_generator.shared import Position

    grid = np.full((20, 20), WorldOccupancy.EMPTY, dtype=np.uint8) if grid is None else grid
    return WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=Position(x=0.0, y=0.0), resolution=1.0, time=Time())


def _grid() -> np.ndarray:
    from task_generator.manager.world_manager.utils import WorldOccupancy

    return np.full((20, 20), WorldOccupancy.EMPTY, dtype=np.uint8)


def test_no_predicate_when_safe_dist_is_zero() -> None:
    from task_generator.tasks.obstacles._validation import make_is_valid

    assert make_is_valid(_map(), 0.0) is None
    assert make_is_valid(_map(), -1.0) is None


def test_free_space_is_valid() -> None:
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_valid

    assert make_is_valid(_map(), 1.0)(Position(10.0, 10.0)) is True


def test_occupied_cell_is_invalid() -> None:
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_valid

    grid = _grid()
    grid[9:12, 9:12] = WorldOccupancy.FULL
    is_valid = make_is_valid(_map(grid), 1.0)
    assert is_valid(Position(10.0, 10.0)) is False
    assert is_valid(Position(3.0, 3.0)) is True


def test_safe_dist_widens_the_footprint() -> None:
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_valid

    grid = _grid()
    grid[17:20, 0:3] = WorldOccupancy.FULL  # the bottom-left corner, x and y 0..3
    assert make_is_valid(_map(grid), 1.0)(Position(6.0, 6.0)) is True
    assert make_is_valid(_map(grid), 4.0)(Position(6.0, 6.0)) is False


def test_out_of_bounds_is_invalid() -> None:
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_valid

    assert make_is_valid(_map(), 1.0)(Position(500.0, 500.0)) is False


def test_in_map_bounds_accepts_a_point_inside_the_grid() -> None:
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import in_map_bounds

    assert in_map_bounds(_map(), Position(10.0, 10.0)) is True


def test_in_map_bounds_rejects_a_point_off_the_grid() -> None:
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import in_map_bounds

    assert in_map_bounds(_map(), Position(400.0, 400.0)) is False
    assert in_map_bounds(_map(), Position(-5.0, 10.0)) is False


def test_in_map_bounds_is_not_navigability() -> None:
    from task_generator.manager.world_manager.utils import WorldOccupancy
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import in_map_bounds, make_is_valid

    grid = _grid()
    grid[:] = WorldOccupancy.FULL
    assert in_map_bounds(_map(grid), Position(10.0, 10.0)) is True
    assert make_is_valid(_map(grid), 1.0)(Position(10.0, 10.0)) is False


def test_traversability_ignores_the_forbidden_layer() -> None:
    from task_generator.shared import Position
    from task_generator.tasks.obstacles._validation import make_is_traversable, make_is_valid

    world_map = _map()
    world_map.occupancy.forbidden_occupy((9, 9), (11, 11))
    assert make_is_valid(world_map, 1.0)(Position(10.0, 10.0)) is False, "nothing may spawn there"
    assert make_is_traversable(world_map, 1.0)(Position(10.0, 10.0)) is True, "the robot may still drive there"
