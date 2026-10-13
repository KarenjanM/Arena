from __future__ import annotations

import asyncio
import math
import os
import typing

import numpy as np
import pytest
import yaml

OFFICE = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]
LAB = [(10.0, 0.0), (16.0, 0.0), (16.0, 4.0), (10.0, 4.0)]
OFFICE_TILES = [(1.0, 1.0), (3.0, 1.0), (5.0, 1.0), (1.0, 3.0), (3.0, 3.0), (5.0, 3.0)]
OUTER_BOX = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0), (0.0, 0.0)]


class Bench:
    def __init__(self, node, ctx, tmp_path) -> None:
        self.node = node
        self.ctx = ctx
        self.tmp_path = tmp_path
        self.configs = 0

    def world(self, *, zones: dict[str, list[tuple[float, float]]] | None = None, walls: typing.Sequence[typing.Sequence[tuple[float, float]]] = ()) -> None:
        from arena_rclpy_mixins.Time import Time
        from arena_simulation_setup.tree.World import Level, World, Zone
        from arena_simulation_setup.tree.World.model import WallPath

        from task_generator.manager.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
        from task_generator.shared import Position

        level = Level(
            zones=[Zone(name=name, polygon=[Position(x, y) for x, y in polygon]) for name, polygon in (zones or {}).items()],
            walls=[WallPath(path=[Position(x, y) for x, y in path]) for path in walls],
        )
        grid = np.full((60, 60), WorldOccupancy.EMPTY, dtype=np.uint8)
        world_map = WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=Position(x=0.0, y=0.0), resolution=0.5, time=Time())
        self.ctx.world_manager.update_world(world_map, World.from_levels(level))

    def mode(self, groups: list[dict] | None = None):
        from task_generator.constants import Constants
        from task_generator.tasks.registry import OBSTACLES_MODES

        key = Constants.TaskMode.TM_Obstacles.ENVIRONMENT
        tm = OBSTACLES_MODES.get(key)(node=self.node, ctx=self.ctx, namespace=OBSTACLES_MODES.meta(key).namespace)
        if groups is not None:
            self.load(tm, groups)
        return tm

    def load(self, tm, groups: list[dict]) -> None:
        from rclpy.parameter import Parameter

        self.configs += 1
        path = self.tmp_path / f"environment_{self.configs}.yaml"
        path.write_text(yaml.safe_dump({"groups": groups}))
        assert self.node.set_parameters([Parameter(tm._config.name, value=str(path))])[0].successful


@pytest.fixture
def on_bench(tmp_path):
    def run(body: typing.Callable[[Bench], typing.Awaitable[None]]) -> None:
        from arena_runtime.sim.dummy_simulator import DummySimulator

        from task_generator.constants import Constants
        from task_generator.manager.environment_manager import EnvironmentManager
        from task_generator.manager.realizer import Realizer
        from task_generator.manager.robot_manager import RobotsManager
        from task_generator.manager.world_manager.world_manager_ros import WorldManagerROS
        from task_generator.node import TaskGenerator
        from task_generator.simulators.acoustics import AcousticsSimulatorRegistry
        from task_generator.simulators.human.noop import NoopHumanSimulator
        from task_generator.tasks.context import TaskContext

        async def main() -> None:
            node = TaskGenerator()
            try:
                realizer = Realizer(level_origins={"0": (0.0, 0.0)})
                sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
                acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
                human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
                env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
                world = WorldManagerROS(node=node, environment_manager=env)
                ctx = TaskContext(environment_manager=env, robots_manager=RobotsManager(node=node, environment_manager=env), world_manager=world, abort_episode=lambda reason: None)
                await body(Bench(node, ctx, tmp_path))
            finally:
                node._heartbeat_stop.set()
                node.destroy_node()

        asyncio.run(main())

    return run


def _xy(entity) -> tuple[float, float]:
    return entity.pose.position.x, entity.pose.position.y


def _heading(entity) -> tuple[float, float]:
    yaw = entity.pose.orientation.to_yaw()
    return math.cos(yaw), math.sin(yaw)


def _inside(polygon, x: float, y: float) -> bool:
    import shapely

    return shapely.Polygon(polygon).covers(shapely.Point(x, y))


def test_group_tiles_a_zone_on_a_size_plus_margin_grid_with_rotated_offsets(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([{"name": "desk", "size": [1, 1], "margin": 1, "rotations": [90], "entities": {"static": [{"position": [1.0, 0.0, 90.0], "model": "chair"}, {"position": [0.0, 0.0, 0.0], "model": "table"}]}}])
        static, dynamic = await tm.reset(seed=0)
        assert dynamic == []
        assert [entity.name for entity in static] == [name for n in range(6) for name in (f"G_desk_0_{n}_chair_0", f"G_desk_0_{n}_table_1")]
        assert [entity.model.name for entity in static] == ["chair", "table"] * 6
        chairs, tables = static[0::2], static[1::2]
        assert [_xy(table) for table in tables] == pytest.approx(OFFICE_TILES)
        assert [_xy(chair) for chair in chairs] == pytest.approx([(x, y + 1.0) for x, y in OFFICE_TILES])
        assert all(_heading(chair) == pytest.approx((-1.0, 0.0), abs=1e-9) for chair in chairs)
        assert all(_heading(table) == pytest.approx((0.0, 1.0), abs=1e-9) for table in tables)

    on_bench(body)


def test_group_without_margin_or_rotations_defaults_to_half_meter_and_no_turn(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([{"name": "crate", "size": [2.5, 1.5], "entities": {"static": [{"position": [0.0, 0.5, 0.0], "model": "box"}]}}])
        static, _ = await tm.reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx([(1.5, 1.5), (4.5, 1.5), (1.5, 3.5), (4.5, 3.5)])
        assert all(_heading(entity) == pytest.approx((1.0, 0.0), abs=1e-9) for entity in static)

    on_bench(body)


def test_dynamic_entities_resolve_zone_and_coordinate_waypoints(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([{"name": "walk", "size": [4, 2], "margin": 0, "rotations": [0], "entities": {"dynamic": [{"position": [0.5, 0.0, 0.0], "model": "human", "waypoints": ["office", [2.0, 3.0]]}]}}])
        static, dynamic = await tm.reset(seed=0)
        assert static == []
        assert [entity.name for entity in dynamic] == ["G_walk_0_0_human_0", "G_walk_0_1_human_0"]
        assert [_xy(entity) for entity in dynamic] == pytest.approx([(2.5, 1.0), (2.5, 3.0)])
        for entity in dynamic:
            sampled, fixed = entity.waypoints
            assert _inside(OFFICE, sampled.x, sampled.y)
            assert (fixed.x, fixed.y) == (2.0, 3.0)

    on_bench(body)


def test_dynamic_entities_without_waypoints_stand_still(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([{"name": "idle", "size": [4, 2], "margin": 0, "entities": {"dynamic": [{"position": [0.0, 0.0, 0.0], "model": "human"}]}}])
        _, dynamic = await tm.reset(seed=0)
        assert [entity.waypoints for entity in dynamic] == [[], []]

    on_bench(body)


def test_zone_restricted_group_only_lands_in_its_zones(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE, "lab": LAB})
        tm = bench.mode([{"name": "bench", "size": [1, 1], "margin": 1, "zones": ["lab"], "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "stool"}]}}])
        static, _ = await tm.reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx([(x + 10.0, y) for x, y in OFFICE_TILES])
        assert {entity.name.split("_")[2] for entity in static} == {"0"}

    on_bench(body)


def test_tiles_outside_a_concave_zone_are_skipped(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"ell": [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 6.0), (0.0, 6.0)]})
        tm = bench.mode([{"name": "box", "size": [1, 1], "margin": 1, "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "box"}]}}])
        static, _ = await tm.reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx([(1.0, 1.0), (3.0, 1.0), (5.0, 1.0), (1.0, 3.0), (1.0, 5.0)])

    on_bench(body)


def test_without_zones_rooms_come_from_the_walls_and_skip_the_world_box(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[OUTER_BOX, [(2.0, 2.0), (6.0, 2.0), (6.0, 5.0), (2.0, 5.0), (2.0, 2.0)]])
        tm = bench.mode([{"name": "box", "size": [1, 1], "margin": 1, "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "box"}]}}])
        static, _ = await tm.reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx([(3.0, 3.0), (5.0, 3.0)])

    on_bench(body)


def test_reloading_the_file_parameter_replaces_the_obstacles(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([{"name": "box", "size": [1, 1], "margin": 1, "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "box"}]}}])
        assert len((await tm.reset(seed=0))[0]) == 6
        bench.load(tm, [{"name": "big", "size": [4, 2], "margin": 0, "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "box"}]}}])
        assert [_xy(entity) for entity in (await tm.reset(seed=1))[0]] == pytest.approx([(2.0, 1.0), (2.0, 3.0)])

    on_bench(body)


def test_world_bounds_span_every_wall_end(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[[(1.0, -2.0), (4.0, 3.0)], [(-1.0, 5.0), (7.0, 5.0)]])
        assert bench.mode().calculate_world_bounds() == (-1.0, 7.0, -2.0, 5.0)
        bench.world()
        assert bench.mode().calculate_world_bounds() == (math.inf, -math.inf, math.inf, -math.inf)

    on_bench(body)


def test_merge_walls_joins_collinear_segments_across_door_gaps(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[[(0.0, 0.0), (3.0, 0.0)], [(8.0, 0.0), (4.0, 0.0)], [(20.0, 0.0), (25.0, 0.0)], [(0.0, 0.0), (0.0, 2.0)], [(0.0, 9.0), (0.0, 5.0)], [(0.0, 0.0), (3.0, 3.0)]])
        tm = bench.mode()
        assert tm.merge_walls(2.0) == ([(0.0, 0.0, 8.0), (0.0, 20.0, 25.0)], [(0.0, 0.0, 2.0), (0.0, 5.0, 9.0)])
        assert tm.merge_walls([4.0]) == ([(0.0, 0.0, 8.0), (0.0, 20.0, 25.0)], [(0.0, 0.0, 9.0)])
        assert tm.merge_walls([]) == tm.merge_walls(2.0)

    on_bench(body)


def test_room_search_needs_a_covering_top_wall_and_both_side_walls(on_bench):
    async def body(bench: Bench):
        bench.world()
        tm = bench.mode()
        horizontal = [(0.0, 0.0, 20.0), (10.0, 0.0, 20.0), (2.0, 2.0, 6.0), (5.0, 2.0, 6.0)]
        vertical = [(0.0, 0.0, 10.0), (20.0, 10.0, 0.0), (2.0, 2.0, 3.0), (2.0, 4.0, 5.0), (6.0, 2.0, 5.0)]
        rooms = tm.find_room_polygons(horizontal, vertical, door_threshold=2.0)
        assert rooms == [[(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)], [(2.0, 2.0), (6.0, 2.0), (6.0, 5.0), (2.0, 5.0)]]
        assert tm.find_room_polygons(horizontal, [v for v in vertical if v[0] != 6.0], door_threshold=2.0) == rooms[:1]
        assert tm.filter_world_rooms(rooms, 0.0, 20.0, 0.0, 10.0) == rooms[1:]
        assert tm.filter_world_bounds(rooms, 0.0, 20.0, 0.0, 10.0) == rooms[1:]
        assert tm.find_room_polygons([*horizontal, (5.0, 1.0, 7.0)], vertical, door_threshold=2.0) == rooms

    on_bench(body)


def test_vertical_coverage_tolerates_gaps_up_to_the_door_width(on_bench):
    async def body(bench: Bench):
        bench.world()
        tm = bench.mode()
        assert tm.check_vertical_coverage([(2.5, 5.0), (0.0, 2.0)], 0.0, 5.0, 1.0)
        assert not tm.check_vertical_coverage([(0.0, 2.0), (4.0, 5.0)], 0.0, 5.0, 1.0)
        assert not tm.check_vertical_coverage([(0.0, 2.0), (2.0, 3.0)], 0.0, 5.0, 1.0)
        assert not tm.check_vertical_coverage([], 0.0, 5.0, 1.0)
        assert tm.check_vertical_coverage([], 5.0, 5.0, 1.0)
        assert not tm.check_vertical_coverage([(3.0, 9.0)], 0.0, 5.0, 2.0)

    on_bench(body)


def test_room_helpers_compare_and_filter_rectangles(on_bench):
    async def body(bench: Bench):
        bench.world()
        tm = bench.mode()
        room = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        assert tm.is_duplicate(list(reversed(room)), [room])
        assert not tm.is_duplicate(room, [[(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]])
        assert tm.filter_rooms([room, [(0.0, 0.0), (2.0, 0.0), (2.0, 5.0), (0.0, 5.0)], [(0.0, 0.0), (0.0, 0.0), (0.0, 20.0), (0.0, 20.0)]]) == [room]
        assert tm._is_room_inside([(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)], room)
        assert not tm._is_room_inside([(1.0, 1.0), (5.0, 1.0), (5.0, 2.0), (1.0, 2.0)], room)
        assert tm.find_matching_vertical({2.0: [(0.0, 5.0), (6.0, 9.0)]}, 2.0, 7.0) == [(6.0, 9.0)]
        assert tm.find_matching_vertical({}, 2.0, 7.0) == []
        assert tm.has_matching_top_wall({5.0: [(1.0, 7.0)]}, 0.0, 8.0, 5.0)
        assert not tm.has_matching_top_wall({5.0: [(1.0, 7.0)]}, 2.0, 6.0, 5.0)

    on_bench(body)


def test_region_checks_index_the_grid_in_whole_meters(on_bench):
    async def body(bench: Bench):
        bench.world()
        tm = bench.mode()
        grid = np.zeros((10, 10), dtype=np.uint8)
        assert tm._is_region_free(grid, 5.0, 5.0, 2.0, 2.0)
        assert not tm._is_region_free(grid, 1.0, 5.0, 2.0, 2.0, margin=1.0)
        assert not tm._is_region_free(grid, 5.0, 9.0, 2.0, 2.0)
        tm._mark_region_occupied(grid, 5.0, 5.0, 2.0, 2.0)
        assert np.argwhere(grid == 1).tolist() == [[4, 4], [4, 5], [5, 4], [5, 5]]
        assert not tm._is_region_free(grid, 5.0, 5.0, 2.0, 2.0)
        tm._mark_region_occupied(grid, 0.0, 0.0, 2.0, 2.0)
        assert int(grid.sum()) == 4

    on_bench(body)


def test_room_visualization_draws_into_test_png(on_bench, tmp_path):
    pytest.importorskip("matplotlib")

    async def body(bench: Bench):
        from task_generator.shared import Position, Wall

        bench.world()
        tm = bench.mode()
        walls = [Wall(start=Position(0.0, 0.0), end=Position(4.0, 0.0))]
        room = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            tm.visualize_rooms(walls, [room], {"x_min": 0.0, "x_max": 4.0, "y_min": 0.0, "y_max": 3.0})
            assert (tmp_path / "test.png").read_bytes().startswith(b"\x89PNG")
            (tmp_path / "test.png").unlink()
            tm.visualize_rooms(walls, [room])
            assert (tmp_path / "test.png").is_file()
        finally:
            os.chdir(cwd)

    on_bench(body)


@pytest.mark.xfail(strict=True, reason="environment obstacle mode writes its placements into the world map occupancy")
def test_parsing_an_environment_leaves_the_world_map_untouched(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        before = bench.ctx.world_manager.map_for_floor().occupancy.grid.copy()
        tm = bench.mode([{"name": "box", "size": [1, 1], "margin": 1, "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "box"}]}}])
        assert len((await tm.reset(seed=0))[0]) == 6
        after = bench.ctx.world_manager.map_for_floor().occupancy.grid
        assert sorted({int(value) for value in after[after != before]}) == []

    on_bench(body)


@pytest.mark.xfail(strict=True, reason="environment obstacle mode finds no room whose bottom wall is shared with a wider wall")
def test_wall_rooms_split_by_a_divider_are_both_found(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[OUTER_BOX, [(10.0, 0.0), (10.0, 10.0)]])
        rooms = bench.mode()._create_rooms_from_walls()
        assert sorted(map(sorted, rooms)) == [[(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0)], [(10.0, 0.0), (10.0, 10.0), (20.0, 0.0), (20.0, 10.0)]]

    on_bench(body)
