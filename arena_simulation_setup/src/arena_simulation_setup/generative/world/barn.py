import logging
import math
import typing

import shapely
import shapely.affinity
import yaml

from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import Level, World, Zone
from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT

from . import BaseConfiguration, WorldGeneratorImpl
from .utils import to_corners, to_walls, world_of

logger = logging.getLogger(__name__)

Cell = tuple[int, int]
_DIRS: list[Cell] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
_CARDBOARD = MaterialIdentifier('Cardboard')


class BarnBase(WorldGeneratorImpl):
    """Shared scaffolding for BARN-style generators.

    Owns the build lifecycle, the boundary walls, the scenario file and the scenario
    episode binding. Subclasses implement _build() to populate self._world and self._scenario.
    """

    SCENARIO: typing.ClassVar[str]

    config: BaseConfiguration

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)
        self._built = False
        logger.info(self.config)

    def compute(self) -> World:
        self._build()
        return self._world

    def files(self) -> dict[str, bytes]:
        self._build()
        return {f'scenarios/{self.SCENARIO}/scenario.yaml': yaml.safe_dump(self._scenario, sort_keys=False).encode('utf-8')}

    def params(self) -> dict[str, typing.Any]:
        return {'tm_robots': 'scenario', 'robots_params': {'file': self.SCENARIO}}

    def _build(self):
        raise NotImplementedError

    def _arena(self) -> tuple[shapely.Geometry, list]:
        arena = shapely.box(0, 0, self.config.width, self.config.height)
        return arena, to_walls(arena.exterior)

    def _set_level(self, name: str, arena: shapely.Geometry, walls: list, description: str):
        self._world = world_of(Level(zones=[Zone(name=name, polygon=to_corners(arena))], walls=walls), {name: description})

    def _set_scenario(self, start: tuple[float, float, float], goto: tuple[float, float, float]):
        self._scenario = {
            'format': SCENARIO_FORMAT,
            'robots': [
                {
                    'start': [round(start[0], 3), round(start[1], 3), round(start[2], 3)],
                    'phases': [{'goto': [round(goto[0], 3), round(goto[1], 3), round(goto[2], 3)]}],
                }
            ],
        }


class WorldGeneratorBarn(BarnBase):
    """BARN-style passage: one winding corridor walled with cardboard boxes at any angle.

    Inspired by the BARN Challenge (people.cs.gmu.edu/~xiao/Research/BARN_Challenge), original implementation.
    """

    SCENARIO = 'barn'

    class Configuration(BaseConfiguration):
        width: float = 10.0
        height: float = 10.0

        box_size: float = 0.4  # cardboard box footprint (m)
        box_gap: float = 0.0  # spacing between boxes in a wall (m)
        passage_width: float = 1.0  # free lane width (m)
        straightness: float = 0.5  # 0..1, lower = windier / sharper turns
        dead_ends: int = 3  # misleading spurs off the main lane
        dead_end_length: int = 2  # spur length in nav-grid steps
        margin: float = 0.5  # clear border inside the boundary walls

        @property
        def pitch(self) -> float:
            # one box row always separates parallel lane strands
            return self.passage_width + self.box_size + self.box_gap

    config: Configuration

    def _build(self):
        if self._built:
            return

        c = self.config
        usable_w = c.width - 2 * c.margin
        usable_h = c.height - 2 * c.margin
        cols = int(usable_w / c.pitch)
        rows = int(usable_h / c.pitch)
        if cols < 1 or rows < 2:
            raise ValueError(f'arena too small for a BARN passage: nav grid is {cols}x{rows}, need at least 1x2 (reduce margin/passage_width/box_size or grow width/height)')

        x0 = c.margin + (usable_w - cols * c.pitch) / 2 + c.pitch / 2
        y0 = c.margin + (usable_h - rows * c.pitch) / 2 + c.pitch / 2

        def cell_xy(cell: Cell) -> tuple[float, float]:
            return x0 + cell[0] * c.pitch, y0 + cell[1] * c.pitch

        start = (self.rng.randrange(cols), 0)
        goal = (self.rng.randrange(cols), rows - 1)
        path = self._spine(start, goal, cols, rows)

        in_path = set(path)
        spurs: list[list[Cell]] = []
        interior = path[1:-1]
        for _ in range(c.dead_ends):
            if not interior:
                break
            anchor = self.rng.choice(interior)
            spur = self._spur(anchor, in_path, cols, rows, c.dead_end_length)
            if spur:
                spurs.append([anchor, *spur])
                in_path.update(spur)

        lanes = [shapely.LineString([cell_xy(cell) for cell in path])]
        lanes += [shapely.LineString([cell_xy(cell) for cell in spur]) for spur in spurs]
        corridor = shapely.union_all(lanes).buffer(c.passage_width / 2, cap_style='square', join_style='mitre', mitre_limit=5.0)

        arena, walls = self._arena()
        walls += self._walls(corridor)

        self._set_level('barn', arena, walls, f'BARN passage, {len(path)} segments, {c.dead_ends} dead ends')

        sx, sy = cell_xy(path[0])
        nx, ny = cell_xy(path[1])
        gx, gy = cell_xy(path[-1])
        px, py = cell_xy(path[-2])
        self._set_scenario((sx, sy, math.atan2(ny - sy, nx - sx)), (gx, gy, math.atan2(gy - py, gx - px)))

        self._built = True

    def _spine(self, start: Cell, goal: Cell, cols: int, rows: int) -> list[Cell]:
        path = [start]
        in_path = {start}
        stack: list[tuple[Cell, Cell | None, list[Cell]]] = [(start, None, self._moves(start, None, goal))]

        while stack:
            cur, _, moves = stack[-1]
            if cur == goal:
                return path
            while moves:
                d = moves.pop(0)
                n = (cur[0] + d[0], cur[1] + d[1])
                if self._legal(n, in_path, cur, cols, rows):
                    path.append(n)
                    in_path.add(n)
                    stack.append((n, d, self._moves(n, d, goal)))
                    break
            else:
                stack.pop()
                in_path.discard(path.pop())

        raise ValueError('failed to carve a BARN passage; try a coarser grid or fewer constraints')

    def _spur(self, anchor: Cell, in_path: set[Cell], cols: int, rows: int, length: int) -> list[Cell]:
        cur = anchor
        last: Cell | None = None
        spur: list[Cell] = []
        for _ in range(length):
            for d in self._moves(cur, last, cur):
                n = (cur[0] + d[0], cur[1] + d[1])
                if self._legal(n, in_path | set(spur), cur, cols, rows):
                    spur.append(n)
                    cur, last = n, d
                    break
            else:
                break
        return spur

    def _legal(self, n: Cell, in_path: set[Cell], pred: Cell, cols: int, rows: int) -> bool:
        # keep strands >= 2 cells apart in all 8 directions so corridors never merge
        if not (0 <= n[0] < cols and 0 <= n[1] < rows) or n in in_path:
            return False
        return all(v == pred or max(abs(v[0] - n[0]), abs(v[1] - n[1])) != 1 for v in in_path)

    def _moves(self, cur: Cell, last: Cell | None, goal: Cell) -> list[Cell]:
        dist = max(abs(cur[0] - goal[0]), abs(cur[1] - goal[1]))
        weights: list[float] = []
        for d in _DIRS:
            w = 1.0
            if d == last:
                w += 3.0 * self.config.straightness
            if max(abs(cur[0] + d[0] - goal[0]), abs(cur[1] + d[1] - goal[1])) < dist:
                w += 1.5
            weights.append(w)

        dirs = list(_DIRS)
        order: list[Cell] = []
        while dirs:
            pick = self.rng.uniform(0, sum(weights))
            acc = 0.0
            for i, w in enumerate(weights):
                acc += w
                if pick <= acc:
                    order.append(dirs.pop(i))
                    weights.pop(i)
                    break
        return order

    def _walls(self, corridor: shapely.Geometry) -> list:
        # the corridor boundary is the load-bearing seal at exactly passage_width;
        # rotated cardboard tiles ride it for the angled physical-course look
        walls = to_walls(corridor.boundary, material=_CARDBOARD)
        walls += self._tiles(corridor)
        return walls

    def _tiles(self, corridor: shapely.Geometry) -> list:
        c = self.config
        step = c.box_size + c.box_gap
        half = c.box_size / 2
        rings = corridor.boundary
        lines = list(rings.geoms) if rings.geom_type == 'MultiLineString' else [rings]

        walls = []
        for ring in lines:
            n = max(int(ring.length / step), 1)
            for k in range(n):
                s = (k + 0.5) / n * ring.length
                p = ring.interpolate(s)
                p1 = ring.interpolate(max(s - half, 0.0))
                p2 = ring.interpolate(min(s + half, ring.length))
                ang = math.atan2(p2.y - p1.y, p2.x - p1.x)
                ox, oy = -math.sin(ang), math.cos(ang)
                if corridor.contains(shapely.Point(p.x + ox * 0.01, p.y + oy * 0.01)):
                    ox, oy = -ox, -oy
                tile = shapely.box(p.x - half, p.y - half, p.x + half, p.y + half)
                tile = shapely.affinity.rotate(tile, ang, origin=(p.x, p.y), use_radians=True)
                tile = shapely.affinity.translate(tile, ox * half, oy * half)
                tile = tile.difference(corridor)
                if not tile.is_empty:
                    walls += to_walls(tile.boundary, material=_CARDBOARD)
        return walls
