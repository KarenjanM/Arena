"""Grid route planning on a level occupancy grid, and arc-length helpers on the planned polyline."""

from __future__ import annotations

import heapq
import math
import typing
from collections.abc import Callable, Iterable, Sequence

import attrs

if typing.TYPE_CHECKING:
    import numpy as np

Point = tuple[float, float]
Cell = tuple[int, int]

IsValid = Callable[[Point], bool]

_NEIGHBORS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
)

_MAX_EXPANSIONS = 200_000

_LOS_STEP_M = 0.2

_ENDPOINT_SEARCH_CELLS = 40


class PathError(RuntimeError):
    pass


@attrs.define
class OccupancyGrid:
    """A north-up boolean occupancy grid (row 0 is the top edge) with its resolution and the map-frame origin of its bottom-left corner."""

    occupied: typing.Any
    res: float
    ox: float
    oy: float
    _clear_cache: dict[float, typing.Any] = attrs.Factory(dict)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.occupied.shape)

    def cell(self, pt: Point) -> Cell:
        rows = self.shape[0]
        return rows - 1 - int((pt[1] - self.oy) / self.res), int((pt[0] - self.ox) / self.res)

    def xy(self, cell: Cell) -> Point:
        """The center of a cell, so `cell(xy(c)) == c` regardless of float rounding."""
        rows = self.shape[0]
        return ((cell[1] + 0.5) * self.res + self.ox, (rows - 1 - cell[0] + 0.5) * self.res + self.oy)

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def blocked_within(self, radius: float) -> np.ndarray:
        """Cells with an occupied cell in the square of half-width `radius` around them, the window the runtime validates a spawn in."""
        import numpy as np
        import scipy.ndimage

        key = round(radius, 3)
        if key not in self._clear_cache:
            k = max(0, int(round(radius / self.res)))
            self._clear_cache[key] = scipy.ndimage.binary_dilation(self.occupied, structure=np.ones((2 * k + 1, 2 * k + 1), dtype=bool)) if k else np.asarray(self.occupied, dtype=bool)
        return self._clear_cache[key]

    def clear_of(self, radius: float) -> IsValid | None:
        """A predicate true where a point is on the grid with no occupied cell within `radius` square; None for a non-positive radius."""
        if radius <= 0:
            return None
        blocked = self.blocked_within(radius)

        def is_valid(pt: Point) -> bool:
            cell = self.cell(pt)
            return self.inside(cell) and not bool(blocked[cell])

        return is_valid


class _Planner:
    """Memoised: A* revisits cells constantly."""

    def __init__(self, grid: OccupancyGrid, is_valid: IsValid | None) -> None:
        self._grid = grid
        self._is_valid = is_valid
        self.rows, self.cols = grid.shape
        self._cache: dict[Cell, bool] = {}

    def to_cell(self, pt: Point) -> Cell:
        return self._grid.cell(pt)

    def to_point(self, cell: Cell) -> Point:
        return self._grid.xy(cell)

    def in_bounds(self, cell: Cell) -> bool:
        return self._grid.inside(cell)

    def free(self, cell: Cell) -> bool:
        if not self.in_bounds(cell):
            return False
        cached = self._cache.get(cell)
        if cached is None:
            cached = True if self._is_valid is None else bool(self._is_valid(self.to_point(cell)))
            self._cache[cell] = cached
        return cached

    def nearest_free(self, cell: Cell, *, limit: int = _ENDPOINT_SEARCH_CELLS) -> Cell | None:
        if self.free(cell):
            return cell
        for radius in range(1, limit + 1):
            best: Cell | None = None
            best_d = math.inf
            for dr in range(-radius, radius + 1):
                for dc in (-radius, radius) if abs(dr) != radius else range(-radius, radius + 1):
                    candidate = (cell[0] + dr, cell[1] + dc)
                    if not self.free(candidate):
                        continue
                    d = math.dist(candidate, cell)
                    if d < best_d:
                        best, best_d = candidate, d
            if best is not None:
                return best
        return None


def _reconstruct(came_from: dict[Cell, Cell], cell: Cell) -> list[Cell]:
    path = [cell]
    while cell in came_from:
        cell = came_from[cell]
        path.append(cell)
    path.reverse()
    return path


def _astar(grid: _Planner, start: Cell, goal: Cell) -> list[Cell]:
    if start == goal:
        return [start]

    open_heap: list[tuple[float, float, Cell]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from: dict[Cell, Cell] = {}
    best: dict[Cell, float] = {start: 0.0}
    seen: set[Cell] = set()
    expansions = 0

    def h(cell: Cell) -> float:
        return math.dist(cell, goal)

    while open_heap:
        _f, g, cell = heapq.heappop(open_heap)
        if cell == goal:
            return _reconstruct(came_from, cell)
        if cell in seen:
            continue
        seen.add(cell)

        expansions += 1
        if expansions > _MAX_EXPANSIONS:
            raise PathError(f"path search exceeded {_MAX_EXPANSIONS} expansions between {start} and {goal}")

        for dr, dc, step in _NEIGHBORS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in seen or not grid.in_bounds(nxt):
                continue
            if nxt != goal and not grid.free(nxt):
                continue
            if dr and dc and not (grid.free((cell[0] + dr, cell[1])) and grid.free((cell[0], cell[1] + dc))):
                continue
            tentative = g + step
            if tentative < best.get(nxt, math.inf):
                best[nxt] = tentative
                came_from[nxt] = cell
                heapq.heappush(open_heap, (tentative + h(nxt), tentative, nxt))

    free_seen = sum(1 for c in seen if grid.free(c))
    raise PathError(f"no free-space route between cells {start} and {goal} on a {grid.rows}x{grid.cols} grid ({expansions} expansion(s), {free_seen} free cell(s) reachable from the start). The two are in disconnected free space.")


def line_of_sight(a: Point, b: Point, is_valid: IsValid | None, *, step: float = _LOS_STEP_M) -> bool:
    if is_valid is None:
        return True
    length = math.dist(a, b)
    steps = max(1, int(math.ceil(length / max(step, 1e-6))))
    for i in range(steps + 1):
        f = i / steps
        if not is_valid((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)):
            return False
    return True


def simplify(points: Sequence[Point], is_valid: IsValid | None) -> list[Point]:
    if len(points) <= 2:
        return list(points)

    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not line_of_sight(points[i], points[j], is_valid):
            j -= 1
        out.append(points[j])
        i = j

    deduped = [out[0]]
    for pt in out[1:]:
        if math.dist(pt, deduped[-1]) > 1e-9:
            deduped.append(pt)
    return deduped


def plan_route(
    grid: OccupancyGrid,
    start: Point,
    goal: Point,
    *,
    is_valid: IsValid | None,
) -> list[Point]:
    """Returns ``[start, goal]`` unchanged when the straight line is clear; raises PathError when no free route joins them."""
    if line_of_sight(start, goal, is_valid):
        return [start, goal]

    planner = _Planner(grid, is_valid)
    endpoints: list[Cell] = []
    for label, pt in (("start", start), ("goal", goal)):
        cell = planner.to_cell(pt)
        if not planner.in_bounds(cell):
            raise PathError(f"robot {label} ({pt[0]:.2f}, {pt[1]:.2f}) is outside the occupancy grid")
        free_cell = planner.nearest_free(cell)
        if free_cell is None:
            raise PathError(f"no free cell within {_ENDPOINT_SEARCH_CELLS} cells of the robot's {label} ({pt[0]:.2f}, {pt[1]:.2f}); it is walled in, not merely forbidden")
        endpoints.append(free_cell)

    cells = _astar(planner, endpoints[0], endpoints[1])

    points: list[Point] = [start, *(planner.to_point(c) for c in cells[1:-1]), goal]
    return simplify(points, is_valid)


def route_length(route: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(route, route[1:], strict=False))


def point_at_arc(route: Sequence[Point], distance: float) -> tuple[Point, Point]:
    if len(route) < 2:
        raise PathError("a route needs at least two points")

    remaining = max(0.0, distance)
    last_dir: Point | None = None
    for a, b in zip(route, route[1:], strict=False):
        seg = math.dist(a, b)
        if seg <= 1e-12:
            continue
        direction = ((b[0] - a[0]) / seg, (b[1] - a[1]) / seg)
        last_dir = direction
        if remaining <= seg:
            f = remaining / seg
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), direction
        remaining -= seg

    if last_dir is None:
        raise PathError("route has no non-degenerate segment")
    return route[-1], last_dir


def iter_points(route: Sequence[Point], step: float) -> Iterable[Point]:
    total = route_length(route)
    n = max(1, int(math.ceil(total / max(step, 1e-6))))
    for i in range(n + 1):
        yield point_at_arc(route, total * i / n)[0]
