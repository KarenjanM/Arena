from __future__ import annotations

import numpy as np
from arena_simulation_setup.generative.scenario.zone_screen import _cell_centers
from arena_simulation_setup.shared.pathing import OccupancyGrid


def test_screened_points_are_the_cell_centers_the_grid_uses() -> None:
    mask = np.zeros((4, 6), dtype=bool)
    cells = [(0, 0), (3, 0), (1, 4), (3, 5)]
    for cell in cells:
        mask[cell] = True
    grid = OccupancyGrid(occupied=mask, res=0.05, ox=-1.0, oy=2.0)
    xs, ys = _cell_centers(mask, grid.res, grid.ox, grid.oy)
    rows, cols = np.nonzero(mask)
    assert [(round(x, 9), round(y, 9)) for x, y in zip(xs, ys, strict=True)] == [tuple(round(v, 9) for v in grid.xy((int(r), int(c)))) for r, c in zip(rows, cols, strict=True)]
    assert all(grid.cell(grid.xy((int(r), int(c)))) == (int(r), int(c)) for r, c in zip(rows, cols, strict=True))
