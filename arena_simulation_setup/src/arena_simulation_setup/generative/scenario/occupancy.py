"""The level occupancy grid: the level rendered at the runtime map's resolution with world furniture stamped on.
An unannotated object is the box `WorldManager` stamps for it."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .formats import is_world

if TYPE_CHECKING:
    import numpy as np
    import shapely

    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World.model import Level, World
    from arena_simulation_setup.tree.World.view import WorldView

RESOLUTION = 0.05
UNKNOWN_FOOTPRINT_RADIUS = 1.0


def level_of(world: World) -> Level | None:
    """The level the generator works on: the first one with zones, else the first."""
    return next((level for level in world.levels.values() if level.zones), next(iter(world.levels.values()), None))


def _placed(corners: list[tuple[float, float]], obj: Obstacle) -> shapely.Polygon:
    """`corners` in the object's frame placed at its pose, as `Obstacle.footprint` places them."""
    import shapely
    import shapely.affinity

    poly = shapely.affinity.rotate(shapely.Polygon(corners), obj.pose.orientation.to_yaw(), origin=(0, 0), use_radians=True)
    return shapely.affinity.translate(poly, xoff=obj.pose.position.x, yoff=obj.pose.position.y)


def _footprints(view: WorldView, world: World, objects: list[Obstacle]) -> list[shapely.Polygon | None]:
    """Each object's footprint with the world's own and linked assets resolvable, as the runtime does when it loads a world, reading each asset once."""
    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.World.view import world_assets

    async def collect() -> list[shapely.Polygon | None]:
        bounds: dict[Path, list[tuple[float, float]] | None] = {}
        out: list[shapely.Polygon | None] = []
        for obj in objects:
            try:
                path = await obj.model.resolve_path()
                if path not in bounds:
                    bounds[path] = obj.model.load(path).bounds
            except FileNotFoundError:
                out.append(None)
                continue
            corners = bounds[path]
            out.append(None if corners is None else _placed(corners, obj))
        return out

    with world_assets(view, world):
        return Identifier._run_sync(collect())


def static_polygon(obj: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    return obj.get("polygon")


def load_world(world_root: str | Path) -> World:
    """The world at `world_root` as `WorldView` loads it."""
    from arena_simulation_setup.tree.World.view import WorldView

    return WorldView(Path(world_root).resolve()).load()


def world_statics(world_root: str | Path, world: World | None = None) -> list[dict[str, Any]]:
    """Every object of the world's level, with its footprint `polygon` and a covering `radius`; `world` is the loaded world at `world_root` when given."""
    import shapely

    from arena_simulation_setup.tree.World.view import WorldView

    view = WorldView(Path(world_root).resolve())
    if world is None:
        world = view.load()
    level = level_of(world)
    if level is None:
        return []
    out: list[dict[str, Any]] = []
    for obj, footprint in zip(level.objects, _footprints(view, world, level.objects), strict=True):
        x, y = obj.pose.position.x, obj.pose.position.y
        if footprint is None:
            r = UNKNOWN_FOOTPRINT_RADIUS
            footprint = shapely.box(x - r, y - r, x + r, y + r)
        polygon = list(footprint.exterior.coords)[:-1]
        out.append({"name": obj.name, "model": obj.model.name, "pose": [x, y], "polygon": polygon, "radius": max(math.hypot(px - x, py - y) for px, py in polygon)})
    return out


def _stamp(obj: Mapping[str, Any]) -> shapely.Polygon | None:
    """The polygon a static is stamped as: its footprint, else the box of its radius around its pose."""
    import shapely

    poly = static_polygon(obj)
    if poly is not None:
        return shapely.Polygon(poly)
    pose = obj.get("pose")
    if not isinstance(pose, (list, tuple)) or len(pose) < 2:
        return None
    # the box, not a disc: a disc around the reception desk's origin left a goal on its edge looking free
    radius = float(obj.get("radius") or obj.get("interaction_radius") or 0.5)
    x, y = float(pose[0]), float(pose[1])
    return shapely.box(x - radius, y - radius, x + radius, y + radius)


def level_grid(world_root: str | Path, statics: Sequence[Mapping[str, Any]] = (), world: World | None = None) -> tuple[np.ndarray, float, float, float] | None:
    """The level's walls at `RESOLUTION` with `statics` stamped as (occupied, resolution, origin x, origin y); None for a directory that is no world."""
    root = Path(world_root)
    if not is_world(root):
        return None
    level = level_of(world if world is not None else load_world(root))
    if level is None:
        return None
    footprints = [(str(obj.get("name", "")), poly) for obj in statics if (poly := _stamp(obj)) is not None]
    if footprints:
        grid, origin = level.render_grid(resolution=RESOLUTION, asset_color="black", asset_name_color=None, static_objects=footprints)
    else:
        grid, origin = level.render_grid(resolution=RESOLUTION)
    return grid != 255, RESOLUTION, float(origin[0]), float(origin[1])
