from collections.abc import Iterator, Mapping

from shapely import LineString, MultiLineString
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from arena_simulation_setup.shared import Position
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import Level, WallPath, World


def line_pairs(geom: MultiLineString | LineString | Polygon):
    """Create an iterator for all line segments of a geometry.

    Args:
        geom (MultiLineString | LineString | Polygon): The geometry to extract line segments from.

    Yields:
        tuple[Point, Point]: A tuple of start and end points for each line segment.
    """
    if geom.is_empty:
        return

    if isinstance(geom, Polygon):
        yield from line_pairs(geom.exterior)
        for interior_ring in geom.interiors:
            yield from line_pairs(interior_ring)
        return

    if isinstance(geom, LineString):
        geom = MultiLineString((geom,))

    for line in geom.geoms:
        coords = list(line.coords)
        if len(coords) >= 2:
            for start, end in zip(coords[:-1], coords[1:], strict=False):
                yield Point(start), Point(end)


def to_corners(geom: Polygon) -> list[Position]:
    """Convert the corners of a polygon to a list of Positions.

    Args:
        geom (Polygon): The polygon to convert.

    Returns:
        list[Position]: A list of Positions representing the corners of the polygon.
    """
    return [Position(x=pt[0], y=pt[1]) for pt in geom.exterior.coords]


def _lines(geom: BaseGeometry) -> Iterator[LineString]:
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom.exterior
        yield from geom.interiors
    elif isinstance(geom, LineString):
        yield geom
    elif not isinstance(geom, Point):
        for part in geom.geoms:
            yield from _lines(part)


def to_walls(geom: BaseGeometry, material: MaterialIdentifier | None = None) -> list[WallPath]:
    """Wall paths along the lines and polygon rings of a geometry, one per line, rings closed."""
    paths: list[WallPath] = []
    for line in _lines(geom):
        points = [Position(x=x, y=y) for x, y, *_ in line.coords]
        if len(points) < 2:
            continue
        closed = line.is_ring and len(points) > 3
        paths.append(WallPath(path=points[:-1] if closed else points, closed=closed, material=material))
    return paths


def world_of(level: Level, descriptions: Mapping[str, str] | None = None) -> World:
    """A one-level world of a generated level, the zone descriptions in its KG."""
    from arena_simulation_setup.kg.scene import Scene

    world = World.from_levels(level)
    described = {name: text for name, text in (descriptions or {}).items() if text}
    if described:
        world.kg = Scene()
        for name, text in described.items():
            world.kg.add_entity(f'room:{name}', {'description': text})
    return world
