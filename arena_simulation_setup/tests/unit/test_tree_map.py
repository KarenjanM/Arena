from __future__ import annotations

import logging

import pytest
import shapely

from arena_simulation_setup.tree.World.Map import PREVIEW_MIN_RESOLUTION, PREVIEW_PIXEL_BUDGET, Map, preview_resolution


def _simple_rooms():
    return shapely.MultiPolygon([shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])])


def _empty_doors():
    return shapely.MultiPolygon()


def _simple_walls():
    return shapely.MultiLineString([[(0, 0), (10, 0)]])


def test_generate_png_returns_bytes_and_origin():
    result = Map.generate_png(
        rooms=_simple_rooms(),
        doors=_empty_doors(),
        walls=_simple_walls(),
        resolution=0.1,
        padding=2,
    )
    png_bytes, origin = result
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    assert len(origin) == 2


def test_generate_png_asset_color_none_skips_objects():
    png_bytes, origin = Map.generate_png(
        rooms=_simple_rooms(),
        doors=_empty_doors(),
        walls=_simple_walls(),
        resolution=0.1,
        padding=2,
        asset_color=None,
    )
    assert isinstance(png_bytes, bytes)


def test_generate_png_asset_name_color_none():
    obj = shapely.Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    png_bytes, origin = Map.generate_png(
        rooms=_simple_rooms(),
        doors=_empty_doors(),
        walls=_simple_walls(),
        resolution=0.1,
        padding=2,
        static_objects=[("obj1", obj)],
        asset_color="grey",
        asset_name_color=None,
    )
    assert isinstance(png_bytes, bytes)


def test_generate_png_object_with_less_than_3_coords_warns(caplog):
    tiny_obj = shapely.Polygon([(1, 1), (1.001, 1), (1, 1)])
    with caplog.at_level(logging.WARNING):
        png_bytes, origin = Map.generate_png(
            rooms=_simple_rooms(),
            doors=_empty_doors(),
            walls=_simple_walls(),
            resolution=0.05,
            padding=2,
            static_objects=[("tiny", tiny_obj)],
            asset_color="grey",
            asset_name_color=None,
        )
    assert isinstance(png_bytes, bytes)


def test_generate_png_resolution_affects_size():
    _, o1 = Map.generate_png(_simple_rooms(), _empty_doors(), _simple_walls(), resolution=1.0, padding=0)
    _, o2 = Map.generate_png(_simple_rooms(), _empty_doors(), _simple_walls(), resolution=0.1, padding=0)
    # Origin should be same regardless of resolution (before padding)
    assert o1[0] == pytest.approx(o2[0], abs=1e-5)


def test_generate_map_yaml_contains_resolution():
    yaml_str = Map.generate_map_yaml(resolution=0.05, filename="map.png", origin=(0.0, 0.0))
    assert "0.05" in yaml_str
    assert "map.png" in yaml_str


def test_preview_resolution_fits_the_pixel_budget_and_never_goes_finer_than_the_minimum():
    assert preview_resolution(PREVIEW_PIXEL_BUDGET * 0.5) == pytest.approx(0.5)
    assert preview_resolution(1.0) == PREVIEW_MIN_RESOLUTION
    assert preview_resolution(0.0) == PREVIEW_MIN_RESOLUTION
