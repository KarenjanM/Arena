from __future__ import annotations

import asyncio
import io
import math

import PIL.Image
import pytest
import yaml

from arena_simulation_setup.shared.world import Door, Elevator, Schedule, Signal, Sound
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import FORMAT, Level, WallPath, World, WorldMicrophone, Zone, ZoneCeiling, parse_world, write_world
from arena_simulation_setup.tree.World.legacy import upgrade
from arena_simulation_setup.tree.World.write import dump_zone
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Position


def _make_zone(name: str = "zone", corners=None) -> Zone:
    return Zone(name=name, polygon=corners or [])


def _world(**level: object) -> World:
    return parse_world({"format": FORMAT, "levels": [{"id": "0", **level}]})


def _legacy_level(*zones: dict) -> Level:
    data, _, _ = upgrade({"0": {"zones": list(zones)}})
    return parse_world(data).levels["0"]


def _segments(walls: list) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [((wall.start.x, wall.start.y), (wall.end.x, wall.end.y)) for wall in walls]


def test_zone_floor_rectangle():
    zone = _make_zone(
        name="room",
        corners=[
            Position(0.0, 0.0),
            Position(4.0, 0.0),
            Position(4.0, 3.0),
            Position(0.0, 3.0),
        ],
    )
    floor = zone.floor_patch
    assert floor.x_length == pytest.approx(4.0)
    assert floor.y_length == pytest.approx(3.0)
    assert floor.pos.x == pytest.approx(2.0)
    assert floor.pos.y == pytest.approx(1.5)


def test_zone_floor_triangular():
    zone = _make_zone(
        name="tri",
        corners=[
            Position(0.0, 0.0),
            Position(2.0, 0.0),
            Position(1.0, 2.0),
        ],
    )
    floor = zone.floor_patch
    assert floor.x_length == pytest.approx(2.0)
    assert floor.y_length == pytest.approx(2.0)


def test_zone_floor_empty_corners_raises():
    zone = _make_zone(name="empty")
    with pytest.raises(ValueError):
        _ = zone.floor_patch


def test_all_walls_empty():
    assert Level().all_walls == []


def test_all_walls_populated():
    level = Level(walls=[WallPath(path=[Position(0, 0), Position(1, 0)])])
    assert _segments(level.all_walls) == [((0, 0), (1, 0))]


def test_all_walls_multiple_paths():
    level = Level(walls=[WallPath(path=[Position(0, 0), Position(1, 0)]), WallPath(path=[Position(2, 0), Position(3, 0)])])
    assert len(level.all_walls) == 2


# ---------------------------------------------------------------------------
# Zone semantics
# ---------------------------------------------------------------------------


def test_zone_semantics_default_omitted_on_write():
    zone = _make_zone()
    assert zone.semantics == []
    assert "semantics" not in dump_zone(zone, {})


def test_zone_semantics_primitives_round_trip():
    world = _world(
        zones=[
            {
                "name": "lobby",
                "semantics": [
                    {"state": "max_speed", "value": 1.5},
                    {"predicate": "quiet", "value": True},
                    {"predicate": "restricted", "value": False},
                ],
            }
        ]
    )
    zone = world.levels["0"].zones[0]
    assert [c.name for c in zone.semantics] == ["max_speed", "quiet", "restricted"]
    assert [c.value for c in zone.semantics] == [1.5, True, False]
    reparsed = parse_world(yaml.safe_load(write_world(world)))
    assert reparsed.levels["0"].zones[0].semantics == zone.semantics


def test_zone_semantics_author_defined_name_accepted():
    zone = _world(zones=[{"name": "z", "semantics": [{"predicate": "custom_flag"}]}]).levels["0"].zones[0]
    assert zone.semantics[0].name == "custom_flag"


def test_all_doors_empty():
    assert Level().all_doors == []


def test_all_doors_populated():
    d = Door(name="d", start=Position(0, 0), end=Position(1, 0))
    assert Level(doors=[d]).all_doors == [d]


def test_all_elevators_empty():
    assert Level().all_elevators == []


def test_all_elevators_populated():
    elevator = Elevator(name="lift", position=Position(1, 1))
    assert Level(elevators=[elevator]).all_elevators == [elevator]


# ---------------------------------------------------------------------------
# Schedule / Signal (M2 standalone semantic entities)
# ---------------------------------------------------------------------------


def test_schedule_round_trip():
    raw = {
        'name': 'fire_alarm',
        'semantics': [{'preset': 'schedule', 'params': {'windows': [], 'regime': 'alarm'}}],
    }
    schedule = converter.structure(raw, Schedule)
    assert schedule.name == 'fire_alarm'
    assert [(c.role, c.name) for c in schedule.semantics] == [
        ('state', 'state'),
        ('predicate', 'active'),
        ('state', 'window_remaining'),
    ]
    unstructured = converter.unstructure(schedule)
    reparsed = converter.structure(unstructured, Schedule)
    assert reparsed.name == schedule.name
    assert reparsed.semantics == schedule.semantics


def test_signal_round_trip():
    raw = {
        'name': 'crosswalk_light',
        'semantics': [{'preset': 'signal', 'params': {'phases': [{'name': 'go', 'duration': 5.0}]}}],
    }
    signal = converter.structure(raw, Signal)
    assert signal.name == 'crosswalk_light'
    assert [(c.role, c.name) for c in signal.semantics] == [
        ('state', 'state'),
        ('state', 'phase_remaining'),
        ('predicate', 'stop'),
    ]
    unstructured = converter.unstructure(signal)
    reparsed = converter.structure(unstructured, Signal)
    assert reparsed.name == signal.name
    assert reparsed.semantics == signal.semantics


def test_schedule_default_semantics_empty():
    schedule = converter.structure({'name': 'bare'}, Schedule)
    assert schedule.semantics == []


def test_world_schedules_and_signals_default_empty():
    world = World()
    assert world.schedules == []
    assert world.signals == []


def test_world_schedules_populated():
    world = parse_world({"format": FORMAT, "schedules": [{"name": "fire_alarm"}], "levels": [{"id": "0"}]})
    assert [schedule.name for schedule in world.schedules] == ["fire_alarm"]
    assert world.schedules == [Schedule(name="fire_alarm")]


def test_world_signals_populated():
    world = parse_world({"format": FORMAT, "signals": [{"name": "crosswalk_light"}], "levels": [{"id": "0"}]})
    assert world.signals == [Signal(name="crosswalk_light")]


# ---------------------------------------------------------------------------
# Sound (standalone audible semantic entity)
# ---------------------------------------------------------------------------


def test_sound_round_trip():
    raw = {
        'name': 'hall_siren',
        'asset_id': 'alarm_loop',
        'position': [4.0, 2.3],
        'semantics': [{'preset': 'sound', 'params': {'sound_on': 'alarm'}}],
    }
    sound = converter.structure(raw, Sound)
    assert sound.name == 'hall_siren'
    assert sound.asset_id == 'alarm_loop'
    assert sound.position.x == pytest.approx(4.0)
    assert [(c.role, c.name) for c in sound.semantics] == [
        ('predicate', 'sounding'),
        ('state', 'volume_db'),
    ]
    unstructured = converter.unstructure(sound)
    reparsed = converter.structure(unstructured, Sound)
    assert reparsed.name == sound.name
    assert reparsed.semantics == sound.semantics


@pytest.mark.parametrize(
    'raw',
    [
        {'name': '', 'asset_id': 'alarm_loop', 'position': [0.0, 0.0]},
        {'name': 'a:b', 'asset_id': 'alarm_loop', 'position': [0.0, 0.0]},
        {'name': 'siren', 'asset_id': '', 'position': [0.0, 0.0]},
        {'name': 'siren', 'asset_id': 'alarm_loop'},
        {'name': 'siren', 'asset_id': 'alarm_loop', 'position': [0.0, 0.0], 'attach': 'cabinet'},
        {'name': 'siren', 'asset_id': 'alarm_loop', 'attach': 'cabinet', 'level': '1'},
        {'name': 'siren', 'asset_id': 'alarm_loop', 'position': [0.0, 0.0], 'reference_distance_m': 0.0},
        {'name': 'siren', 'asset_id': 'alarm_loop', 'frame': 'jackal/base_link', 'position': [0.0, 0.0]},
        {'name': 'siren', 'asset_id': 'alarm_loop', 'frame': 'jackal/base_link', 'attach': 'cabinet'},
        {'name': 'siren', 'asset_id': 'alarm_loop', 'frame': 'jackal/base_link', 'level': '1'},
    ],
)
def test_sound_rejects_invalid(raw):
    with pytest.raises(Exception):
        converter.structure(raw, Sound)


def test_sound_frame_round_trip():
    raw = {
        'name': 'horn',
        'asset_id': 'alarm_loop',
        'frame': '/jackal/base_link',
        'offset': [0.3, 0.0, 0.8],
        'semantics': [{'preset': 'sound'}],
    }
    sound = converter.structure(raw, Sound)
    assert sound.frame == 'jackal/base_link'
    assert sound.position is None
    assert sound.offset.z == pytest.approx(0.8)
    reparsed = converter.structure(converter.unstructure(sound), Sound)
    assert reparsed.frame == sound.frame


def test_sound_rejects_unknown_semantics_key():
    raw = {
        'name': 'hall_siren',
        'asset_id': 'alarm_loop',
        'position': [4.0, 2.3],
        'semantics': [{'state': 'volume_db', 'bogus': 1}],
    }
    with pytest.raises(Exception):
        converter.structure(raw, Sound)


def test_level_sounds_default_empty():
    assert Level().sounds == []


def test_all_sounds_populated():
    sound = Sound(name="hall_siren", asset_id="alarm_loop", position=Position(4.0, 2.3))
    assert Level(sounds=[sound]).all_sounds == [sound]


def test_all_floors_count():
    z1 = _make_zone("z1", corners=[Position(0, 0), Position(2, 0), Position(2, 2), Position(0, 2)])
    z2 = _make_zone("z2", corners=[Position(0, 0), Position(4, 0), Position(4, 4), Position(0, 4)])
    assert len(Level(zones=[z1, z2]).all_floors) == 2


def test_all_static_entities_empty():
    assert Level(zones=[_make_zone()]).all_static_entities == []


def test_all_dynamic_entities_empty():
    assert Level(zones=[_make_zone()]).all_dynamic_entities == []


def _make_square_level(name: str = "room", size: float = 10.0) -> Level:
    zone = _make_zone(name=name, corners=[Position(0.0, 0.0), Position(size, 0.0), Position(size, size), Position(0.0, size)])
    return Level(zones=[zone], walls=[WallPath(path=[Position(0.0, 0.0), Position(size, 0.0)])])


def test_render_returns_bytes_and_origin_tuple():
    result = _make_square_level().render()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_render_bytes_is_valid_png():
    png_bytes, _ = _make_square_level().render()
    assert isinstance(png_bytes, bytes)
    img = PIL.Image.open(io.BytesIO(png_bytes))
    assert img.format == "PNG"


def test_render_image_mode_is_rgb():
    png_bytes, _ = _make_square_level().render()
    img = PIL.Image.open(io.BytesIO(png_bytes))
    assert img.mode == "RGB"


def test_render_image_size_reflects_zone_aabb():
    size = 10.0
    resolution = 0.05
    padding = 5
    png_bytes, _ = _make_square_level(size=size).render(resolution=resolution)
    img = PIL.Image.open(io.BytesIO(png_bytes))
    expected_px = math.ceil(size / resolution) + 2 * padding
    assert img.width == expected_px
    assert img.height == expected_px


def test_render_origin_reflects_padding():
    size = 10.0
    resolution = 0.05
    padding = 5
    _, origin = _make_square_level(size=size).render(resolution=resolution)
    assert origin[0] == pytest.approx(0.0 - padding * resolution)
    assert origin[1] == pytest.approx(0.0 - padding * resolution)


def _square_corners(size: float = 2.0) -> list[Position]:
    return [Position(0.0, 0.0), Position(size, 0.0), Position(size, size), Position(0.0, size)]


def test_all_floors_skips_missing_and_empty_material():
    kept = Zone(name="kept", polygon=_square_corners())
    dropped = Zone(name="dropped", polygon=_square_corners(), floor=None)
    blank = Zone(name="blank", polygon=_square_corners(), floor="")
    assert [f.name for f in Level(zones=[kept, dropped, blank]).all_floors] == ["kept"]


def test_all_ceilings_skips_missing_and_empty_material():
    kept = Zone(name="kept", polygon=_square_corners(), ceiling=ZoneCeiling(height=2.5))
    dropped = Zone(name="dropped", polygon=_square_corners(), ceiling=None)
    blank = Zone(name="blank", polygon=_square_corners(), ceiling=ZoneCeiling(height=2.5, material=""))
    ceilings = asyncio.run(Level(zones=[kept, dropped, blank]).all_ceilings())
    assert [c.name for c in ceilings] == ["kept"]


def test_legacy_walls_with_empty_material_are_dropped():
    walls = [
        {"start": [0, 0], "end": [1, 0]},
        {"start": [1, 0], "end": [2, 0], "material": "Marble"},
        {"start": [2, 0], "end": [3, 0], "material": ""},
    ]
    level = _legacy_level({"name": "z", "corners": [], "walls": walls})
    assert _segments(level.all_walls) == [((0, 0), (1, 0)), ((1, 0), (2, 0))]
    assert [wall.material for wall in level.all_walls] == [None, MaterialIdentifier("Marble")]


def test_legacy_zone_mat_key_aliases():
    level = _legacy_level({"name": "z", "mat": "", "ceiling_mat": "", "wall_mat": ""})
    zone = level.zones[0]
    assert zone.floor is None
    assert zone.ceiling is None
    assert level.pedestrian_walls is False


def test_legacy_zone_canonical_key_wins_over_alias():
    level = _legacy_level({"name": "z", "material": "Porcelain_Tile_4", "mat": ""})
    assert level.zones[0].floor.name == "Porcelain_Tile_4"


def test_zone_material_defaults_nonempty():
    zone = _make_zone()
    assert zone.floor.name
    assert zone.ceiling.material.name
    assert Level().pedestrian_walls is True


def _reception(*microphones: WorldMicrophone) -> Level:
    return Level(zones=[Zone(name="reception", polygon=_square_corners(10.0), ceiling=ZoneCeiling(height=2.9))], microphones=list(microphones))


def test_world_microphones_validate_zone_position_and_indices():
    level = _reception(
        WorldMicrophone(zone="reception", placement="ceiling", position=Position(2.0, 2.0, 2.9), index=1),
        WorldMicrophone(zone="reception", placement="ceiling", position=Position(4.0, 2.0, 2.9), index=2),
    )
    level.validate_microphones()
    assert [microphone.listener_id for microphone in level.microphones] == [
        "microphone:zone:reception:ceiling:1",
        "microphone:zone:reception:ceiling:2",
    ]


def test_world_microphones_structure_from_world_yaml_shape():
    world = _world(
        zones=[{"name": "reception", "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]], "ceiling": {"height": 2.9}}],
        microphones=[{"zone": "reception", "placement": "ceiling", "position": [2, 2, 2.9], "index": 1}],
    )
    level = world.levels["0"]
    level.validate_microphones()
    assert level.microphones[0].position == Position(2, 2, 2.9)


@pytest.mark.parametrize(
    ("microphone", "message"),
    [
        (WorldMicrophone(zone="missing", placement="ceiling", position=Position(2.0, 2.0, 2.9)), "unknown zone"),
        (WorldMicrophone(zone="reception", placement="ceiling", position=Position(20.0, 2.0, 2.9)), "outside zone"),
        (WorldMicrophone(zone="reception", placement="ceiling", position=Position(2.0, 2.0, 2.0)), "does not match ceiling height"),
    ],
)
def test_world_microphones_reject_invalid_construction(microphone, message):
    with pytest.raises(ValueError, match=message):
        _reception(microphone).validate_microphones()
