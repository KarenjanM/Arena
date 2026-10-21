from __future__ import annotations

import asyncio
import math
import zlib
from pathlib import Path

import pytest
import shapely
import yaml
from arena_simulation_setup.shared import CeilingLights, Light, Obstacle
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.World import FORMAT, Level, WallPath, World, WorldFormatError, WorldView, Zone, ZoneCeiling, parse_world, write_world
from arena_simulation_setup.tree.World.model import AUTO_CEILING_LIGHTS
from arena_simulation_setup.tree.World.write import dump_level
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

_L_SHAPE = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 6.0), (0.0, 6.0)]

_WORLD_YAML = """
format: arena-world/2
levels:
- id: '0'
  lights:
  - {name: ambient, fixture: dome, lux: 50}
  - {name: exit_strip, pose: {x: 10, y: 34, z: 2.2}, fixture: tube, lumens: 300, light_on: blackout}
  zones:
  - name: central_hallway
    polygon: [[0, 30], [20, 30], [20, 38], [0, 38]]
    ceiling:
      height: 2.5
      lights: {fixture: panel, spacing: 2.4, lumens: 3600, cct_K: 4000, light_on: "!blackout"}
"""


def _rect(x_length: float, y_length: float, x0: float = 0.0, y0: float = 0.0) -> list[Position]:
    return [
        Position(x0, y0),
        Position(x0 + x_length, y0),
        Position(x0 + x_length, y0 + y_length),
        Position(x0, y0 + y_length),
    ]


def _raised(length: float) -> WallPath:
    return WallPath(path=[Position(0.0, 0.0, 1.0), Position(length, 0.0, 1.0)])


def _zone(name: str, corners: list[Position], ceiling_height: float | None = None, ceiling: bool = True, ceiling_material: str | None = None, ceiling_lights: CeilingLights | None = None, ceiling_cast_shadows: bool = False) -> Zone:
    if not ceiling:
        return Zone(name=name, polygon=corners, ceiling=None)
    fields: dict = {'height': ceiling_height, 'lights': ceiling_lights, 'cast_shadows': ceiling_cast_shadows}
    if ceiling_material is not None:
        fields['material'] = ceiling_material
    return Zone(name=name, polygon=corners, ceiling=ZoneCeiling(**fields))


def _lit_zone(name: str = 'office', corners: list[Position] | None = None, ceiling_lights: CeilingLights | None = None, **kwargs) -> Zone:
    return _zone(name, corners if corners is not None else _rect(10.0, 6.0), ceiling_lights=ceiling_lights if ceiling_lights is not None else CeilingLights(), **kwargs)


def _strip(name: str = 'exit_strip', **kwargs) -> Light:
    fields = {'name': name, 'fixture': 'tube', 'pose': Pose(position=Position(1.0, 1.0, 2.2)), 'lumens': 300.0}
    fields.update(kwargs)
    return Light(**fields)


def _only_light(zone: Zone) -> Light:
    (light,) = asyncio.run(Level(zones=[zone]).all_lights())
    return light


def _ceiling_fields(level: Level) -> list[tuple]:
    return [(c.name, c.pos.x, c.pos.y, c.x_length, c.y_length, c.z, c.cast_shadows, c.material.name) for c in asyncio.run(level.all_ceilings())]


def _world_level() -> Level:
    return parse_world(yaml.safe_load(_WORLD_YAML)).levels['0']


def test_zone_without_light_keys_yields_no_lights():
    world = parse_world({'format': FORMAT, 'levels': [{'id': '0', 'zones': [{'name': 'room', 'polygon': [[0, 0], [4, 0], [4, 3], [0, 3]], 'ceiling': {'height': 2.5}}]}]})
    level = world.levels['0']
    (zone,) = level.zones
    assert zone.ceiling.lights is None
    assert level.lights == []
    assert asyncio.run(level.all_lights()) == []
    level.validate_lights()


def test_zone_without_light_keys_keeps_all_ceilings_result():
    level = Level(zones=[_zone('room', _rect(4.0, 3.0), ceiling_height=2.5, ceiling_cast_shadows=True)])
    (ceiling,) = asyncio.run(level.all_ceilings())
    assert ceiling.name == 'room'
    assert ceiling.pos.x == pytest.approx(2.0)
    assert ceiling.pos.y == pytest.approx(1.5)
    assert ceiling.x_length == pytest.approx(4.0)
    assert ceiling.y_length == pytest.approx(3.0)
    assert ceiling.z == pytest.approx(2.5)
    assert ceiling.cast_shadows is True


def test_all_ceilings_wall_top_fallback_is_unchanged():
    with_wall = Level(zones=[_zone('tall', _rect(4.0, 3.0))], walls=[_raised(4.0)])
    without_wall = Level(zones=[_zone('bare', _rect(4.0, 3.0))])
    assert asyncio.run(with_wall.all_ceilings())[0].z == pytest.approx(3.0)
    assert asyncio.run(without_wall.all_ceilings())[0].z == pytest.approx(2.0)


def test_ceiling_lights_do_not_change_all_ceilings():
    plain = Level(zones=[_zone('office', _rect(10.0, 6.0), ceiling_height=2.5)])
    lit = Level(zones=[_lit_zone(ceiling_height=2.5)], lights=[_strip()])
    assert _ceiling_fields(lit) == _ceiling_fields(plain)


def test_level_without_lights_writes_no_light_keys():
    written = dump_level(Level(id='0', zones=[Zone(name='room', polygon=_rect(4.0, 3.0))]), {})
    assert 'sounds' not in written
    assert 'lights' not in written
    assert 'ceiling' not in written['zones'][0]


def test_ceiling_rig_grid_on_rectangular_zone():
    zone = _lit_zone(corners=_rect(10.0, 6.0, x0=10.0, y0=30.0), ceiling_height=2.5, ceiling_lights=CeilingLights(spacing=2.4))
    level = Level(zones=[zone])
    (rig,) = asyncio.run(level.all_lights())
    (ceiling,) = asyncio.run(level.all_ceilings())

    assert len(rig.fixtures) == 8
    assert [fixture.x for fixture in rig.fixtures] == pytest.approx([11.4, 13.8, 16.2, 18.6] * 2)
    assert [fixture.y for fixture in rig.fixtures] == pytest.approx([31.8] * 4 + [34.2] * 4)
    assert rig.fixtures[1].x - rig.fixtures[0].x == pytest.approx(2.4)
    assert rig.fixtures[4].y - rig.fixtures[0].y == pytest.approx(2.4)
    assert sum(fixture.x for fixture in rig.fixtures) / 8 == pytest.approx(ceiling.pos.x)
    assert sum(fixture.y for fixture in rig.fixtures) / 8 == pytest.approx(ceiling.pos.y)
    assert ceiling.z == pytest.approx(2.5)
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([ceiling.z - 0.02] * 8)


def test_ceiling_rig_is_one_rig_light_named_after_the_zone():
    cfg = CeilingLights(fixture='tube', spacing=3.0, lumens=2000.0, cct_K=3000.0, cast_shadows=True, light_on='!blackout', dim=0.8, dead_fraction=0.25)
    rig = _only_light(_lit_zone(name='central_hallway', ceiling_height=2.5, ceiling_lights=cfg))
    assert rig.name == 'central_hallway'
    assert rig.rig is True
    assert rig.fixture == 'tube'
    assert rig.lumens == 2000.0
    assert rig.lux is None
    assert rig.cct_K == 3000.0
    assert rig.cast_shadows is True
    assert rig.light_on == '!blackout'
    assert rig.lit is None
    assert rig.dim == pytest.approx(0.8)
    assert rig.dead_fraction == pytest.approx(0.25)
    assert rig.pose is None
    assert rig.attach == ''
    assert rig.frame == ''
    assert len(rig.fixture_ranks) == len(rig.fixtures)
    assert [(entry.role, entry.name) for entry in rig.semantics] == [('predicate', 'lit'), ('state', 'dim'), ('state', 'dead_fraction')]


def test_ceiling_rig_carries_lit_false():
    rig = _only_light(_lit_zone(ceiling_height=2.5, ceiling_lights=CeilingLights(lit=False)))
    assert rig.lit is False
    assert rig.semantics[0].value is False


def test_ceiling_rig_keeps_rect_fixtures_inside_l_shaped_zone():
    zone = _lit_zone(name='ell', corners=[Position(x, y) for x, y in _L_SHAPE], ceiling_height=2.5, ceiling_lights=CeilingLights(spacing=1.5))
    rig = _only_light(zone)
    polygon = shapely.Polygon(_L_SHAPE)

    assert len(rig.fixtures) == 7
    assert [fixture.x for fixture in rig.fixtures] == pytest.approx([0.75, 2.25, 3.75, 5.25, 0.75, 0.75, 0.75])
    assert [fixture.y for fixture in rig.fixtures] == pytest.approx([0.75, 0.75, 0.75, 0.75, 2.25, 3.75, 5.25])
    for fixture in rig.fixtures:
        assert polygon.covers(shapely.box(fixture.x - 0.3, fixture.y - 0.3, fixture.x + 0.3, fixture.y + 0.3))


def test_ceiling_rig_keeps_disk_fixtures_inside_l_shaped_zone():
    zone = _lit_zone(name='ell', corners=[Position(x, y) for x, y in _L_SHAPE], ceiling_height=2.5, ceiling_lights=CeilingLights(fixture='downlight', spacing=1.0))
    rig = _only_light(zone)
    polygon = shapely.Polygon(_L_SHAPE)

    assert len(rig.fixtures) == 20
    for fixture in rig.fixtures:
        assert polygon.covers(shapely.Point(fixture.x, fixture.y).buffer(0.1))


def test_ceiling_rig_falls_back_to_one_fixture_in_a_small_zone():
    zone = _lit_zone(name='closet', corners=_rect(0.4, 0.4, x0=3.0, y0=3.0), ceiling_height=2.5)
    rig = _only_light(zone)
    (fixture,) = rig.fixtures
    assert shapely.Polygon([(3.0, 3.0), (3.4, 3.0), (3.4, 3.4), (3.0, 3.4)]).covers(shapely.Point(fixture.x, fixture.y))
    assert fixture.z == pytest.approx(2.48)
    assert len(rig.fixture_ranks) == 1


def test_ceiling_rig_grid_has_at_least_one_row_and_column():
    zone = _lit_zone(name='corridor', corners=_rect(10.0, 1.5), ceiling_height=2.5, ceiling_lights=CeilingLights(spacing=2.4))
    rig = _only_light(zone)
    assert len(rig.fixtures) == 4
    assert [fixture.y for fixture in rig.fixtures] == pytest.approx([0.75] * 4)


def test_ceiling_rig_z_follows_ceiling_height():
    level = Level(zones=[_lit_zone(ceiling_height=3.1)])
    (rig,) = asyncio.run(level.all_lights())
    (ceiling,) = asyncio.run(level.all_ceilings())
    assert ceiling.z == pytest.approx(3.1)
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([3.08] * len(rig.fixtures))


def test_ceiling_rig_z_falls_back_to_wall_top():
    level = Level(zones=[_lit_zone()], walls=[_raised(10.0)])
    (rig,) = asyncio.run(level.all_lights())
    (ceiling,) = asyncio.run(level.all_ceilings())
    assert ceiling.z == pytest.approx(3.0)
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([ceiling.z - 0.02] * len(rig.fixtures))


def test_ceiling_rig_z_defaults_without_walls_or_height():
    level = Level(zones=[_lit_zone()])
    (rig,) = asyncio.run(level.all_lights())
    (ceiling,) = asyncio.run(level.all_ceilings())
    assert ceiling.z == pytest.approx(2.0)
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([1.98] * len(rig.fixtures))


def test_fixture_ranks_identical_across_derivations():
    first = _only_light(_lit_zone(corners=_rect(20.0, 13.0), ceiling_height=2.5))
    second = _only_light(_lit_zone(corners=_rect(20.0, 13.0), ceiling_height=2.5))
    assert len(first.fixtures) == 40
    assert first.fixture_ranks == second.fixture_ranks
    assert first.fixture_ranks == [zlib.crc32(f'office:{i}'.encode()) / 2**32 for i in range(40)]
    assert all(0.0 <= rank < 1.0 for rank in first.fixture_ranks)


def test_fixture_ranks_depend_on_zone_name():
    office = _only_light(_lit_zone(name='office', ceiling_height=2.5))
    lobby = _only_light(_lit_zone(name='lobby', ceiling_height=2.5))
    assert office.fixture_ranks != lobby.fixture_ranks


def test_alive_is_monotone_in_dead_fraction():
    rig = _only_light(_lit_zone(corners=_rect(20.0, 13.0), ceiling_height=2.5))
    alive_at_02 = {i for i, alive in enumerate(rig.alive(0.2)) if alive}
    alive_at_05 = {i for i, alive in enumerate(rig.alive(0.5)) if alive}
    assert alive_at_05 <= alive_at_02
    assert rig.alive(0.0) == [True] * 40
    assert rig.alive(1.0) == [False] * 40

    previous = set(range(40))
    for step in range(1, 11):
        current = {i for i, alive in enumerate(rig.alive(step / 10)) if alive}
        assert current <= previous
        previous = current


def test_all_lights_orders_level_lights_then_rigs():
    ambient = Light(name='ambient', fixture='dome', lux=50.0)
    first = _lit_zone(name='first', ceiling_height=2.5)
    second = _zone('second', _rect(4.0, 4.0, x0=20.0))
    third = _lit_zone(name='third', corners=_rect(6.0, 6.0, x0=40.0), ceiling_height=2.5)
    level = Level(zones=[first, second, third], lights=[ambient, _strip('first_strip'), _strip('second_strip', pose=Pose(position=Position(21.0, 1.0, 2.2)))])
    lights = asyncio.run(level.all_lights())
    assert [light.name for light in lights] == ['ambient', 'first_strip', 'second_strip', 'first', 'third']
    assert [light.rig for light in lights] == [False, False, False, True, True]
    assert lights[0] is ambient


def test_validate_lights_accepts_ambient_explicit_and_rig():
    level = Level(
        zones=[_lit_zone(ceiling_height=2.5)],
        lights=[_strip(), Light(name='ambient', fixture='dome', lux=50.0), Light(name='sun', fixture='sun', lux=10000.0)],
    )
    level.validate_lights()


@pytest.mark.parametrize('fixture', ['panel', 'tube', 'downlight', 'spot', 'bulb'])
def test_validate_lights_accepts_every_local_fixture_at_level(fixture):
    lamp = Light(name='lamp', fixture=fixture, pose=Pose(position=Position(30.0, 30.0, 2.0)), lumens=800.0)
    level = Level(zones=[_zone('room', _rect(4.0, 4.0))], lights=[lamp])
    level.validate_lights()
    assert asyncio.run(level.all_lights()) == [lamp]


def test_validate_lights_rejects_duplicate_names_in_a_level():
    with pytest.raises(ValueError, match="duplicate light 'strip'"):
        Level(zones=[_zone('room', _rect(4.0, 4.0))], lights=[_strip('strip'), _strip('strip')]).validate_lights()


def test_zones_take_no_lights_key():
    raw = {'format': FORMAT, 'levels': [{'id': '0', 'zones': [{'name': 'room', 'polygon': [[0, 0], [4, 0], [4, 4], [0, 4]], 'lights': [{'name': 'sky', 'fixture': 'dome', 'lux': 50.0}]}]}]}
    with pytest.raises(WorldFormatError, match=r"levels\[0\]\.zones\[0\]: unknown keys \['lights'\]"):
        parse_world(raw)


def test_world_rejects_duplicate_light_names_across_levels():
    first = Level(id='0', zones=[_zone('first', _rect(4.0, 4.0))], lights=[_strip('strip')])
    second = Level(id='1', zones=[_zone('second', _rect(4.0, 4.0, x0=10.0))], lights=[_strip('strip')])
    with pytest.raises(ValueError, match="name 'strip' is used by a light and a light"):
        World.from_levels(first, second).validate()


def test_validate_lights_rejects_duplicate_ambient_names():
    level = Level(lights=[Light(name='ambient', fixture='dome', lux=50.0), Light(name='ambient', fixture='sun', lux=100.0)])
    with pytest.raises(ValueError, match="duplicate light 'ambient'"):
        level.validate_lights()


def test_validate_lights_rejects_explicit_light_named_like_a_rig_zone():
    with pytest.raises(ValueError, match="duplicate light 'office'"):
        Level(zones=[_zone('first', _rect(4.0, 4.0)), _lit_zone(name='office')], lights=[_strip('office')]).validate_lights()


def test_validate_lights_rejects_ambient_light_named_like_a_rig_zone():
    level = Level(zones=[_lit_zone(name='office')], lights=[Light(name='office', fixture='dome', lux=50.0)])
    with pytest.raises(ValueError, match="duplicate light 'office'"):
        level.validate_lights()


def test_validate_lights_rejects_two_rig_zones_with_one_name():
    level = Level(zones=[_lit_zone(name='office'), _lit_zone(name='office', corners=_rect(4.0, 4.0, x0=20.0))])
    with pytest.raises(ValueError, match="duplicate light 'office'"):
        level.validate_lights()


def test_light_named_like_an_unlit_zone_passes_the_level_but_not_the_world():
    level = Level(zones=[_zone('room', _rect(4.0, 4.0))], lights=[_strip('room')])
    level.validate_lights()
    with pytest.raises(ValueError, match='names are unique across the world'):
        World.from_levels(level).validate()


def test_validate_lights_rejects_ceiling_lights_without_ceiling_material():
    with pytest.raises(ValueError, match='has ceiling lights but no ceiling'):
        Level(zones=[_lit_zone(ceiling_material='')]).validate_lights()


def test_zone_without_ceiling_carries_no_ceiling_lights():
    world = parse_world({'format': FORMAT, 'levels': [{'id': '0', 'zones': [{'name': 'open', 'polygon': [[0, 0], [4, 0], [4, 4], [0, 4]], 'ceiling': False}]}]})
    assert world.levels['0'].zones[0].ceiling is None
    assert asyncio.run(world.levels['0'].all_lights()) == []


def test_validate_lights_rejects_ceiling_lights_without_polygon():
    with pytest.raises(ValueError, match='has ceiling lights but no polygon'):
        Level(zones=[_lit_zone(corners=[])]).validate_lights()


def test_world_validate_runs_validate_lights():
    world = World.from_levels(Level(zones=[_lit_zone(ceiling_material='')]))
    with pytest.raises(ValueError, match='has ceiling lights but no ceiling'):
        world.validate()


def test_shifted_moves_positioned_lights_only():
    positioned = _strip('positioned', pose=Pose(position=Position(1.0, 1.0, 2.2)))
    attached = _strip('attached', pose=None, attach='desk')
    ambient = Light(name='ambient', fixture='dome', lux=50.0)
    level = Level(zones=[_zone('hall', _rect(4.0, 4.0))], lights=[positioned, attached, ambient])

    moved = {light.name: light for light in level.shifted(3.0, -2.0).lights}

    assert moved['positioned'].position.x == pytest.approx(4.0)
    assert moved['positioned'].position.y == pytest.approx(-1.0)
    assert moved['positioned'].position.z == pytest.approx(2.2)
    assert moved['attached'].pose is None
    assert moved['attached'].offset == Pose()
    assert moved['ambient'].pose is None
    assert positioned.position.x == pytest.approx(1.0)


def test_shifted_leaves_the_sun_aim_alone():
    aim = Pose(orientation=Orientation.from_euler((0.0, -0.5, 0.0)))
    level = Level(lights=[Light(name='sun', fixture='sun', lux=10000.0, pose=aim)]).shifted(3.0, -2.0)
    assert level.lights[0].pose == Pose(orientation=Orientation.from_euler((0.0, -0.5, 0.0)))
    level.validate_lights()


def test_shifted_moves_rig_fixtures_with_the_zone():
    level = Level(zones=[_lit_zone(ceiling_height=2.5)])
    (before,) = asyncio.run(level.all_lights())
    (after,) = asyncio.run(level.shifted(100.0, 50.0).all_lights())
    assert [fixture.x for fixture in after.fixtures] == pytest.approx([fixture.x + 100.0 for fixture in before.fixtures])
    assert [fixture.y for fixture in after.fixtures] == pytest.approx([fixture.y + 50.0 for fixture in before.fixtures])
    assert after.fixture_ranks == before.fixture_ranks


def test_world_level_carries_lights_and_ceiling_lights():
    ambient = Light(name='ambient', fixture='dome', lux=50.0)
    level = World.from_levels(Level(zones=[_lit_zone()], lights=[ambient])).levels['0']
    assert level.lights == [ambient]
    assert level.zones[0].ceiling.lights == CeilingLights()


def test_compact_world_carries_and_shifts_lights():
    ambient = Light(name='ambient', fixture='dome', lux=50.0)
    strip = _strip(pose=Pose(position=Position(1.0, 1.0, 2.2)))
    world = World.from_levels(Level(zones=[_lit_zone('hall', _rect(4.0, 4.0))], lights=[ambient, strip]))

    compacted = world.compact_world({'0': (5.0, 7.0)})

    assert [light.name for light in compacted.lights] == ['ambient', 'exit_strip']
    assert compacted.lights[1].position.x == pytest.approx(6.0)
    assert compacted.lights[1].position.y == pytest.approx(8.0)
    assert compacted.zones[0].ceiling.lights == CeilingLights()
    assert strip.position.x == pytest.approx(1.0)


def test_world_yaml_structures_lights_and_ceiling_lights():
    level = _world_level()
    (zone,) = level.zones

    ambient, strip = level.lights
    assert ambient.name == 'ambient'
    assert ambient.fixture == 'dome'
    assert ambient.lux == 50.0

    assert zone.ceiling.lights == CeilingLights(fixture='panel', spacing=2.4, lumens=3600.0, cct_K=4000.0, light_on='!blackout')

    assert strip.name == 'exit_strip'
    assert strip.fixture == 'tube'
    assert strip.position == Position(10.0, 34.0, 2.2)
    assert strip.lumens == 300.0
    assert strip.light_on == 'blackout'
    assert strip.rig is False

    level.validate_lights()


def test_world_yaml_derives_rig_and_semantics():
    ambient, strip, rig = asyncio.run(_world_level().all_lights())

    assert [ambient.name, strip.name, rig.name] == ['ambient', 'exit_strip', 'central_hallway']
    assert rig.rig is True
    assert len(rig.fixtures) == 24
    assert [fixture.z for fixture in rig.fixtures] == pytest.approx([2.48] * 24)
    assert rig.semantics[0].value is None
    assert rig.semantics[0].params == {'light_on': '!blackout'}
    assert strip.semantics[0].params == {'light_on': 'blackout'}
    assert ambient.semantics[0].value is True


def test_zone_yaml_rejects_unknown_ceiling_lights_key():
    raw = {'format': FORMAT, 'levels': [{'id': '0', 'zones': [{'name': 'room', 'polygon': [[0, 0], [4, 0], [4, 3], [0, 3]], 'ceiling': {'lights': {'fixture': 'panel', 'pitch': 2.0}}}]}]}
    with pytest.raises(WorldFormatError, match=r'levels\[0\]\.zones\[0\]\.ceiling'):
        parse_world(raw)


def test_level_yaml_rejects_invalid_light():
    raw = {'format': FORMAT, 'levels': [{'id': '0', 'lights': [{'name': 'lamp', 'fixture': 'panel', 'lumens': 3600}]}]}
    with pytest.raises(WorldFormatError, match=r'levels\[0\]\.lights\[0\]'):
        parse_world(raw)


def test_level_write_omits_derived_light_fields():
    written = dump_level(_world_level(), {})

    ambient, strip = written['lights']
    assert ambient == {'name': 'ambient', 'fixture': 'dome', 'lux': 50.0}
    (zone,) = written['zones']
    assert zone['ceiling']['lights'] == {'cct_K': 4000.0, 'light_on': '!blackout'}
    assert set(strip) == {'name', 'fixture', 'pose', 'lumens', 'light_on'}
    assert 'rig' not in strip
    assert 'fixtures' not in strip
    assert 'fixture_ranks' not in strip


def test_level_lights_round_trip_through_yaml():
    world = parse_world(yaml.safe_load(_WORLD_YAML))
    reloaded = parse_world(yaml.safe_load(write_world(world))).levels['0']
    level = world.levels['0']

    assert [(light.name, light.fixture, light.lux) for light in reloaded.lights][:1] == [('ambient', 'dome', 50.0)]
    assert reloaded.zones[0].ceiling.lights == level.zones[0].ceiling.lights
    strip = reloaded.lights[1]
    assert (strip.name, strip.fixture, strip.lumens, strip.light_on) == ('exit_strip', 'tube', 300.0, 'blackout')
    assert strip.position == Position(10.0, 34.0, 2.2)

    original = asyncio.run(level.all_lights())
    derived = asyncio.run(reloaded.all_lights())
    assert [light.name for light in derived] == [light.name for light in original]
    assert derived[-1].fixtures == original[-1].fixtures
    assert derived[-1].fixture_ranks == original[-1].fixture_ranks


def test_world_save_writes_lights_that_load_from_disk(tmp_path: Path):
    WorldView(tmp_path / 'lit_world').save(parse_world(yaml.safe_load(_WORLD_YAML)))

    loaded = WorldView(tmp_path / 'lit_world').load()

    level = loaded.levels['0']
    assert [(light.name, light.fixture, light.lux) for light in level.lights] == [('ambient', 'dome', 50.0), ('exit_strip', 'tube', None)]
    assert level.zones[0].ceiling.lights == CeilingLights(cct_K=4000.0, light_on='!blackout')
    assert [light.name for light in asyncio.run(level.all_lights())] == ['ambient', 'exit_strip', 'central_hallway']


def _desk(name: str = 'desk', x: float = 4.0, yaw: float = 0.0) -> Obstacle:
    return Obstacle(name=name, pose=Pose(Position(x, 2.0, 0.0), Orientation.from_yaw(yaw)), model=ObjectIdentifier('test_model'), scale=None)


def _desk_level(lights: list[Light], yaw: float = 0.0, desks: int = 1) -> Level:
    return Level(zones=[_zone('office', _rect(10.0, 6.0))], lights=lights, objects=[_desk(yaw=yaw) for _ in range(desks)])


def test_attached_light_sits_on_its_object_with_turned_offset() -> None:
    lamp = Light(name='desk_lamp', fixture='spot', attach='desk', offset=Pose(position=Position(1.0, 0.0, 0.8), orientation=Orientation.from_euler((0.0, -math.pi / 4, 0.0))), lumens=400.0)
    (light,) = asyncio.run(_desk_level([lamp], yaw=math.pi / 2).all_lights())
    assert light.attach == ''
    assert (light.position.x, light.position.y, light.position.z) == pytest.approx((4.0, 3.0, 0.8))
    assert light.direction == pytest.approx((0.0, 2**-0.5, -(2**-0.5)))


@pytest.mark.parametrize(('desks', 'word'), [(0, 'unknown'), (2, 'ambiguous')])
def test_attached_light_rejects_unresolvable_objects(desks: int, word: str) -> None:
    lamp = Light(name='desk_lamp', fixture='downlight', attach='desk', lumens=400.0)
    level = _desk_level([lamp], desks=desks)
    with pytest.raises(ValueError, match=word):
        level.validate_lights()
    with pytest.raises(ValueError, match=word):
        asyncio.run(level.all_lights())


def test_frame_light_keeps_its_frame_and_offset() -> None:
    headlight = Light(name='headlight', fixture='spot', frame='jackal/base_link', offset=Pose(position=Position(0.2, 0.0, 0.3)), lumens=800.0)
    (light,) = asyncio.run(_desk_level([headlight]).all_lights())
    assert light.frame == 'jackal/base_link'
    assert light.pose is None
    assert (light.offset.position.x, light.offset.position.z) == (0.2, 0.3)


_LAMP_ANNOTATION = {
    'name': 'Desk_Lamp',
    'path': 'Object/Desk_Lamp',
    'bounding_box': [[-0.1, 0.1], [-0.1, 0.1], [0.0, 0.5]],
    'lights': [
        {'name': 'bulb', 'fixture': 'bulb', 'pose': [[0.1, 0.0, 0.45], [1.0, 0.0, 0.0, 0.0]], 'lumens': 800.0, 'cct_K': 2700.0, 'glow': 'Shade'},
        {'name': 'reading', 'fixture': 'spot', 'pose': [0.0, 0.0, 0.4, 0.0, -math.pi / 4, 0.0], 'lumens': 300.0, 'cone_deg': 60.0},
    ],
}


@pytest.fixture
def world_assets(tmp_path: Path):
    from arena_simulation_setup.tree import DynamicPaths

    lamp = tmp_path / 'assets' / 'Common' / 'Object' / 'Desk_Lamp'
    lamp.mkdir(parents=True)
    (lamp / 'annotation.yaml').write_text(yaml.safe_dump(_LAMP_ANNOTATION))
    plain = tmp_path / 'assets' / 'Common' / 'Object' / 'Plain_Desk'
    plain.mkdir(parents=True)
    (plain / 'annotation.yaml').write_text(yaml.safe_dump({'name': 'Plain_Desk', 'path': 'Object/Plain_Desk'}))
    previous = DynamicPaths.WORLD.path
    DynamicPaths.WORLD.path = tmp_path
    yield tmp_path
    DynamicPaths.WORLD.path = previous


def _lamp_level(*objects: dict, lights: list[Light] | None = None) -> Level:
    return Level(zones=[_zone('study', _rect(6.0, 6.0))], objects=[converter.structure(raw, Obstacle) for raw in objects], lights=lights or [])


def test_object_lights_follow_their_entity_pose_and_scale(world_assets: Path) -> None:
    lamp = {'name': 'desk_lamp', 'model': 'Common/Desk_Lamp', 'pose': {'position': [2.0, 1.0, 0.7], 'orientation': {'x': 0.0, 'y': 0.0, 'z': math.sin(math.pi / 4), 'w': math.cos(math.pi / 4)}}, 'scale': [2.0, 2.0, 2.0]}
    bulb, reading = asyncio.run(_lamp_level(lamp).all_lights())
    assert (bulb.name, reading.name) == ('desk_lamp_bulb', 'desk_lamp_reading')
    assert bulb.intrinsic and reading.intrinsic
    assert bulb.attach == ''
    assert (bulb.owner, bulb.glow) == ('desk_lamp', 'Shade')
    assert (reading.owner, reading.glow) == ('desk_lamp', '')
    assert (bulb.position.x, bulb.position.y, bulb.position.z) == pytest.approx((2.0, 1.2, 1.6))
    assert reading.direction == pytest.approx((0.0, 2**-0.5, -(2**-0.5)))
    assert reading.spec.cone_deg == 60.0
    assert bulb.cct_K == 2700.0


def test_object_lights_take_the_placement_settings(world_assets: Path) -> None:
    lamp = {'name': 'desk_lamp', 'model': 'Common/Desk_Lamp', 'pose': {'position': [2.0, 1.0, 0.7]}, 'light': {'light_on': '!daylight', 'dim': 0.5}}
    lights = asyncio.run(_lamp_level(lamp).all_lights())
    assert [(light.light_on, light.dim) for light in lights] == [('!daylight', 0.5), ('!daylight', 0.5)]


def test_objects_without_annotated_lights_add_none(world_assets: Path) -> None:
    desk = {'name': 'desk', 'model': 'Common/Plain_Desk', 'pose': {'position': [1.0, 1.0, 0.0]}}
    assert asyncio.run(_lamp_level(desk).all_lights()) == []


def test_object_lights_follow_declared_lights_and_count_once_per_placement(world_assets: Path) -> None:
    lamps = [{'name': f'lamp_{index}', 'model': 'Common/Desk_Lamp', 'pose': {'position': [float(index), 1.0, 0.0]}} for index in (1, 2)]
    names = [light.name for light in asyncio.run(_lamp_level(*lamps, lights=[_strip()]).all_lights())]
    assert names == ['exit_strip', 'lamp_1_bulb', 'lamp_1_reading', 'lamp_2_bulb', 'lamp_2_reading']


def test_object_light_named_like_a_declared_light_is_rejected(world_assets: Path) -> None:
    level = _lamp_level({'name': 'desk_lamp', 'model': 'Common/Desk_Lamp', 'pose': {'position': [2.0, 1.0, 0.0]}}, lights=[_strip(name='desk_lamp_bulb')])
    with pytest.raises(ValueError, match='collides'):
        asyncio.run(level.all_lights())


def test_validate_lights_leaves_object_lights_to_spawn_time(world_assets: Path) -> None:
    _lamp_level({'name': 'desk_lamp', 'model': 'Common/Desk_Lamp', 'pose': {'position': [2.0, 1.0, 0.0]}}, lights=[_strip(name='desk_lamp_bulb')]).validate_lights()


def _auto_level() -> Level:
    own = CeilingLights(lumens=3000.0)
    lamp = Light(name='lamp', fixture='bulb', pose=Pose(position=Position(11.0, 1.0, 1.0)), lumens=800.0)
    return Level(
        zones=[
            _zone('bare', _rect(4.0, 4.0)),
            _zone('rigged', _rect(4.0, 4.0, 5.0), ceiling_lights=own),
            _zone('lamped', _rect(4.0, 4.0, 10.0)),
            _zone('open', _rect(4.0, 4.0, 15.0), ceiling=False),
        ],
        lights=[lamp],
    )


def test_auto_lighting_rigs_only_ceilinged_zones_without_lights() -> None:
    zones = {zone.name: zone for zone in _auto_level().with_lighting('auto').zones}
    assert zones['bare'].ceiling.lights == AUTO_CEILING_LIGHTS
    assert zones['rigged'].ceiling.lights.lumens == 3000.0
    assert zones['lamped'].ceiling.lights is None
    assert zones['open'].ceiling is None
    rig = next(light for light in asyncio.run(_auto_level().with_lighting('auto').all_lights()) if light.name == 'bare')
    assert (rig.fixture, rig.lumens, rig.cct_K, rig.rig) == ('panel', 7200.0, 6500.0, True)


def test_authored_lighting_leaves_the_level_alone() -> None:
    level = _auto_level()
    assert level.with_lighting('authored') is level


def test_unknown_lighting_mode_names_the_modes() -> None:
    with pytest.raises(ValueError, match="world.lighting 'bright' is not one of authored, auto"):
        _auto_level().with_lighting('bright')


def test_level_objects_are_the_static_entities() -> None:
    level = Level(zones=[_zone('office', _rect(10.0, 6.0))], objects=[_desk('crate', x=30.0), _desk()])
    assert [entity.name for entity in level.all_static_entities] == ['crate', 'desk']


def test_level_light_anchors_on_an_object_outside_every_zone() -> None:
    lamp = Light(name='lamp', fixture='downlight', attach='crate', offset=Pose(position=Position(0.5, 0.0, 0.8)), lumens=400.0)
    level = Level(zones=[_zone('office', _rect(10.0, 6.0))], lights=[lamp], objects=[_desk('crate', x=30.0)])
    level.validate_lights()
    (light,) = asyncio.run(level.all_lights())
    assert (light.owner, light.attach) == ('crate', '')
    assert (light.position.x, light.position.y, light.position.z) == pytest.approx((30.5, 2.0, 0.8))


def test_level_light_anchors_on_an_object_inside_a_zone() -> None:
    lamp = Light(name='lamp', fixture='downlight', attach='crate', lumens=400.0)
    (light,) = asyncio.run(Level(zones=[_zone('office', _rect(10.0, 6.0))], lights=[lamp], objects=[_desk('crate', x=5.0)]).all_lights())
    assert light.position.x == pytest.approx(5.0)


def test_shifted_moves_objects_and_lights() -> None:
    lamp = Light(name='lamp', fixture='bulb', pose=Pose(position=Position(1.0, 1.0, 1.0)), lumens=800.0)
    level = Level(lights=[lamp], objects=[_desk('crate', x=30.0)]).shifted(5.0, 7.0)
    assert (level.lights[0].position.x, level.lights[0].position.y) == pytest.approx((6.0, 8.0))
    assert (level.objects[0].pose.position.x, level.objects[0].pose.position.y) == pytest.approx((35.0, 9.0))


def test_compact_world_carries_objects() -> None:
    crate = _desk('crate', x=30.0)
    world = World.from_levels(Level(objects=[crate]))
    compacted = world.compact_world({'0': (5.0, 0.0)})
    assert [entity.name for entity in compacted.objects] == ['crate']
    assert compacted.objects[0].pose.position.x == pytest.approx(35.0)
    assert crate.pose.position.x == pytest.approx(30.0)


def test_world_yaml_structures_objects_and_level_lights() -> None:
    raw = yaml.safe_load(
        """
format: arena-world/2
levels:
- id: '0'
  lights:
  - {name: lamp, fixture: bulb, pose: {x: 30, y: 2, z: 1.5}, lumens: 800}
  objects:
  - {name: crate, model: test_model, pose: [30, 2]}
  zones:
  - name: room
    polygon: [[0, 0], [4, 0], [4, 4], [0, 4]]
"""
    )
    world = parse_world(raw)
    level = world.levels['0']
    assert [entity.name for entity in level.objects] == ['crate']
    assert level.lights[0].fixture == 'bulb'
    again = parse_world(yaml.safe_load(write_world(world))).levels['0']
    assert [entity.name for entity in again.objects] == ['crate']


def test_world_identifiers_include_object_models() -> None:
    world = World.from_levels(Level(objects=[_desk('crate', x=30.0)]))
    assert [identifier.name for identifier in world.identifiers()] == ['test_model']


def test_auto_lighting_skips_a_zone_holding_a_level_light() -> None:
    lamp = Light(name='lamp', fixture='bulb', pose=Pose(position=Position(2.0, 2.0, 1.0)), lumens=800.0)
    sun = Light(name='sun', fixture='sun', lux=10000.0)
    level = Level(zones=[_zone('held', _rect(4.0, 4.0)), _zone('bare', _rect(4.0, 4.0, 10.0))], lights=[lamp, sun])
    zones = {zone.name: zone for zone in level.with_lighting('auto').zones}
    assert zones['held'].ceiling.lights is None
    assert zones['bare'].ceiling.lights == AUTO_CEILING_LIGHTS


def test_auto_lighting_skips_a_zone_holding_an_anchored_level_light() -> None:
    lamp = Light(name='lamp', fixture='downlight', attach='crate', lumens=400.0)
    level = Level(zones=[_zone('held', _rect(4.0, 4.0))], lights=[lamp], objects=[_desk('crate', x=2.0)])
    assert level.with_lighting('auto').zones[0].ceiling.lights is None
