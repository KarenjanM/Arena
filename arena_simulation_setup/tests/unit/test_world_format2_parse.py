from __future__ import annotations

import pytest
import yaml
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier
from arena_simulation_setup.tree.World import FORMAT, WorldFormatError, parse_world
from arena_simulation_setup.tree.World.model import World

FULL_WORLD = """\
format: arena-world/2
defaults:
  floor: Concrete_Smooth
  ceiling: {height: 2.7, material: Concrete_Smooth}
  wall: {material: Plaster_Wall}
  door: {kind: hinged, material: Aluminum_Anodized}
schedules:
- {name: fire_alarm, semantics: [{preset: schedule, params: {regime: alarm}}]}
signals:
- {name: crossing, semantics: [{preset: signal}]}
levels:
- id: '0'
  pedestrian_walls: false
  zones:
  - name: hall
    polygon: [[0, 0], [8, 0], [8, 4], [0, 4]]
    walls: outline
    ceiling: {lights: {fixture: panel, spacing: 2.4, lumens: 3600}}
    semantics: [{state: max_speed, value: 1.5}]
  - name: office
    polygon: [[0, 4], [8, 4], [8, 8], [0, 8]]
    floor: Marble
    walls: {outline: true, open_to: [hall], material: Brick}
  walls:
  - {path: [[0, 8], [0, 12], [8, 12]], kind: patient_room_walls}
  - {path: [[10, 0], [12, 0], [12, 2]], closed: true}
  doors:
  - {name: main_entrance, span: [[3, 0], [4, 0]]}
  elevators:
  - {name: lift_0, position: [6, 6], destinations: [lift_1]}
  objects:
  - {name: desk_1, model: Desk, pose: [2, 1, 1.5708], humansim: {type: desk}}
  humans:
  - {name: walker, model: adult, pose: [1, 1], waypoints: [[2, 2], [3, 3]], velocity: 0.8}
  lights:
  - {name: exit_strip, fixture: tube, lumens: 300, pose: {x: 4, y: 3.9, z: 2.2}}
  sounds:
  - {name: hall_siren, asset_id: alarm_loop, position: [4, 2]}
- id: '1'
  zones:
  - {name: upstairs, polygon: [[0, 0], [4, 0], [4, 4], [0, 4]], floor: false, ceiling: false}
  elevators:
  - {name: lift_1, position: [2, 2], destinations: [lift_0]}
"""


def _parse(text: str) -> World:
    return parse_world(yaml.safe_load(text))


def _level(**level: object) -> dict:
    return {'format': FORMAT, 'levels': [{'id': '0', **level}]}


def _square(name: str, x: float = 0.0, **zone: object) -> dict:
    return {'name': name, 'polygon': [[x, 0], [x + 4, 0], [x + 4, 4], [x, 4]], **zone}


def test_world_keys_parse_into_world_scope_lists() -> None:
    world = _parse(FULL_WORLD)
    assert list(world.levels) == ['0', '1']
    assert [level.id for level in world.levels.values()] == ['0', '1']
    assert [schedule.name for schedule in world.schedules] == ['fire_alarm']
    assert [cfg.name for cfg in world.schedules[0].semantics] == ['state', 'active', 'window_remaining']
    assert [signal.name for signal in world.signals] == ['crossing']
    assert world.defaults['door'] == {'kind': 'hinged', 'material': 'Aluminum_Anodized'}
    assert world.validate() == []


def test_level_keys_parse_into_level_lists() -> None:
    level = _parse(FULL_WORLD).levels['0']
    assert [zone.name for zone in level.zones] == ['hall', 'office']
    assert [len(path.path) for path in level.walls] == [3, 3]
    assert [door.name for door in level.doors] == ['main_entrance']
    assert [elevator.name for elevator in level.elevators] == ['lift_0']
    assert level.elevators[0].destinations == ['lift_1']
    assert [entity.name for entity in level.objects] == ['desk_1']
    assert level.objects[0].humansim == {'type': 'desk'}
    assert [human.name for human in level.humans] == ['walker']
    assert [(point.x, point.y) for point in level.humans[0].waypoints] == [(2.0, 2.0), (3.0, 3.0)]
    assert level.humans[0].velocity == 0.8
    assert [light.name for light in level.lights] == ['exit_strip']
    assert [sound.name for sound in level.sounds] == ['hall_siren']
    assert level.pedestrian_walls is False
    assert _parse(FULL_WORLD).levels['1'].pedestrian_walls is True


def test_zone_keys_parse() -> None:
    hall, office = _parse(FULL_WORLD).levels['0'].zones
    assert [(point.x, point.y) for point in hall.polygon] == [(0, 0), (8, 0), (8, 4), (0, 4)]
    assert hall.floor == MaterialIdentifier('Concrete_Smooth')
    assert hall.ceiling is not None and hall.ceiling.height == 2.7
    assert hall.ceiling.lights is not None and hall.ceiling.lights.fixture == 'panel'
    assert hall.walls is not None and hall.walls.open_to == [] and hall.walls.material == MaterialIdentifier('Plaster_Wall')
    assert [(cfg.name, cfg.value) for cfg in hall.semantics] == [('max_speed', 1.5)]
    assert office.floor == MaterialIdentifier('Marble')
    assert office.walls is not None and office.walls.open_to == ['hall'] and office.walls.material == MaterialIdentifier('Brick')


def test_zone_floor_false_means_no_floor() -> None:
    upstairs = _parse(FULL_WORLD).levels['1'].zones[0]
    assert upstairs.floor is None
    assert upstairs.ceiling is None


@pytest.mark.parametrize(('ceiling', 'height', 'material'), [(True, None, 'Concrete_Smooth'), ({'height': 3.1, 'material': 'Marble', 'cast_shadows': True}, 3.1, 'Marble'), ({}, None, 'Concrete_Smooth')])
def test_zone_ceiling_true_or_mapping_gives_a_ceiling(ceiling: object, height: float | None, material: str) -> None:
    zone = parse_world(_level(zones=[_square('z', ceiling=ceiling)])).levels['0'].zones[0]
    assert zone.ceiling is not None
    assert zone.ceiling.height == height
    assert zone.ceiling.material == MaterialIdentifier(material)


def test_zone_ceiling_false_means_no_ceiling() -> None:
    zone = parse_world(_level(zones=[_square('z', ceiling=False)])).levels['0'].zones[0]
    assert zone.ceiling is None


def test_zone_walls_take_a_kind() -> None:
    zone = parse_world(_level(zones=[_square('z', walls={'outline': True, 'open_to': [], 'kind': 'brick_walls'})])).levels['0'].zones[0]
    assert zone.walls is not None and zone.walls.kind == 'brick_walls' and zone.walls.material is None


def test_zone_without_walls_draws_none() -> None:
    zone = parse_world(_level(zones=[_square('z')])).levels['0'].zones[0]
    assert zone.walls is None


def test_closed_wall_path_returns_to_its_first_point() -> None:
    path = _parse(FULL_WORLD).levels['0'].walls[1]
    assert path.closed is True
    assert [(wall.end.x, wall.end.y) for wall in path.segments()] == [(12, 0), (12, 2), (10, 0)]


@pytest.mark.parametrize(
    ('pose', 'expected'),
    [
        ([1, 2], (1.0, 2.0, 0.0, 0.0)),
        ([1, 2, 0.5], (1.0, 2.0, 0.0, 0.5)),
        ({'x': 1, 'y': 2, 'z': 0.3, 'yaw': 0.5}, (1.0, 2.0, 0.3, 0.5)),
    ],
)
def test_object_pose_forms(pose: object, expected: tuple[float, float, float, float]) -> None:
    entity = parse_world(_level(objects=[{'name': 'o', 'model': 'Desk', 'pose': pose}])).levels['0'].objects[0]
    x, y, z, yaw = expected
    assert (entity.pose.position.x, entity.pose.position.y, entity.pose.position.z) == pytest.approx((x, y, z))
    assert entity.pose.orientation.to_yaw() == pytest.approx(yaw)


def test_object_pose_mapping_takes_roll_and_pitch() -> None:
    entity = parse_world(_level(objects=[{'name': 'o', 'model': 'Desk', 'pose': {'x': 1, 'y': 2, 'roll': 0.1, 'pitch': 0.2}}])).levels['0'].objects[0]
    assert entity.pose.orientation.to_euler() == pytest.approx((0.1, 0.2, 0.0))


def test_light_placements() -> None:
    lights = [
        {'name': 'posed', 'fixture': 'panel', 'lumens': 100, 'pose': {'x': 1, 'y': 1, 'z': 2.5}},
        {'name': 'attached', 'fixture': 'bulb', 'lumens': 100, 'attach': 'lamp', 'offset': {'x': 0, 'y': 0, 'z': 1.2}},
        {'name': 'framed', 'fixture': 'spot', 'lumens': 100, 'frame': '/base_link', 'offset': {'x': 0.2, 'y': 0, 'pitch': 0.3}},
    ]
    level = parse_world(_level(objects=[{'name': 'lamp', 'model': 'Lamp', 'pose': [1, 1]}], lights=lights)).levels['0']
    posed, attached, framed = level.lights
    assert posed.pose is not None and (posed.position.x, posed.position.y, posed.position.z) == (1.0, 1.0, 2.5)
    assert (posed.attach, posed.frame) == ('', '')
    assert attached.pose is None and attached.attach == 'lamp' and attached.offset.position.z == 1.2
    assert framed.pose is None and framed.frame == 'base_link' and framed.offset.orientation.to_euler()[1] == pytest.approx(0.3)


def test_sound_placements() -> None:
    sounds = [
        {'name': 'placed', 'asset_id': 'alarm', 'position': [1, 2]},
        {'name': 'attached', 'asset_id': 'alarm', 'attach': 'radio'},
        {'name': 'framed', 'asset_id': 'alarm', 'frame': 'base_link', 'offset': [0, 0, 1]},
    ]
    level = parse_world(_level(objects=[{'name': 'radio', 'model': 'Radio', 'pose': [1, 1]}], sounds=sounds)).levels['0']
    placed, attached, framed = level.sounds
    assert placed.position is not None and (placed.position.x, placed.position.y) == (1.0, 2.0)
    assert attached.position is None and attached.attach == 'radio'
    assert framed.position is None and framed.frame == 'base_link' and framed.offset.z == 1.0


@pytest.mark.parametrize(
    ('section', 'item', 'path'),
    [
        ('lights', {'name': 'l', 'fixture': 'bulb', 'lumens': 1, 'pose': [1, 1], 'frame': 'base_link'}, 'levels[0].lights[0]'),
        ('sounds', {'name': 's', 'asset_id': 'alarm', 'position': [1, 1], 'frame': 'base_link'}, 'levels[0].sounds[0]'),
    ],
)
def test_two_placements_are_rejected_with_their_path(section: str, item: dict, path: str) -> None:
    with pytest.raises(WorldFormatError, match=r'requires exactly one of') as raised:
        parse_world(_level(**{section: [item]}))
    assert str(raised.value).startswith(f'{path}: ')


@pytest.mark.parametrize('position', [[1, 1, 2.7], {'x': 1, 'y': 1, 'z': 2.7}])
def test_microphone_position_parses(position: object) -> None:
    level = parse_world(_level(zones=[_square('z', ceiling={'height': 2.7})], microphones=[{'zone': 'z', 'placement': 'ceiling', 'position': position}])).levels['0']
    (microphone,) = level.microphones
    assert (microphone.position.x, microphone.position.y, microphone.position.z) == (1.0, 1.0, 2.7)
    assert microphone.listener_id == 'microphone:zone:z:ceiling:1'


BOGUS = {'bogus': 1}


@pytest.mark.parametrize(
    ('raw', 'path'),
    [
        ({'format': FORMAT, 'levels': [], **BOGUS}, ''),
        ({'format': FORMAT, 'defaults': BOGUS, 'levels': []}, 'defaults'),
        ({'format': FORMAT, 'defaults': {'door': BOGUS}, 'levels': []}, 'defaults.door'),
        (_level(**BOGUS), 'levels[0]'),
        (_level(zones=[_square('a'), _square('b', 4, **BOGUS)]), 'levels[0].zones[1]'),
        (_level(zones=[_square('a', ceiling=BOGUS)]), 'levels[0].zones[0].ceiling'),
        (_level(zones=[_square('a', walls={'outline': True, **BOGUS})]), 'levels[0].zones[0].walls'),
        (_level(walls=[{'path': [[0, 0], [1, 0]], **BOGUS}]), 'levels[0].walls[0]'),
        (_level(doors=[{'name': 'd', 'span': [[0, 0], [1, 0]], **BOGUS}]), 'levels[0].doors[0]'),
        (_level(elevators=[{'name': 'e', 'position': [0, 0], **BOGUS}]), 'levels[0].elevators[0]'),
        (_level(objects=[{'name': 'o', 'model': 'Desk', 'pose': [0, 0], **BOGUS}]), 'levels[0].objects[0]'),
        (_level(humans=[{'name': 'h', 'model': 'adult', 'pose': [0, 0], **BOGUS}]), 'levels[0].humans[0]'),
        (_level(lights=[{'name': 'l', 'fixture': 'bulb', 'lumens': 1, 'pose': [0, 0], **BOGUS}]), 'levels[0].lights[0]'),
        (_level(sounds=[{'name': 's', 'asset_id': 'alarm', 'position': [0, 0], **BOGUS}]), 'levels[0].sounds[0]'),
        ({'format': FORMAT, 'schedules': [{'name': 's', **BOGUS}], 'levels': []}, 'schedules[0]'),
        ({'format': FORMAT, 'signals': [{'name': 's', **BOGUS}], 'levels': []}, 'signals[0]'),
        pytest.param(
            _level(zones=[_square('a')], microphones=[{'zone': 'a', 'placement': 'wall', 'position': [1, 1, 1], **BOGUS}]),
            'levels[0].microphones[0]',
        ),
    ],
)
def test_unknown_keys_are_rejected_with_their_yaml_path(raw: dict, path: str) -> None:
    with pytest.raises(WorldFormatError) as raised:
        parse_world(raw)
    message = str(raised.value)
    assert message.startswith(f'{path}: ')
    assert "unknown keys ['bogus']" in message


@pytest.mark.parametrize('version', ['arena-world/1', 'arena-world/3', ''])
def test_wrong_format_is_rejected(version: str) -> None:
    with pytest.raises(WorldFormatError, match=r'^format: expected'):
        parse_world({'format': version, 'levels': []})


def test_missing_format_is_rejected() -> None:
    with pytest.raises(WorldFormatError, match=r"missing keys \['format'\]"):
        parse_world({'levels': []})


def test_duplicate_level_ids_are_rejected() -> None:
    with pytest.raises(WorldFormatError, match=r"^levels\[1\]: level id '0' appears twice"):
        parse_world({'format': FORMAT, 'levels': [{'id': '0'}, {'id': 0}]})


@pytest.mark.parametrize(
    ('raw', 'path'),
    [
        (_level(walls=[{'path': [[0, 0], [1, 0]], 'kind': 'brick_walls', 'material': 'Brick'}]), 'levels[0].walls[0]'),
        ({'format': FORMAT, 'defaults': {'wall': {'kind': 'brick_walls', 'material': 'Brick'}}, 'levels': []}, 'defaults.wall'),
        pytest.param(
            _level(zones=[_square('a', walls={'outline': True, 'kind': 'brick_walls', 'material': 'Brick'})]),
            'levels[0].zones[0].walls',
        ),
    ],
)
def test_kind_and_material_together_are_rejected(raw: dict, path: str) -> None:
    with pytest.raises(WorldFormatError, match=r'takes kind or material, not both') as raised:
        parse_world(raw)
    assert str(raised.value).startswith(f'{path}: ')


def test_outline_false_is_rejected() -> None:
    with pytest.raises(WorldFormatError, match=r'^levels\[0\]\.zones\[0\]\.walls: outline must be true'):
        parse_world(_level(zones=[_square('a', walls={'outline': False})]))


DEFAULTS_WORLD = """\
format: arena-world/2
defaults:
  floor: Marble
  ceiling: {height: 3.0, material: Concrete_Smooth, cast_shadows: true}
  wall: {material: Brick}
  door: {kind: hinged, height: 2.2}
levels:
- id: '0'
  zones:
  - {name: inherits, polygon: [[0, 0], [4, 0], [4, 4], [0, 4]], walls: outline}
  - name: overrides
    polygon: [[4, 0], [8, 0], [8, 4], [4, 4]]
    floor: Porcelain_Tile_4
    ceiling: {height: 2.5}
    walls: {outline: true, open_to: [inherits], kind: glass_walls}
  - {name: bare, polygon: [[8, 0], [12, 0], [12, 4], [8, 4]], floor: false, ceiling: false}
  walls:
  - {path: [[0, 4], [0, 8]]}
  - {path: [[4, 4], [4, 8]], material: Plaster_Wall}
  doors:
  - {name: inherited_door, span: [[1, 0], [2, 0]]}
  - {name: own_door, span: [[5, 0], [6, 0]], kind: sliding, height: 2.0}
"""


def test_defaults_apply_to_items_without_their_own_values() -> None:
    level = _parse(DEFAULTS_WORLD).levels['0']
    inherits = level.zones[0]
    assert inherits.floor == MaterialIdentifier('Marble')
    assert inherits.ceiling is not None
    assert (inherits.ceiling.height, inherits.ceiling.material, inherits.ceiling.cast_shadows) == (3.0, MaterialIdentifier('Concrete_Smooth'), True)
    assert inherits.walls is not None and inherits.walls.material == MaterialIdentifier('Brick') and inherits.walls.kind == ''
    assert level.walls[0].material == MaterialIdentifier('Brick')
    door = level.doors[0]
    assert (door.kind, door.height) == ('hinged', 2.2)


def test_items_override_defaults() -> None:
    level = _parse(DEFAULTS_WORLD).levels['0']
    overrides, bare = level.zones[1], level.zones[2]
    assert overrides.floor == MaterialIdentifier('Porcelain_Tile_4')
    assert overrides.ceiling is not None
    assert (overrides.ceiling.height, overrides.ceiling.material, overrides.ceiling.cast_shadows) == (2.5, MaterialIdentifier('Concrete_Smooth'), True)
    assert overrides.walls is not None and (overrides.walls.kind, overrides.walls.material) == ('glass_walls', None)
    assert (bare.floor, bare.ceiling) == (None, None)
    assert level.walls[1].material == MaterialIdentifier('Plaster_Wall')
    door = level.doors[1]
    assert (door.kind, door.height) == ('sliding', 2.0)


def test_ceiling_default_false_leaves_zones_without_ceiling_unless_they_set_one() -> None:
    raw = {'format': FORMAT, 'defaults': {'ceiling': False, 'floor': False}, 'levels': [{'id': '0', 'zones': [_square('plain'), _square('lidded', 4, ceiling=True, floor='Marble')]}]}
    plain, lidded = parse_world(raw).levels['0'].zones
    assert (plain.floor, plain.ceiling) == (None, None)
    assert lidded.floor == MaterialIdentifier('Marble')
    assert lidded.ceiling is not None and lidded.ceiling.height is None


@pytest.mark.parametrize(
    ('levels', 'message'),
    [
        ([{'id': '0', 'zones': [_square('room')]}, {'id': '1', 'zones': [_square('room')]}], "name 'room' is used by a zone and a zone"),
        ([{'id': '0', 'zones': [_square('room')], 'doors': [{'name': 'room', 'span': [[1, 0], [2, 0]]}]}], "name 'room' is used by a zone and a door"),
        ([{'id': '0', 'objects': [{'name': 'thing', 'model': 'Desk', 'pose': [1, 1]}]}, {'id': '1', 'sounds': [{'name': 'thing', 'asset_id': 'alarm', 'position': [1, 1]}]}], "name 'thing' is used by a obj and a sound"),
    ],
)
def test_names_are_unique_across_the_world(levels: list[dict], message: str) -> None:
    world = parse_world({'format': FORMAT, 'levels': levels})
    with pytest.raises(ValueError, match=message):
        world.validate()


def test_open_to_an_unknown_zone_is_rejected() -> None:
    world = parse_world(_level(zones=[_square('a', walls={'outline': True, 'open_to': ['nowhere']})]))
    with pytest.raises(ValueError, match=r"zone 'a' is open to unknown zones \['nowhere'\]"):
        world.validate()
