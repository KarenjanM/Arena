"""Read a format 2 world mapping into a World, defaults applied, every error prefixed with its YAML path."""

from __future__ import annotations

import typing
from collections.abc import Callable, Mapping

import attrs

from arena_simulation_setup.shared import Door, DynamicObstacle, Elevator, Light, Obstacle, Schedule, Signal, Sound
from arena_simulation_setup.utils.cattrs import ArenaConverter, converter

from .model import FORMAT, Level, WallPath, World, WorldMicrophone, Zone, ZoneCeiling, ZoneWalls

T = typing.TypeVar('T')

WORLD_KEYS = frozenset({'format', 'defaults', 'schedules', 'signals', 'levels'})
DEFAULT_KEYS = frozenset({'floor', 'ceiling', 'wall', 'door'})
LEVEL_KEYS = frozenset({'id', 'zones', 'walls', 'doors', 'elevators', 'objects', 'humans', 'lights', 'sounds', 'microphones', 'pedestrian_walls'})
ZONE_KEYS = frozenset({'name', 'polygon', 'floor', 'ceiling', 'walls', 'semantics'})
CEILING_KEYS = frozenset({'height', 'material', 'cast_shadows', 'lights'})
ZONE_WALL_KEYS = frozenset({'outline', 'open_to', 'kind', 'material'})
WALL_KEYS = frozenset({'path', 'closed', 'kind', 'material'})
WALL_STYLE_KEYS = frozenset({'kind', 'material'})
MICROPHONE_KEYS = frozenset({'zone', 'placement', 'position', 'frame', 'index'})
LINK_KEYS = frozenset({'id', 'from', 'prefix'})


class WorldFormatError(ValueError):
    """A world file that does not follow the format, with the YAML path of the offending entry."""


def _at(path: str, fn: Callable[[], T]) -> T:
    try:
        return fn()
    except WorldFormatError:
        raise
    except Exception as exc:
        raise WorldFormatError(f'{path}: {exc}') from exc


def _keys(path: str, value: object, allowed: frozenset[str], required: frozenset[str] = frozenset()) -> Mapping:
    if not isinstance(value, Mapping):
        raise WorldFormatError(f'{path}: expected a mapping, got {value!r}')
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise WorldFormatError(f'{path}: unknown keys {unknown}, expected some of {sorted(allowed)}')
    missing = sorted(required - set(value))
    if missing:
        raise WorldFormatError(f'{path}: missing keys {missing}')
    return value


def _items(path: str, value: object) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise WorldFormatError(f'{path}: expected a list, got {value!r}')
    return value


def _structure(c: ArenaConverter, value: object, cls: type[T]) -> T:
    return typing.cast(T, c.structure(value, cls))


def _wall_style(path: str, value: object) -> dict:
    style = dict(_keys(path, value, WALL_STYLE_KEYS))
    if 'kind' in style and 'material' in style:
        raise WorldFormatError(f'{path}: takes kind or material, not both')
    return style


def _ceiling(path: str, value: object, default: object) -> ZoneCeiling | None:
    if value is False or value is None:
        return None
    base = dict(default) if isinstance(default, Mapping) else {}
    if value is True:
        value = {}
    merged = {**base, **_keys(path, value, CEILING_KEYS)}
    return _at(path, lambda: ZoneCeiling(**merged))


def _zone(path: str, raw: object, defaults: Mapping) -> Zone:
    raw = _keys(path, raw, ZONE_KEYS, frozenset({'name'}))
    fields: dict = {'name': str(raw['name']), 'polygon': raw.get('polygon') or [], 'semantics': raw.get('semantics') or []}
    floor = raw['floor'] if 'floor' in raw else defaults.get('floor', ...)
    if floor is not ...:
        fields['floor'] = None if floor in (False, None, '') else floor
    if 'ceiling' in raw:
        fields['ceiling'] = _ceiling(f'{path}.ceiling', raw['ceiling'], defaults.get('ceiling'))
    elif 'ceiling' in defaults:
        fields['ceiling'] = _ceiling(f'{path}.ceiling', defaults['ceiling'] if defaults['ceiling'] is not False else False, None)
    walls = raw.get('walls')
    if walls is not None and walls is not False:
        style = _wall_style('defaults.wall', defaults.get('wall') or {})
        if walls == 'outline' or walls is True:
            fields['walls'] = ZoneWalls(**style)
        else:
            walls = dict(_keys(f'{path}.walls', walls, ZONE_WALL_KEYS))
            if walls.pop('outline', True) is not True:
                raise WorldFormatError(f'{path}.walls: outline must be true, drop walls to draw none')
            own = _wall_style(f'{path}.walls', {key: walls.pop(key) for key in ('kind', 'material') if key in walls})
            fields['walls'] = _at(f'{path}.walls', lambda: ZoneWalls(**{**(own or style), **walls}))
    return _at(path, lambda: Zone(**fields))


def _wall(path: str, raw: object, defaults: Mapping) -> WallPath:
    raw = dict(_keys(path, raw, WALL_KEYS, frozenset({'path'})))
    if 'kind' in raw and 'material' in raw:
        raise WorldFormatError(f'{path}: takes kind or material, not both')
    if 'kind' not in raw and 'material' not in raw:
        raw.update(_wall_style('defaults.wall', defaults.get('wall') or {}))
    return _at(path, lambda: WallPath(**raw))


def _named(path: str, raw: object, cls: type[T], c: ArenaConverter, default: Mapping | None = None) -> T:
    if not isinstance(raw, Mapping):
        raise WorldFormatError(f'{path}: expected a mapping, got {raw!r}')
    merged = {**(default or {}), **raw}
    return _at(path, lambda: _structure(c, merged, cls))


def parse_level(path: str, raw: object, defaults: Mapping, c: ArenaConverter = converter) -> Level:
    raw = _keys(path, raw, LEVEL_KEYS, frozenset({'id'}))
    level_id = str(raw['id'])
    door_default = dict(defaults.get('door') or {})
    return Level(
        id=level_id,
        zones=[_zone(f'{path}.zones[{i}]', item, defaults) for i, item in enumerate(_items(f'{path}.zones', raw.get('zones')))],
        walls=[_wall(f'{path}.walls[{i}]', item, defaults) for i, item in enumerate(_items(f'{path}.walls', raw.get('walls')))],
        doors=[_named(f'{path}.doors[{i}]', item, Door, c, door_default) for i, item in enumerate(_items(f'{path}.doors', raw.get('doors')))],
        elevators=[_named(f'{path}.elevators[{i}]', item, Elevator, c) for i, item in enumerate(_items(f'{path}.elevators', raw.get('elevators')))],
        objects=[_named(f'{path}.objects[{i}]', item, Obstacle, c) for i, item in enumerate(_items(f'{path}.objects', raw.get('objects')))],
        humans=[_named(f'{path}.humans[{i}]', item, DynamicObstacle, c) for i, item in enumerate(_items(f'{path}.humans', raw.get('humans')))],
        lights=[_named(f'{path}.lights[{i}]', item, Light, c) for i, item in enumerate(_items(f'{path}.lights', raw.get('lights')))],
        sounds=[_named(f'{path}.sounds[{i}]', item, Sound, c) for i, item in enumerate(_items(f'{path}.sounds', raw.get('sounds')))],
        microphones=[_at(f'{path}.microphones[{i}]', lambda item=item, i=i: WorldMicrophone(**_keys(f'{path}.microphones[{i}]', item, MICROPHONE_KEYS, frozenset({'zone', 'placement', 'position'})))) for i, item in enumerate(_items(f'{path}.microphones', raw.get('microphones')))],
        pedestrian_walls=bool(raw.get('pedestrian_walls', True)),
    )


def parse_defaults(raw: object) -> dict:
    defaults = dict(_keys('defaults', raw or {}, DEFAULT_KEYS))
    if 'wall' in defaults:
        _wall_style('defaults.wall', defaults['wall'])
    if 'ceiling' in defaults and defaults['ceiling'] is not False:
        _ceiling('defaults.ceiling', defaults['ceiling'], None)
    if 'door' in defaults:
        defaults['door'] = dict(_keys('defaults.door', defaults['door'], _door_default_keys()))
    return defaults


def _door_default_keys() -> frozenset[str]:
    import attrs

    return frozenset(field.name for field in attrs.fields(Door) if field.name not in ('name', 'start', 'end', 'extra', 'semantics'))


def _linked(path: str, raw: Mapping, link: Callable[[str], Level] | None) -> Level:
    raw = _keys(path, raw, LINK_KEYS, frozenset({'from'}))
    source = str(raw['from'])
    if link is None:
        raise WorldFormatError(f'{path}.from: a linked level is read through WorldView, which knows where {source!r} lives')
    try:
        level = link(source)
    except (OSError, LookupError, ValueError) as exc:
        raise WorldFormatError(f'{path}.from: {exc}') from exc
    prefix = str(raw.get('prefix', ''))
    return attrs.evolve(level.prefixed(prefix) if prefix else level, id=str(raw.get('id', level.id)), source=source, prefix=prefix)


def parse_world(raw: object, c: ArenaConverter = converter, link: Callable[[str], Level] | None = None) -> World:
    """A World from the mapping of a format 2 world.yaml, its `from:` levels read through `link`."""
    raw = _keys('', raw, WORLD_KEYS, frozenset({'format', 'levels'}))
    if raw['format'] != FORMAT:
        raise WorldFormatError(f'format: expected {FORMAT!r}, got {raw["format"]!r}')
    defaults = parse_defaults(raw.get('defaults'))
    levels: dict[str, Level] = {}
    for index, item in enumerate(_items('levels', raw['levels'])):
        level = _linked(f'levels[{index}]', item, link) if isinstance(item, Mapping) and 'from' in item else parse_level(f'levels[{index}]', item, defaults, c)
        if level.id in levels:
            raise WorldFormatError(f'levels[{index}]: level id {level.id!r} appears twice')
        levels[level.id] = level
    return World(
        levels=levels,
        defaults=defaults,
        schedules=[_named(f'schedules[{i}]', item, Schedule, c) for i, item in enumerate(_items('schedules', raw.get('schedules')))],
        signals=[_named(f'signals[{i}]', item, Signal, c) for i, item in enumerate(_items('signals', raw.get('signals')))],
    )
