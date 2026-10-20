from __future__ import annotations

import typing
from collections.abc import Mapping
from pathlib import Path
from types import NotImplementedType
from typing import TYPE_CHECKING, Union

import attrs
import cattrs
from typing_extensions import Self

from arena_simulation_setup.tree.assets.Human import HumanIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.utils.cattrs import (
    ArenaConverter,
    Parseable,
    Serializable,
    converter,
)
from arena_simulation_setup.utils.geometry import Pose, Position, Scale, Vector3
from arena_simulation_setup.utils.resolution import resolve_zone_point

from .light_state import ObjectLightSettings

if TYPE_CHECKING:
    import shapely


@attrs.define(auto_attribs=True, kw_only=True)
class Waypoint(Position):
    level: str = ''
    zone: str = ''

    @property
    def position(self) -> Position:
        return self

    @position.setter
    def position(self, value: Position) -> None:
        self.x = value.x
        self.y = value.y
        self.z = value.z

    @classmethod
    def from_any(cls, obj: Union[Position, Mapping, str, typing.Sequence[float], 'Waypoint']) -> 'Waypoint':
        """A waypoint from a zone name, a position, `[x, y[, z]]` or `{x, y[, z][, level]}`."""
        if isinstance(obj, str):
            return cls.from_position(pos=resolve_zone_point(obj), zone=obj)
        if isinstance(obj, Waypoint):
            return obj
        if isinstance(obj, Position):
            return cls.from_position(pos=obj)
        if isinstance(obj, Mapping):
            unknown = set(obj) - {'x', 'y', 'z', 'level'}
            if unknown:
                raise ValueError(f'waypoint has unknown keys {sorted(unknown)}')
            pos = Position.parse({k: obj[k] for k in ('x', 'y', 'z') if k in obj})
            return cls.from_position(pos=pos, level=str(obj.get('level', '')))
        return cls.from_position(pos=Position.parse(obj))

    @classmethod
    def from_position(cls, pos: Position, level: str = '', zone: str = '') -> 'Waypoint':
        return cls(x=pos.x, y=pos.y, z=pos.z, level=level, zone=zone)

    def __add__(self, other: Vector3) -> Waypoint | NotImplementedType:
        if isinstance(other, Position):
            return type(self).from_position(pos=super().__add__(other), level=self.level, zone=self.zone)
        return NotImplemented

    def __radd__(self, other: Position) -> Waypoint | NotImplementedType:
        if isinstance(other, Position):
            return type(self).from_position(pos=other + Position(x=self.x, y=self.y, z=self.z), level=self.level, zone=self.zone)
        return NotImplemented


@attrs.define(kw_only=True)
class Named(Parseable, Serializable):
    name: str
    extra: dict = attrs.field(factory=dict, eq=False)

    @property
    def sim_path(self) -> str:
        return self.extra.get('sim_path', self.name)

    @sim_path.setter
    def sim_path(self, value: str) -> None:
        self.extra['sim_path'] = str(value)

    @classmethod
    def parse(cls, value: dict) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError(f'{cls.__name__} must be a mapping, got {value!r}')
        unknown = set(value) - {field.name for field in attrs.fields(cls)} | ({'extra'} & set(value))
        if unknown:
            raise ValueError(f"{cls.__name__.lower()} {value.get('name')!r} has unknown keys {sorted(unknown)}")
        return ArenaConverter.current().structure_attrs_fromdict(dict(value), cls)

    def serialize(self) -> dict:
        result = cattrs.gen.make_dict_unstructure_fn(type(self), converter, _cattrs_omit_if_default=True)(self)
        result.pop('extra', None)
        return result


@attrs.define(kw_only=True)
class Entity(Named, Parseable, Serializable):
    pose: Pose = attrs.field(converter=Pose.converter)
    model: ObjectIdentifier = attrs.field(converter=ObjectIdentifier.converter)
    level: str = ''

    included_from: Path | None = attrs.field(default=None, repr=False)

    @classmethod
    def parse(cls, value: dict) -> Self:
        if isinstance(value, Mapping) and 'included_from' in value:
            raise ValueError(f"{cls.__name__.lower()} {value.get('name')!r} has unknown keys ['included_from']")
        return super().parse(value)


HUMANSIM_OBJECT_KEYS = frozenset({'type', 'capacity', 'satisfies', 'interaction_radius', 'formation', 'seats'})


def _humansim_object(value: Mapping[str, object] | None) -> dict:
    value = dict(value or {})
    unknown = set(value) - HUMANSIM_OBJECT_KEYS
    if unknown:
        raise ValueError(f'humansim block has unknown keys {sorted(unknown)}, expected some of {sorted(HUMANSIM_OBJECT_KEYS)}')
    return value


@attrs.define
class Obstacle(Entity):
    scale: Scale | None = None
    light: ObjectLightSettings | None = None
    humansim: dict = attrs.field(factory=dict, converter=_humansim_object)

    async def footprint(self) -> shapely.Polygon | None:
        """Map-frame 2D footprint from the asset's annotated bounds, None when unannotated."""
        import shapely
        import shapely.affinity

        try:
            view = await self.model.resolve()
        except FileNotFoundError:
            return None
        bounds = view.bounds
        if bounds is None:
            return None
        poly = shapely.Polygon(bounds)
        poly = shapely.affinity.rotate(poly, self.pose.orientation.to_yaw(), origin=(0, 0), use_radians=True)
        return shapely.affinity.translate(poly, xoff=self.pose.position.x, yoff=self.pose.position.y)


converter.register_structure_hook_func(lambda t: t == dict | str | None, lambda v, _t: v)


def _waypoints_converter(value: object) -> list[Waypoint]:
    if not isinstance(value, list):
        raise TypeError("waypoints must be a list")
    return [Waypoint.from_any(wp) for wp in value]


WAYPOINT_MODES = ('repeat', 'reverse', 'once', 'random')


@attrs.define
class DynamicObstacle(Entity):
    model: HumanIdentifier = attrs.field(converter=HumanIdentifier.converter)
    waypoints: list[Waypoint] = attrs.field(factory=list, converter=_waypoints_converter)
    waypoint_mode: str = attrs.field(default='repeat', validator=attrs.validators.in_(WAYPOINT_MODES))
    velocity: float = attrs.field(converter=float, default=1.0)  # m/s
    agent: dict | str | None = None
    hunav: dict = attrs.field(factory=dict)


@attrs.define
class CustomDynamicObstacle(DynamicObstacle):
    """
    DynamicObstacles but with properties can be define in runtime
    """

    def __getattr__(self, name: str) -> object:
        """
        Allow access to dynamic attributes "attr_name" via self.attr_name
        """
        if name in self.extra:
            return self.extra[name]
        raise AttributeError(f"{name} not found")
