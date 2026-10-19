from __future__ import annotations

import re
import typing

import attrs

from arena_simulation_setup.utils.cattrs import Parseable, Serializable

_ENV_PREFIX = re.compile(r'^env_\d+/')
_ENTITY_ATOM = re.compile(r'^(\S+)\.(\S+) == (\S+)$')
_MEMBERSHIP_ATOM = re.compile(r'^(\S+) in (\S+)$')
_WITHIN_ATOM = re.compile(r'^(\S+) within (\S+) of (\S+)$')
_SEES_ATOM = re.compile(r'^(\S+) sees (\S+?)(?: within (\S+))?$')
_PLAYS_ATOM = re.compile(r'^(\S+) plays (\S+)$')
_FOR_SUFFIX = re.compile(r'^(.*\S) for (\S+)$')
_AND = ' and '
_OR = ' or '
_NOT_PREFIX = 'not '

SEES_RANGE_M = 12.0

_OPS: frozenset[str] = frozenset({'always', 'never', 'eventually', 'before', 'never_during'})
_BINARY_OPS: frozenset[str] = frozenset({'before', 'never_during'})
_ON_FAILURE: frozenset[str] = frozenset({'continue', 'stop_task', 'abort_episode'})

OnFailure = typing.Literal['continue', 'stop_task', 'abort_episode']


@attrs.frozen
class EntityAtom:
    """Entity field test, `<entity>.<field> == <value>`."""

    entity: str
    field: str
    value: str


@attrs.frozen
class MembershipAtom:
    """Zone membership test, `<subject> in <zone>`."""

    subject: str
    zone: str


@attrs.frozen
class WithinAtom:
    """Proximity test, `<subject> within <radius> of <other>`, both subjects robots or peds."""

    subject: str
    radius: float
    other: str


@attrs.frozen
class SeesAtom:
    """Perception test, `<robot> sees <ped> [within <range>]`: within range and no wall cell on the segment between them."""

    subject: str
    other: str
    range: float = SEES_RANGE_M


@attrs.frozen
class PlaysAtom:
    """Gesture test, `<ped> plays <clip>`, the clip is among the gestures the pedestrian publishes."""

    subject: str
    clip: str


@attrs.frozen
class NotAtom:
    """Negation, `not <atom>`."""

    atom: EntityAtom | MembershipAtom | WithinAtom | SeesAtom | PlaysAtom


Atom = EntityAtom | MembershipAtom | WithinAtom | SeesAtom | PlaysAtom | NotAtom


@attrs.frozen
class UntilClause:
    """One clause of an until, `<atom>[ and <atom>...][ for <seconds>]`: every atom holding together for the dwell."""

    atoms: tuple[Atom, ...]
    dwell: float = 0.0


@attrs.frozen
class Until:
    """Phase exit test, `<clause>[ or <clause>...]`: any clause holding."""

    clauses: tuple[UntilClause, ...]


def _reject_env_prefix(component: str, label: str) -> None:
    if _ENV_PREFIX.match(component):
        raise ValueError(f'atom {label} must be bare (env-stripped), got {component!r}')


def parse_atom(value: str) -> Atom:
    """Total parser over the atom surface forms, raises ValueError on any other shape."""
    text = value.strip()

    if text.startswith(_NOT_PREFIX):
        inner = parse_atom(text[len(_NOT_PREFIX) :])
        if isinstance(inner, NotAtom):
            raise ValueError(f'atom negates a negation: {value!r}')
        return NotAtom(atom=inner)

    match = _ENTITY_ATOM.match(text)
    if match:
        entity, field, atom_value = match.groups()
        _reject_env_prefix(entity, 'entity')
        return EntityAtom(entity=entity, field=field, value=atom_value)

    match = _MEMBERSHIP_ATOM.match(text)
    if match:
        subject, zone = match.groups()
        _reject_env_prefix(subject, 'subject')
        _reject_env_prefix(zone, 'zone')
        return MembershipAtom(subject=subject, zone=zone)

    match = _WITHIN_ATOM.match(text)
    if match:
        subject, radius, other = match.groups()
        _reject_env_prefix(subject, 'subject')
        _reject_env_prefix(other, 'subject')
        return WithinAtom(subject=subject, radius=_non_negative(radius, 'radius', value), other=other)

    match = _SEES_ATOM.match(text)
    if match:
        subject, other, range_text = match.groups()
        _reject_env_prefix(subject, 'subject')
        _reject_env_prefix(other, 'subject')
        return SeesAtom(subject=subject, other=other, range=_non_negative(range_text, 'range', value) if range_text is not None else SEES_RANGE_M)

    match = _PLAYS_ATOM.match(text)
    if match:
        subject, clip = match.groups()
        _reject_env_prefix(subject, 'subject')
        return PlaysAtom(subject=subject, clip=clip)

    raise ValueError(f'atom does not match any surface form: {value!r}')


def _non_negative(text: str, label: str, value: str) -> float:
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f'atom {label} must be a number: {value!r}') from None
    if number < 0:
        raise ValueError(f'atom {label} must not be negative: {value!r}')
    return number


def parse_until(value: str) -> Until:
    """Parser of the `until` form: clauses joined by ` or `, each atoms joined by ` and ` with an optional trailing ` for <seconds>` dwell."""
    return Until(clauses=tuple(_parse_clause(part, value) for part in value.strip().split(_OR)))


def _parse_clause(text: str, value: str) -> UntilClause:
    text = text.strip()
    if not text:
        raise ValueError(f'until has an empty clause: {value!r}')
    dwell = 0.0
    match = _FOR_SUFFIX.match(text)
    if match:
        text, dwell_text = match.groups()
        dwell = _non_negative(dwell_text, 'dwell', value)
    return UntilClause(atoms=tuple(parse_atom(part) for part in text.split(_AND)), dwell=dwell)


def render_atom(atom: Atom) -> str:
    """Surface form of a parsed atom, the inverse of parse_atom."""
    if isinstance(atom, NotAtom):
        return f'not {render_atom(atom.atom)}'
    if isinstance(atom, EntityAtom):
        return f'{atom.entity}.{atom.field} == {atom.value}'
    if isinstance(atom, MembershipAtom):
        return f'{atom.subject} in {atom.zone}'
    if isinstance(atom, SeesAtom):
        return f'{atom.subject} sees {atom.other} within {atom.range:g}'
    if isinstance(atom, PlaysAtom):
        return f'{atom.subject} plays {atom.clip}'
    return f'{atom.subject} within {atom.radius:g} of {atom.other}'


def render_until(until: Until) -> str:
    """Surface form of a parsed until, the inverse of parse_until."""
    return _OR.join(_render_clause(clause) for clause in until.clauses)


def _render_clause(clause: UntilClause) -> str:
    text = _AND.join(render_atom(atom) for atom in clause.atoms)
    return f'{text} for {clause.dwell:g}' if clause.dwell > 0 else text


def atom_subjects(atom: Atom) -> frozenset[str]:
    """Robot or ped names the atom refers to, `robot` included."""
    if isinstance(atom, NotAtom):
        return atom_subjects(atom.atom)
    if isinstance(atom, EntityAtom):
        return frozenset()
    if isinstance(atom, (MembershipAtom, PlaysAtom)):
        return frozenset({atom.subject})
    return frozenset({atom.subject, atom.other})


@attrs.define
class EpisodeCondition(Parseable, Serializable):
    """One clause: one operator over one or two atoms in surface form."""

    op: typing.Literal['always', 'never', 'eventually', 'before', 'never_during']
    p: str
    q: str | None = None
    text: str = ''
    on_failure: OnFailure = 'continue'

    def __attrs_post_init__(self) -> None:
        binary = self.op in _BINARY_OPS
        if binary and self.q is None:
            raise ValueError(f"clause op {self.op!r} requires a 'q' atom")
        if not binary and self.q is not None:
            raise ValueError(f"clause op {self.op!r} does not take a 'q' atom")
        if self.on_failure not in _ON_FAILURE:
            raise ValueError(f'clause on_failure must be one of {sorted(_ON_FAILURE)}, got {self.on_failure!r}')
        parse_atom(self.p)
        if self.q is not None:
            parse_atom(self.q)

    @classmethod
    def parse(cls, value: dict) -> 'EpisodeCondition':
        value = dict(value)
        op = value.pop('op', None)
        if op not in _OPS:
            raise ValueError(f'unknown clause op: {op!r}')

        p = value.pop('p', None)
        if not isinstance(p, str):
            raise ValueError(f"clause 'p' must be a string atom: {p!r}")
        parse_atom(p)

        q = value.pop('q', None)
        if op in _BINARY_OPS:
            if not isinstance(q, str):
                raise ValueError(f"clause op {op!r} requires a 'q' atom")
            parse_atom(q)
        elif q is not None:
            raise ValueError(f"clause op {op!r} does not take a 'q' atom")

        text = str(value.pop('text', ''))
        on_failure = typing.cast('OnFailure', str(value.pop('on_failure', 'continue')))

        if value:
            raise ValueError(f'unknown clause keys: {sorted(value)}')

        return cls(op=op, p=p, q=q, text=text, on_failure=on_failure)

    def serialize(self) -> dict:
        result: dict = {'op': self.op, 'p': self.p}
        if self.q is not None:
            result['q'] = self.q
        if self.text:
            result['text'] = self.text
        if self.on_failure != 'continue':
            result['on_failure'] = self.on_failure
        return result


__all__ = [
    'Atom',
    'EntityAtom',
    'EpisodeCondition',
    'MembershipAtom',
    'NotAtom',
    'OnFailure',
    'PlaysAtom',
    'SEES_RANGE_M',
    'SeesAtom',
    'Until',
    'UntilClause',
    'WithinAtom',
    'atom_subjects',
    'parse_atom',
    'parse_until',
    'render_atom',
    'render_until',
]
