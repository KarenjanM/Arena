"""The KG scene of one world: entities, facts and constraints over a vocabulary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import attrs

from .vocab import ANY, LITERAL, Vocabulary

Value = str | int | float | bool
GENERATED = "generated"
DERIVED = "derived"
SOURCES = (GENERATED, DERIVED)
DEFAULT_USES = ("core@2",)


def kind_of(id: str) -> str:
    """The `<kind>` of an id `<kind>:<name>`."""
    kind, sep, name = id.partition(":")
    if not sep or not kind or not name:
        raise ValueError(f"id {id!r} is not of the form <kind>:<name>")
    return kind


@attrs.define
class Entity:
    id: str
    kind: str
    props: dict[str, Value] = attrs.field(factory=dict)

    @property
    def name(self) -> str:
        return self.id.split(":", 1)[1]


@attrs.define
class Fact:
    subject: str
    verb: str
    object: Value
    params: dict[str, Value] = attrs.field(factory=dict)
    source: str = GENERATED

    @property
    def key(self) -> tuple:
        return fact_key(self.subject, self.verb, self.object, self.params)


def fact_key(subject: str, verb: str, object: Value, params: Mapping[str, Value] | None = None) -> tuple:
    """The identity of a fact: subject, verb and object, plus the door for a `connects`, so two doors between two rooms are two facts."""
    if verb == 'connects' and params and 'door' in params:
        return (subject, verb, object, params['door'])
    return (subject, verb, object)


@attrs.define
class Constraint:
    id: str
    kind: str
    scope: list[str] = attrs.field(factory=list)
    params: dict[str, Value] = attrs.field(factory=dict)
    hard: bool = True


class Scene:
    """Entities keyed by id, facts keyed by `fact_key`, constraints keyed by id."""

    def __init__(self, uses: Sequence[str] = DEFAULT_USES):
        self.vocab = Vocabulary(uses)
        self.entities: dict[str, Entity] = {}
        self._facts: dict[tuple, Fact] = {}
        self.constraints: dict[str, Constraint] = {}

    @property
    def uses(self) -> list[str]:
        return list(self.vocab.uses)

    @property
    def facts(self) -> list[Fact]:
        return list(self._facts.values())

    def add_entity(self, id: str, props: Mapping[str, Value] | None = None) -> Entity:
        """Declare `id`, or merge `props` into the existing entity."""
        entity = self.entities.get(id)
        if entity is None:
            entity = Entity(id=id, kind=kind_of(id), props=dict(props or {}))
            self.entities[id] = entity
        else:
            entity.props.update(props or {})
        return entity

    def get_entity(self, id: str) -> Entity | None:
        return self.entities.get(id)

    def remove_entity(self, id: str) -> tuple[Entity | None, list[Fact], list[Constraint]]:
        """Remove `id` with every fact and constraint that references it."""
        entity = self.entities.pop(id, None)
        facts = [f for f in self._facts.values() if id in (f.subject, f.object)]
        for f in facts:
            del self._facts[f.key]
        constraints = [c for c in self.constraints.values() if id in c.scope]
        for c in constraints:
            del self.constraints[c.id]
        return entity, facts, constraints

    def add_fact(self, subject: str, verb: str, object: Value, params: Mapping[str, Value] | None = None, source: str = GENERATED) -> Fact:
        """Add a fact, or merge `params` into the fact with the same key."""
        if source not in SOURCES:
            raise ValueError(f"fact source {source!r} is not one of {SOURCES}")
        self._declare_auto(object)
        existing = self._facts.get(fact_key(subject, verb, object, params))
        if existing is not None:
            existing.params.update(params or {})
            return existing
        fact = Fact(subject=subject, verb=verb, object=object, params=dict(params or {}), source=source)
        self._facts[fact.key] = fact
        return fact

    def get_fact(self, subject: str, verb: str, object: Value, params: Mapping[str, Value] | None = None) -> Fact | None:
        return self._facts.get(fact_key(subject, verb, object, params))

    def remove_fact(self, subject: str, verb: str, object: Value, params: Mapping[str, Value] | None = None) -> Fact | None:
        return self._facts.pop(fact_key(subject, verb, object, params), None)

    def stated(self) -> list[Fact]:
        return [f for f in self._facts.values() if f.source == GENERATED]

    def drop_derived(self) -> list[Fact]:
        """Remove and return every derived fact."""
        derived = [f for f in self._facts.values() if f.source == DERIVED]
        for f in derived:
            del self._facts[f.key]
        return derived

    def add_constraint(self, id: str, kind: str, scope: Sequence[str], params: Mapping[str, Value] | None = None, hard: bool = True) -> Constraint:
        for ref in scope:
            self._declare_auto(ref)
        constraint = Constraint(id=id, kind=kind, scope=list(scope), params=dict(params or {}), hard=hard)
        self.constraints[id] = constraint
        return constraint

    def get_constraint(self, id: str) -> Constraint | None:
        return self.constraints.get(id)

    def remove_constraint(self, id: str) -> Constraint | None:
        return self.constraints.pop(id, None)

    def _declare_auto(self, ref: Value) -> None:
        if isinstance(ref, str) and self.vocab.is_entity_ref(ref) and ref not in self.entities and self.vocab.is_auto(kind_of(ref)):
            self.add_entity(ref)

    def validate(self) -> list[str]:
        """Every vocabulary violation: kinds, props, verb signatures, constraint arity, params and references."""
        problems = []
        for e in self.entities.values():
            if e.kind not in self.vocab.nodes:
                problems.append(f"entity {e.id}: unknown kind {e.kind!r}")
                continue
            declared = self.vocab.props(e.kind)
            for prop in e.props:
                if prop not in declared:
                    problems.append(f"entity {e.id}: undeclared prop {prop!r}")
        for f in self._facts.values():
            spec = self.vocab.verbs.get(f.verb)
            if spec is None:
                problems.append(f"fact {(f.subject, f.verb, f.object)}: unknown verb {f.verb!r}")
                continue
            subject = self.entities.get(f.subject)
            if subject is None:
                problems.append(f"fact {(f.subject, f.verb, f.object)}: unknown subject {f.subject!r}")
            elif spec["from"] != ANY and subject.kind not in spec["from"]:
                problems.append(f"fact {(f.subject, f.verb, f.object)}: subject kind {subject.kind!r} not allowed for {f.verb!r}")
            if spec["to"] != LITERAL:
                obj = self.entities.get(str(f.object))
                if obj is None:
                    problems.append(f"fact {(f.subject, f.verb, f.object)}: unknown object {f.object!r}")
                elif spec["to"] != ANY and obj.kind not in spec["to"]:
                    problems.append(f"fact {(f.subject, f.verb, f.object)}: object kind {obj.kind!r} not allowed for {f.verb!r}")
            for p, v in f.params.items():
                if self.vocab.is_entity_ref(v) and v not in self.entities:
                    problems.append(f"fact {(f.subject, f.verb, f.object)}: param {p}={v!r} references unknown entity")
        for c in self.constraints.values():
            spec = self.vocab.constraints.get(c.kind)
            if spec is None:
                problems.append(f"constraint {c.id}: unknown kind {c.kind!r}")
                continue
            if spec.get("arity") is not None and len(c.scope) != spec["arity"]:
                problems.append(f"constraint {c.id}: expects {spec['arity']} ids, got {len(c.scope)}")
            for ref in c.scope:
                if ref not in self.entities:
                    problems.append(f"constraint {c.id}: unknown id {ref!r} in scope")
            for p, v in c.params.items():
                if p not in spec.get("params", []):
                    problems.append(f"constraint {c.id}: undeclared param {p!r}")
                elif self.vocab.is_entity_ref(v) and v not in self.entities:
                    problems.append(f"constraint {c.id}: param {p}={v!r} references unknown entity")
        return problems
