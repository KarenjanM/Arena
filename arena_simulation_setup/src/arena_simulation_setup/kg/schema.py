"""JSON Schema of the `kg.yaml` file shape for a set of packages."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .io import FORMAT
from .scene import SOURCES
from .vocab import ANY, LITERAL, Vocabulary

NUMERIC_PROPS = frozenset({"area", "width", "velocity", "scale", "radius"})
BOOLEAN_PROPS = frozenset({"placed"})
NUMERIC_PARAMS = frozenset({"dist", "min", "rate", "max_concurrent"})
PARAM_ENUMS: dict[str, tuple[str, ...]] = {
    "dir": ("left_of", "right_of", "above", "below"),
    "wall": ("north", "east", "south", "west"),
}

_SCALAR: dict[str, Any] = {"type": ["string", "number", "boolean"]}
_SCALARS: dict[str, Any] = {"type": "object", "additionalProperties": _SCALAR}


def _prop_field(name: str) -> dict[str, Any]:
    if name in NUMERIC_PROPS:
        return {"type": "number"}
    if name in BOOLEAN_PROPS:
        return {"type": "boolean"}
    return {"type": "string"}


def _param_field(name: str) -> dict[str, Any]:
    if name in PARAM_ENUMS:
        return {"enum": list(PARAM_ENUMS[name])}
    return {"type": "number"} if name in NUMERIC_PARAMS else {"type": "string"}


def _id_pattern(kinds: Iterable[str]) -> str:
    """An id is `<kind>:<name>` and never contains a space."""
    return f"^({'|'.join(re.escape(k) for k in kinds)}):[^ ]+$"


def _entity_variant(kind: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {"id": {"type": "string", "pattern": _id_pattern([kind])}}
    if not spec.get("propless", False):
        properties["props"] = {
            "type": "object",
            "properties": {p: _prop_field(p) for p in spec.get("props", [])},
            "additionalProperties": False,
        }
    return {"type": "object", "properties": properties, "required": ["id"], "additionalProperties": False}


def _side(side: str | list[str], nodes: Mapping[str, Any]) -> dict[str, Any]:
    if side == LITERAL:
        return _SCALAR
    return {"type": "string", "pattern": _id_pattern(nodes if side == ANY else side)}


def _fact_variant(verb: str, spec: Mapping[str, Any], nodes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "subject": _side(spec["from"], nodes),
            "verb": {"const": verb},
            "object": _side(spec["to"], nodes),
            "params": _SCALARS,
            "source": {"enum": list(SOURCES)},
        },
        "required": ["subject", "verb", "object"],
        "additionalProperties": False,
    }


def _constraint_variant(kind: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    scope: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    arity = spec.get("arity")
    if arity is not None:
        scope["minItems"] = arity
        scope["maxItems"] = arity
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": _id_pattern(["con"])},
            "kind": {"const": kind},
            "scope": scope,
            "params": {
                "type": "object",
                "properties": {p: _param_field(p) for p in spec.get("params", [])},
                "additionalProperties": False,
            },
            "hard": {"type": "boolean"},
        },
        "required": ["id", "kind", "scope"],
        "additionalProperties": False,
    }


def schema_for(vocab: Vocabulary) -> dict[str, Any]:
    """One inlined draft 2020-12 schema over every kind, verb and constraint of `vocab`."""
    nodes = vocab.nodes
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{FORMAT} ({' '.join(p.key for p in vocab.packages)})",
        "type": "object",
        "properties": {
            "format": {"const": FORMAT},
            "uses": {"type": "array", "items": {"enum": [p.key for p in vocab.packages]}, "minItems": 1, "uniqueItems": True},
            "entities": {"type": "array", "items": {"oneOf": [_entity_variant(k, s) for k, s in nodes.items()]}},
            "facts": {"type": "array", "items": {"oneOf": [_fact_variant(v, s, nodes) for v, s in vocab.verbs.items()]}},
            "constraints": {"type": "array", "items": {"oneOf": [_constraint_variant(k, s) for k, s in vocab.constraints.items()]}},
        },
        "required": ["format", "uses"],
        "additionalProperties": False,
    }
