"""Lifting the legacy airchitect `graph.yaml` scene graph into a KG scene."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from .scene import Scene

GRAPH_USES = ("core@2",)
DIRECTIONS = {"left": "left_of", "right": "right_of", "above": "above", "below": "below"}

_TOP_KEYS = ("prompt", "nodes", "edges")
_NODE_KEYS = ("id", "room_type", "assets", "description", "attributes")
_EDGE_KEYS = ("id", "source", "target", "attributes")


def parse_direction(attribute: str) -> str | None:
    """`position: left of` to `left_of`, None for an attribute that is no position."""
    value = attribute.split(":", 1)[-1].strip().lower().removesuffix(" of").strip()
    return DIRECTIONS.get(value)


def _item(raw: object, where: str, allowed: tuple[str, ...], required: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    for key in raw:
        if key not in allowed:
            raise ValueError(f"{where}: unknown key {key!r}")
    for key in required:
        if not isinstance(raw.get(key), str):
            raise ValueError(f"{where}: {key!r} must be a string")
    return raw


def _strings(raw: object, where: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        raise ValueError(f"{where}: expected a list of strings")
    return raw


def _door(edge: Mapping[str, Any], doors: Collection[str] | None) -> str | None:
    candidates = [f"door_{edge['id']}", f"door_{edge['source']}__{edge['target']}", f"door_{edge['target']}__{edge['source']}"]
    if doors is None:
        return f"door:{candidates[0]}"
    return next((f"door:{name}" for name in candidates if name in doors), None)


def lift_graph(data: Mapping[str, Any], doors: Collection[str] | None = None) -> Scene:
    """The `core@2` scene of a single-level graph: rooms, wish-list objects, door connections and positions.

    Given the world's door names, an edge connects through the door named `door_<edge id>` or `door_<a>__<b>`, and without one it states no connection.
    """
    top = _item(data, "top level", _TOP_KEYS, ())
    scene = Scene(GRAPH_USES)
    for i, raw in enumerate(top.get("nodes") or []):
        where = f"nodes[{i}]"
        node = _item(raw, where, _NODE_KEYS, ("id", "room_type"))
        room = f"room:{node['id']}"
        props = {"type": node["room_type"]}
        description = node.get("description") or ""
        if not isinstance(description, str):
            raise ValueError(f"{where}.description: expected a string")
        if description:
            props["description"] = description
        scene.add_entity(room, props)
        for attribute in _strings(node.get("attributes"), f"{where}.attributes"):
            scene.add_fact(room, "attr", attribute)
        for n, asset in enumerate(_strings(node.get("assets"), f"{where}.assets")):
            obj = f"obj:{node['id']}_{asset.replace(' ', '_')}_{n}"
            if obj in scene.entities:
                raise ValueError(f"{where}.assets[{n}]: object id {obj!r} already taken")
            scene.add_entity(obj, {"label": asset, "placed": False})
            scene.add_fact(room, "contains", obj)
    for i, raw in enumerate(top.get("edges") or []):
        where = f"edges[{i}]"
        edge = _item(raw, where, _EDGE_KEYS, ("id", "source", "target"))
        door = _door(edge, doors)
        source, target = f"room:{edge['source']}", f"room:{edge['target']}"
        if door is not None:
            scene.add_entity(door)
            scene.add_fact(source, "connects", target, {"door": door})
        for j, attribute in enumerate(_strings(edge.get("attributes"), f"{where}.attributes")):
            direction = parse_direction(attribute)
            if direction is None:
                raise ValueError(f"{where}.attributes[{j}]: {attribute!r} is no position (left of, right of, above, below)")
            scene.add_fact(source, direction, target)
    prompt = top.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, str):
            raise ValueError("prompt: expected a string")
        scene.add_entity("floor:0")
        scene.add_fact("floor:0", "note", prompt)
    return scene
