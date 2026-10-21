"""Zone roles: the world's KG facts first, substring rules for common names, `UNKNOWN` for the rest."""

from __future__ import annotations

import enum
import typing
from collections.abc import Mapping, Sequence
from typing import Any

if typing.TYPE_CHECKING:
    from arena_simulation_setup.kg import Fact, Scene

PACKAGE = "space@1"
VERB = "has_purpose"
KIND = "purpose"


class Role(enum.StrEnum):
    WORK = "work"
    BREAK = "break"
    TRANSIT = "transit"
    PRIVATE = "private"
    UNKNOWN = "unknown"


#: Substring rules, first match wins: BREAK comes first so `nurse_resting_room` and `wc_waiting` reach it.
_RULES: tuple[tuple[Role, tuple[str, ...]], ...] = (
    (
        Role.BREAK,
        (
            "waiting",
            "break_room",
            "resting",
            "lounge",
            "cafeteria",
            "dining",
            "kitchen",
            "kitchenette",
            "staff_room",
            "bathroom",
            "ensuite",
            "wc_",
            "restroom",
            "toilet",
            "balcony",
            "living_room",
            "phone_booth",
        ),
    ),
    (Role.PRIVATE, ("bedroom", "master_bed")),
    (
        Role.WORK,
        (
            "work",
            "office",
            "nurse_station",
            "reception",
            "patient_room",
            "patient_ward",
            "exam",
            "consultation",
            "operating",
            "procedure",
            "triage",
            "lab",
            "meeting_room",
            "conference",
            "pharmacy",
            "file_room",
            "utility",
            "storage",
            "workspace",
        ),
    ),
    (
        Role.TRANSIT,
        (
            "hallway",
            "corridor",
            "junction",
            "arm_",
            "entrance",
            "entry",
            "link_",
            "lobby",
            "destination",
            "empty_zone",
        ),
    ),
)


def classify(name: str) -> Role:
    lowered = name.strip().lower()
    for role, needles in _RULES:
        if any(n in lowered for n in needles):
            return role
    return Role.UNKNOWN


def proposed(fact: Fact) -> bool:
    """Whether a `has_purpose` fact is a model's proposal rather than a human's statement."""
    return fact.params.get("by") == "llm"


def stated_roles(scene: Scene | None) -> dict[str, Role]:
    """Zone name to role from the `has_purpose` facts of `scene`, a human's fact winning over a model's."""
    ranked: dict[str, tuple[set[Role], set[Role]]] = {}
    for fact in scene.facts if scene is not None else ():
        if fact.verb != VERB:
            continue
        kind, _, name = str(fact.object).partition(":")
        if kind != KIND or name not in [r.value for r in Role]:
            raise ValueError(f"{fact.subject} {VERB} {fact.object!r}: expected {KIND}:<{'|'.join(r.value for r in Role)}>")
        human, model = ranked.setdefault(fact.subject.partition(":")[2], (set(), set()))
        (model if proposed(fact) else human).add(Role(name))
    roles: dict[str, Role] = {}
    for zone, (human, model) in ranked.items():
        chosen = human or model
        if len(chosen) > 1:
            raise ValueError(f"room:{zone}: {VERB} states the roles {sorted(r.value for r in chosen)}, expected one")
        roles[zone] = next(iter(chosen))
    return roles


def classify_all(names: Sequence[str], overrides: Mapping[str, str] | None = None) -> dict[str, Role]:
    """`overrides` (roles stated in the world's KG) win over the rules."""
    resolved = {n: classify(n) for n in names}
    for name, role in (overrides or {}).items():
        if name in resolved:
            resolved[name] = Role(role)
    return resolved


def zones_by_role(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> dict[Role, list[str]]:
    names = [str(z["name"]) for z in zones if isinstance(z, Mapping) and z.get("name")]
    out: dict[Role, list[str]] = {r: [] for r in Role}
    for name, role in classify_all(names, overrides).items():
        out[role].append(name)
    return {r: sorted(v) for r, v in out.items()}
