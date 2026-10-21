"""Model-proposed roles for the zone names `zones.classify` leaves `UNKNOWN`, merged into `kg.yaml` as `by: llm` facts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs

from ..zones import KIND, PACKAGE, VERB, Role, classify, fact_role, proposed
from .client import DEFAULT_MODEL, Unavailable, ask, prompt_sha256, strip_fence

SYSTEM = """\
You classify rooms in a robot navigation simulator by what people use them for.

Answer with one role per zone, from exactly this set:

  work     - where people are while working: desks, stations, treatment rooms, labs, meeting rooms
  break    - where people go when they are not working: kitchens, waiting areas, lounges, toilets
  transit  - corridors, junctions, lobbies, entrances. People pass through; nobody stays
  private  - bedrooms and similar. A robot has no business routing through them
  unknown  - the evidence does not support any of the above

`unknown` is a real answer and the right one when a zone's name is a placeholder and it holds
nothing that says what it is for. A wrong confident answer puts a crowd in a cupboard; an honest
`unknown` sends the zone to a human. Do not guess from the room number.

Reply with JSON only, no prose and no code fence:
{"<zone name>": {"role": "<role>", "why": "<at most 12 words>"}}
"""


@attrs.frozen(kw_only=True)
class Proposal:
    zone: str
    role: Role
    why: str = ""
    model: str = ""
    prompt_sha: str = ""

    @property
    def confident(self) -> bool:
        return self.role is not Role.UNKNOWN


def unknown_names(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> list[str]:
    """Zone names no rule and no stated role accounts for, sorted and deduplicated."""
    reviewed = set(overrides or {})
    return sorted({str(z["name"]) for z in zones if isinstance(z, Mapping) and z.get("name") and str(z["name"]) not in reviewed and classify(str(z["name"])) is Role.UNKNOWN})


def _evidence(zone: Mapping[str, Any]) -> dict[str, Any]:
    """Extent and contents of one zone."""
    corners = [(float(c["x"]), float(c["y"])) for c in zone.get("corners", []) if isinstance(c, Mapping)]
    statics = (zone.get("entities") or {}).get("static") or []
    contents = sorted({str(e.get("model", "")).split("/")[-1] for e in statics if isinstance(e, Mapping) and e.get("model")})
    out: dict[str, Any] = {"contents": contents or "nothing"}
    if corners:
        width = max(x for x, _ in corners) - min(x for x, _ in corners)
        height = max(y for _, y in corners) - min(y for _, y in corners)
        out["size_m"] = [round(width, 1), round(height, 1)]
    if doors := zone.get("doors"):
        out["doors"] = len(doors)
    return out


def build_prompt(zones: Sequence[Mapping[str, Any]], names: Sequence[str], *, world: str) -> str:
    """The user half of the request: the world's whole vocabulary, then the zones in question."""
    by_name = {str(z["name"]): z for z in zones if isinstance(z, Mapping) and z.get("name")}
    vocabulary = {name: classify(name).value for name in sorted(by_name)}
    asked = {name: _evidence(by_name[name]) for name in names if name in by_name}
    return f"World: {world}\n\nEvery zone this world declares, with the role already known where a rule covers it:\n{json.dumps(vocabulary, indent=2)}\n\nClassify only these, using what they contain and what the rest of the world suggests:\n{json.dumps(asked, indent=2)}\n"


def parse(text: str, asked: Sequence[str], *, model: str, sha: str) -> tuple[list[Proposal], list[str]]:
    """Proposals and complaints. Anything unasked-for or unrecognized is dropped, never coerced."""
    try:
        answer = json.loads(strip_fence(text))
    except json.JSONDecodeError as exc:
        return [], [f"response was not JSON: {exc}"]
    if not isinstance(answer, dict):
        return [], [f"response was {type(answer).__name__}, expected an object"]

    proposals: list[Proposal] = []
    complaints: list[str] = []
    wanted = set(asked)
    for name, entry in answer.items():
        if name not in wanted:
            complaints.append(f"{name}: not one of the zones asked about")
            continue
        role_name = entry.get("role") if isinstance(entry, Mapping) else entry
        try:
            role = Role(str(role_name))
        except ValueError:
            complaints.append(f"{name}: {role_name!r} is not a role")
            continue
        why = str(entry.get("why", "")).strip() if isinstance(entry, Mapping) else ""
        proposals.append(Proposal(zone=name, role=role, why=why, model=model, prompt_sha=sha))

    for missing in sorted(wanted - {p.zone for p in proposals}):
        complaints.append(f"{missing}: no answer returned")
    return proposals, complaints


def classify_tail(
    zones: Sequence[Mapping[str, Any]],
    *,
    world: str,
    names: Sequence[str],
    model: str | None = None,
) -> tuple[list[Proposal], list[str]]:
    """Ask the model about `names` of this world. One call, or none if `names` is empty."""
    asked = list(names)
    if not asked:
        return [], []
    user = build_prompt(zones, asked, world=world)
    sha = prompt_sha256(SYSTEM, user)
    return parse(ask(SYSTEM, user, model=model), asked, model=model or DEFAULT_MODEL, sha=sha)


def merge(proposals: Sequence[Proposal], kg: Path) -> tuple[int, list[str]]:
    """Write confident proposals into the `kg.yaml` at `kg`, never over a human's fact; returns (written, notes)."""
    from arena_simulation_setup.kg import Scene, dump, load
    from arena_simulation_setup.kg.scene import DEFAULT_USES
    from arena_simulation_setup.tree.World.write import to_yaml

    scene = load(kg) if kg.is_file() else Scene([*DEFAULT_USES, PACKAGE])
    if VERB not in scene.vocab.verbs:
        scene = load({**dump(scene), "uses": [*scene.uses, PACKAGE]})

    written = 0
    kept: list[str] = []
    for proposal in proposals:
        if not proposal.confident:
            kept.append(f"{proposal.zone}: model returned `unknown` - left for a human")
            continue
        room = f"room:{proposal.zone}"
        facts = [f for f in scene.facts if f.subject == room and fact_role(f) is not None]
        human = [str(f.object).partition(":")[2] for f in facts if not proposed(f)]
        if human:
            kept.append(f"{proposal.zone}: reviewed by a human as {', '.join(human)!r}")
            continue
        target = f"{KIND}:{proposal.role.value}"
        if any(f.object == target for f in facts):
            continue
        for f in facts:
            scene.remove_fact(f.subject, f.verb, f.object, f.params)
        scene.add_entity(room)
        params = {"by": "llm", "model": proposal.model, "prompt_sha": proposal.prompt_sha, "why": proposal.why}
        scene.add_fact(room, VERB, target, {k: v for k, v in params.items() if v})
        written += 1

    if written:
        referenced = {str(f.object) for f in scene.facts} | {ref for c in scene.constraints.values() for ref in c.scope}
        for entity in [e for e in scene.entities.values() if e.kind == KIND and e.id not in referenced]:
            scene.remove_entity(entity.id)
        kg.parent.mkdir(parents=True, exist_ok=True)
        kg.write_text(to_yaml(dump(scene)), encoding="utf-8")
    return written, kept


__all__ = [
    "SYSTEM",
    "Proposal",
    "Unavailable",
    "build_prompt",
    "classify_tail",
    "merge",
    "parse",
    "unknown_names",
]
