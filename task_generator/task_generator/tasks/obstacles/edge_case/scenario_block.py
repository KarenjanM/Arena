from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import attrs
import yaml

from .effects import Effect, EffectError, parse_effects

BLOCK_KEY = "edge_case"
SIDECAR = "edge_case.yaml"

LABELS: tuple[str, ...] = ("id", "note", "base", "prompt_id", "steps", "provenance", "decisions", "owned", "formation")

ACTIVE: tuple[str, ...] = ("effects", "objects")


class BlockError(Exception):
    pass


@attrs.frozen
class Block:
    id: str = ""
    note: str = ""
    base: str = ""
    prompt_id: str = ""
    steps: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = attrs.Factory(dict)
    decisions: Mapping[str, Any] = attrs.Factory(dict)
    owned: Mapping[str, tuple[str, ...]] = attrs.Factory(dict)
    formation: Mapping[str, Any] = attrs.Factory(dict)
    effects: tuple[Effect, ...] = ()
    objects: str = ""

    @property
    def empty(self) -> bool:
        return not self.effects and not self.objects

    def describe(self) -> str:
        kinds = ", ".join(str(e.kind) + (f"({e.label})" if e.label else "") for e in self.effects)
        return f"{self.id or '?'}: {len(self.effects)} effect(s) [{kinds}]" + (f", objects={self.objects}" if self.objects else "")


def scenario_path(world: str, scenario: str) -> Path | None:
    if not world or not scenario:
        return None
    # outside the try: an API move must fail loudly, not read as "no block" (the 2026-10-13 sync moved this module)
    from arena_simulation_setup.tree.World import WorldIdentifier  # noqa: PLC0415

    try:
        view = WorldIdentifier(world).resolve_sync().scenario(scenario).resolve_sync()
    except FileNotFoundError:
        return None
    path = Path(view.scenario_path)
    return path if path.is_file() else None


def parse(raw: object, *, where: str = BLOCK_KEY) -> Block:
    """Strict: an unknown key is a typo that would otherwise be a silent no-op."""
    if not isinstance(raw, Mapping):
        raise BlockError(f"{where}: must be a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(LABELS) - set(ACTIVE))
    if unknown:
        raise BlockError(f"{where}: unknown key(s) {unknown}; allowed: {sorted((*LABELS, *ACTIVE))}")
    try:
        effects = parse_effects(raw.get("effects"))
    except EffectError as exc:
        raise BlockError(f"{where}: {exc}") from None
    steps = raw.get("steps") or ()
    if isinstance(steps, str):
        steps = (steps,)
    if not isinstance(steps, (list, tuple)):
        raise BlockError(f"{where}.steps: expected a list of step labels")
    provenance = raw.get("provenance") or {}
    decisions = raw.get("decisions") or {}
    owned_raw = raw.get("owned") or {}
    if not isinstance(provenance, Mapping) or not isinstance(decisions, Mapping) or not isinstance(owned_raw, Mapping):
        raise BlockError(f"{where}: provenance, decisions and owned must be mappings")
    owned = {str(k): tuple(str(n) for n in (v or ())) for k, v in owned_raw.items()}
    formation = raw.get("formation") or {}
    if not isinstance(formation, Mapping):
        raise BlockError(f"{where}: formation must be a mapping")
    return Block(
        id=str(raw.get("id") or ""),
        note=str(raw.get("note") or ""),
        base=str(raw.get("base") or ""),
        prompt_id=str(raw.get("prompt_id") or ""),
        steps=tuple(str(s) for s in steps),
        provenance=dict(provenance),
        decisions=dict(decisions),
        owned=owned,
        formation=dict(formation),
        effects=effects,
        objects=str(raw.get("objects") or ""),
    )


def read(path: Path) -> Block | None:
    """The `edge_case.yaml` beside `path`, else its own `edge_case:` key; None when neither, BlockError when malformed."""
    sidecar = path.with_name(SIDECAR)
    source = sidecar if sidecar.is_file() else path
    try:
        doc = yaml.safe_load(source.read_text()) or {}
    except yaml.YAMLError as exc:
        raise BlockError(f"{source}: not valid YAML: {exc}") from None
    raw = doc if source is sidecar else (doc.get(BLOCK_KEY) if isinstance(doc, dict) else None)
    if raw is None:
        return None
    try:
        return parse(raw)
    except BlockError as exc:
        raise BlockError(f"{source}: {exc}") from None
