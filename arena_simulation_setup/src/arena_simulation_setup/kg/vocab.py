"""Versioned KG vocabulary packages: node kinds, verbs and constraint kinds."""

from __future__ import annotations

import functools
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs

ANY = "*"
LITERAL = "literal"
PACKAGES_DIR = Path(__file__).parent / "packages"

_TABLE_KEYS = {"nodes": (), "verbs": ("from", "to"), "constraints": ("arity",)}
_FILE_KEYS = frozenset({"name", "version", "requires", *_TABLE_KEYS})


@attrs.frozen
class Package:
    name: str
    version: int
    nodes: Mapping[str, Mapping[str, Any]] = attrs.field(factory=dict)
    verbs: Mapping[str, Mapping[str, Any]] = attrs.field(factory=dict)
    constraints: Mapping[str, Mapping[str, Any]] = attrs.field(factory=dict)
    requires: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


def load_package(path: str | Path) -> Package:
    """Read one `<name>@<version>.toml` package file."""
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    unknown = sorted(set(data) - _FILE_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {', '.join(map(repr, unknown))}")
    for required in ("name", "version"):
        if required not in data:
            raise ValueError(f"{path}: package file misses required key {required!r}")
    for table, keys in _TABLE_KEYS.items():
        for word, spec in data.get(table, {}).items():
            missing = [k for k in keys if k not in spec]
            if missing:
                raise ValueError(f"{path}: {table}.{word} misses required key(s) {', '.join(map(repr, missing))}")
    for kind, spec in data.get("nodes", {}).items():
        if spec.get("propless") and spec.get("props"):
            raise ValueError(f"{path}: nodes.{kind} is propless but declares props {spec['props']}")
    package = Package(
        name=str(data["name"]),
        version=int(data["version"]),
        nodes=data.get("nodes", {}),
        verbs=data.get("verbs", {}),
        constraints=data.get("constraints", {}),
        requires=tuple(str(r) for r in data.get("requires", [])),
    )
    if path.stem != package.key:
        raise ValueError(f"{path}: file declares package {package.key!r}, expected the file name {path.stem!r}")
    return package


@functools.cache
def available(directory: Path = PACKAGES_DIR) -> dict[str, Package]:
    """Every package in `directory`, keyed `<name>@<version>`."""
    return {p.key: p for p in map(load_package, sorted(Path(directory).glob("*.toml")))}


def resolve(name: str, directory: Path = PACKAGES_DIR) -> Package:
    """The package `<name>@<version>`, or the newest version of a bare `<name>`."""
    packages = available(directory)
    if name in packages:
        return packages[name]
    candidates = [p for p in packages.values() if p.name == name]
    if not candidates:
        raise ValueError(f"unknown package {name!r} (available: {', '.join(sorted(packages))})")
    return max(candidates, key=lambda p: p.version)


class Vocabulary:
    """The union of the packages in `uses`, with one owner per word."""

    def __init__(self, uses: Sequence[str], directory: Path = PACKAGES_DIR):
        self.uses: list[str] = list(uses)
        self.packages: list[Package] = [resolve(u, directory) for u in self.uses]
        self.nodes: dict[str, Mapping[str, Any]] = {}
        self.verbs: dict[str, Mapping[str, Any]] = {}
        self.constraints: dict[str, Mapping[str, Any]] = {}
        loaded = {p.key for p in self.packages} | {p.name for p in self.packages}
        for pkg in self.packages:
            for need in pkg.requires:
                if need not in loaded:
                    raise ValueError(f"package {pkg.key} requires {need!r}, which is not loaded (uses: {', '.join(self.uses)})")
        owner: dict[str, str] = {}
        for pkg in self.packages:
            for table, words in ((self.nodes, pkg.nodes), (self.verbs, pkg.verbs), (self.constraints, pkg.constraints)):
                for word, spec in words.items():
                    if word in owner:
                        raise ValueError(f"word {word!r} defined by both {owner[word]} and {pkg.key}")
                    owner[word] = pkg.key
                    table[word] = spec

    def props(self, kind: str) -> list[str]:
        return list(self.nodes[kind].get("props", []))

    def is_auto(self, kind: str) -> bool:
        """Whether an entity of `kind` is declared by its first reference."""
        return bool(self.nodes[kind].get("auto", False))

    def is_propless(self, kind: str) -> bool:
        return bool(self.nodes[kind].get("propless", False))

    def authorable(self) -> dict[str, Mapping[str, Any]]:
        """The node kinds a model may write."""
        return {k: s for k, s in self.nodes.items() if s.get("authorable", True)}

    def is_entity_ref(self, value: object) -> bool:
        return isinstance(value, str) and ":" in value and value.split(":", 1)[0] in self.nodes
