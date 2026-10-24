from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import AsyncIterator, Iterable
from importlib.metadata import entry_points
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import GROUP
from .core.domain import Domain
from .core.jobs import JOBS
from .core.ros import ROS

INSTRUCTIONS = """\
Arena: ROS 2 navigation research platform (simulators, task generator, robots, worlds, benchmarks).
Tools are named <domain>_<verb>. Read arena://domains for what each domain covers and why a domain is missing.
Env-scoped tools take env_id; it may be omitted when exactly one env runs (runtime_envs lists them).
Long operations return a job: poll jobs_status or jobs_wait, read jobs_log, stop with jobs_cancel.
Errors carry a hint line with the next step."""


@dataclasses.dataclass
class Loaded:
    name: str
    source: str
    domain: Domain | None = None
    reason: str | None = None

    @property
    def served(self) -> bool:
        return self.domain is not None and self.reason is None


def discover(selected: Iterable[str] | None = None) -> list[Loaded]:
    """Every domain registered under the arena_mcp.domains entry-point group, loaded and checked."""
    wanted = set(selected or ())
    found: list[Loaded] = []
    for point in sorted(entry_points(group=GROUP), key=lambda point: point.name):
        if wanted and point.name not in wanted:
            continue
        loaded = Loaded(name=point.name, source=point.value)
        try:
            domain = point.load()
        except Exception as exc:
            loaded.reason = f"import failed: {type(exc).__name__}: {exc}"
        else:
            if not isinstance(domain, Domain) or domain.name != point.name:
                loaded.reason = f"{point.value} is not the Domain {point.name!r}"
            else:
                loaded.domain = domain
                loaded.reason = domain.unavailable()
        found.append(loaded)
    unknown = wanted - {loaded.name for loaded in found}
    if unknown:
        known = sorted(point.name for point in entry_points(group=GROUP))
        raise SystemExit(f"unknown domain(s) {', '.join(sorted(unknown))}, installed: {', '.join(known)}")
    return found


def describe(loaded: Loaded) -> dict[str, Any]:
    out: dict[str, Any] = {"domain": loaded.name, "served": loaded.served, "source": loaded.source}
    if loaded.reason:
        out["reason"] = loaded.reason
    if loaded.domain is not None:
        out["doc"] = loaded.domain.doc
        out["tools"] = {spec.name: spec.summary for spec in loaded.domain.tools}
        out["resources"] = [resource.uri for resource in loaded.domain.resources]
    return out


@contextlib.asynccontextmanager
async def _lifespan(_server: MCPServer) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await JOBS.shutdown()
        await ROS.shutdown()


def build(selected: Iterable[str] | None = None) -> tuple[MCPServer, list[Loaded]]:
    loaded = discover(selected)
    served = [entry for entry in loaded if entry.served]
    summary = "\n".join(f"- {entry.name}: {entry.domain.doc.splitlines()[0]}" for entry in served if entry.domain is not None)
    server = MCPServer(name="arena", instructions=f"{INSTRUCTIONS}\n\nDomains:\n{summary}", lifespan=_lifespan)
    for entry in served:
        if entry.domain is not None:
            entry.domain.install(server)

    @server.resource("arena://domains", mime_type="application/json", description="Installed domains, whether each is served and why not")
    def domains() -> list[dict[str, Any]]:
        return [describe(entry) for entry in loaded]

    @server.resource("arena://domains/{name}", mime_type="application/json", description="One domain: its doc, tools and resources")
    def domain(name: str) -> dict[str, Any]:
        for entry in loaded:
            if entry.name == name:
                return describe(entry)
        raise ValueError(f"no domain {name!r}")

    return server, loaded


async def serve(selected: Iterable[str] | None = None) -> None:
    server, _ = build(selected)
    await server.run_stdio_async()
