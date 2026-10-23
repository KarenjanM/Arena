from __future__ import annotations

import asyncio
import fnmatch
import importlib
import json
import os
import tempfile
import typing
from pathlib import Path
from typing import Annotated

from pydantic import Field

from arena_mcp.core import JOBS, Domain, ToolError, cli

if typing.TYPE_CHECKING:
    from types import ModuleType

    from arena_mcp.core import JobHandle


def _ready() -> str | None:
    try:
        cli.arena_dir()
    except ToolError as exc:
        return exc.message
    return None


domain = Domain(
    "assets",
    """Resolvable Arena data (objects, humans, materials, walls, sounds, animations, worlds, suites, contests, manifests) through the same code the `arena asset` verbs run.
    find and ls show what each resolver answers, pull fetches into the local cache, push publishes to the kind's bucket after the preflight that refuses unresolvable references.
    The import tools convert upstream datasets into local items when the importers registry of arena_assets is installed; imports run as jobs.""",
    check=_ready,
)

Kind = Annotated[str, Field(description="asset kind, assets_kinds lists them")]
Bucket = Annotated[str | None, Field(description="bucket to use instead of the kind's first configured one")]
LISTING_LIMIT = 500


def _json(*args: str, check: bool = True) -> object:
    done = cli.run(*args, check=check, clip=None)
    try:
        return json.loads(done["stdout"])
    except json.JSONDecodeError as exc:
        tail = (done["stderr"] or done["stdout"]).strip().splitlines()[-10:]
        raise ToolError(f"{done['command']} exited {done['exit_code']} without JSON:\n" + "\n".join(tail), hint="assets_kinds lists the kinds") from exc


@domain.tool(read_only=True)
def assets_kinds() -> dict[str, object]:
    """The resolvable kinds `arena asset` moves, with a description each."""
    return typing.cast(dict[str, object], _json("asset", "ls", "--json"))


@domain.tool(read_only=True, open_world=True)
def assets_find(kind: Kind, name: Annotated[str, Field(description="asset name, e.g. domain/name for domain assets")]) -> dict[str, object]:
    """Every resolver's verdict for one asset (hit, miss, shadowed, stale) without downloading it, like `arena asset find`."""
    verdicts = typing.cast(list[dict[str, object]], _json("asset", "find", kind, name, "--json", check=False))
    return {"kind": kind, "name": name, "resolves": any(verdict["verdict"] in ("hit", "stale") for verdict in verdicts), "verdicts": verdicts}


@domain.tool(read_only=True, open_world=True)
def assets_ls(
    kind: Kind,
    network: Annotated[bool, Field(description="also list bucket-only entries and mark local copies that shadow them")] = False,
    pattern: Annotated[str | None, Field(description="glob over the names")] = None,
) -> dict[str, object]:
    """Every available asset of one kind, like `arena asset ls`."""
    listed = typing.cast(list[object], _json("asset", "ls", kind, "--json", *(("--network",) if network else ())))
    rows = [entry if isinstance(entry, dict) else {"name": entry} for entry in listed]
    if pattern is not None:
        rows = [row for row in rows if fnmatch.fnmatchcase(str(row["name"]), pattern)]
    return {"kind": kind, "count": len(rows), "assets": rows}


@domain.tool(open_world=True, idempotent=True)
def assets_pull(kind: Kind, name: Annotated[str, Field(description="asset name")], bucket: Bucket = None) -> dict[str, object]:
    """Download one asset from its bucket into the local cache and return the cache path, like `arena asset pull`."""
    done = cli.run("asset", "pull", kind, name, *(("--bucket", bucket) if bucket else ()), timeout=1800, check=False)
    lines = done["stdout"].strip().splitlines()
    if done["exit_code"] != 0 or not lines:
        reason = (done["stderr"] or done["stdout"]).strip().splitlines()[-5:]
        raise ToolError(f"{kind} {name} was not pulled:\n" + "\n".join(reason), hint="assets_find shows which resolvers know it")
    return {"kind": kind, "name": name, "path": lines[-1]}


@domain.tool(destructive=True, open_world=True)
def assets_push(
    kind: Kind,
    name: Annotated[str, Field(description="asset name")],
    bucket: Bucket = None,
    force: Annotated[bool, Field(description="publish despite a failed preflight or a world that ships in the repo")] = False,
) -> dict[str, object]:
    """Publish one asset to its kind's bucket after the preflight `arena asset push` runs, which refuses references nobody else could resolve, with GCS_ACCESS_TOKEN set."""
    done = cli.run("asset", "push", kind, name, "--yes", *(("--bucket", bucket) if bucket else ()), *(("--force",) if force else ()), timeout=3600, check=False)
    if done["exit_code"] != 0:
        reason = (done["stderr"] + done["stdout"]).strip().splitlines()[-30:]
        raise ToolError(f"{kind} {name} was not published:\n" + "\n".join(reason), hint="bundle the missing references under the asset directory, or pass force=true")
    return {"kind": kind, "name": name, "published": done["stdout"].strip().splitlines()[-1:]}


def _importers() -> ModuleType:
    try:
        importers = importlib.import_module("arena_assets.importers")
    except ImportError as exc:
        raise ToolError("the importers registry arena_assets.importers is not installed in this checkout", hint="it ships with the dataset adapters of arena_assets, the other assets tools work without it") from exc
    return importers


def _cache(source: str) -> Path:
    root = os.environ.get("ARENA_ASSETS_DIR")
    return Path(root) / ".import" / source if root else Path(tempfile.gettempdir()) / "arena_assets_import" / source


def _importer(kind: str, source: str) -> type:
    registry = _importers().IMPORTERS
    if kind not in registry.kinds():
        raise ToolError(f"no importer kind {kind!r}, kinds: {', '.join(registry.kinds())}", hint="assets_import_sources lists the sources")
    if (kind, source) not in registry:
        raise ToolError(f"no {kind} importer {source!r}, registered: {', '.join(registry.names(kind))}", hint="assets_import_sources lists the sources")
    return registry.load(kind, source)


def _target(importer: type) -> str | None:
    try:
        return str(importer.default_out())
    except RuntimeError:
        return None


@domain.tool(read_only=True)
def assets_import_sources(kind: Annotated[str | None, Field(description="only the sources producing this kind")] = None) -> list[dict[str, object]]:
    """The registered upstream datasets by kind with license verdict (yes, nc, sa, no, unclear), host, on-demand granularity and description."""
    registry = _importers().IMPORTERS
    return [{"kind": meta.kind, "source": meta.name, "license": meta.license.value, "host": meta.host, "on_demand": meta.on_demand.value, "package": meta.package, "description": meta.description} for meta in registry if kind is None or meta.kind == kind]


@domain.tool(read_only=True, open_world=True)
def assets_import_items(
    kind: Kind,
    source: Annotated[str, Field(description="upstream dataset, assets_import_sources lists them")],
    pattern: Annotated[str | None, Field(description="glob over the item ids")] = None,
    limit: Annotated[int, Field(description="most item ids to return")] = LISTING_LIMIT,
) -> dict[str, object]:
    """The item ids one source offers (cached for a day, for good for pinned sources) and the root an import writes to."""
    importer = _importer(kind, source)
    items = list(importer(cache=_cache(source)).cached_items())
    if pattern is not None:
        items = [item for item in items if fnmatch.fnmatchcase(item, pattern)]
    return {"kind": kind, "source": source, "count": len(items), "items": items[:limit], "truncated": len(items) > limit, "target": _target(importer)}


@domain.tool(destructive=True, open_world=True)
async def assets_import(
    kind: Kind,
    source: Annotated[str, Field(description="upstream dataset, assets_import_sources lists them")],
    items: Annotated[list[str], Field(description="item ids, assets_import_items lists them")],
    out: Annotated[str | None, Field(description="target root instead of the imported root the resolvers read")] = None,
) -> dict[str, object]:
    """Convert items of an upstream dataset into local items, each with an ATTRIBUTION.yaml, as a job whose result maps every item to its path or failure."""
    if not items:
        raise ToolError("items is empty", hint="assets_import_items lists the item ids")
    importer_class = _importer(kind, source)
    if out is not None:
        target = Path(out).expanduser()
    else:
        default = _target(importer_class)
        if default is None:
            raise ToolError("the importer has no default target because ARENA_ASSETS_DIR is not set", hint="pass out")
        target = Path(default)
    importer = importer_class(cache=_cache(source))

    async def body(handle: JobHandle) -> dict[str, object]:
        imported: dict[str, str] = {}
        failed: dict[str, str] = {}
        for index, item in enumerate(items):
            handle.log(f"importing {kind} {source} {item}")
            try:
                imported[item] = str(await asyncio.to_thread(importer.import_item, item, target))
            except Exception as exc:
                failed[item] = str(exc.args[0]) if exc.args else repr(exc)
                handle.log(f"failed {item}: {failed[item]}")
            handle.progress(done=index + 1, total=len(items), failed=len(failed))
        return {"kind": kind, "source": source, "target": str(target), "imported": imported, "failed": failed}

    return JOBS.start_task("import", f"import {kind} {source} {len(items)} item(s)", body).as_dict()
