from __future__ import annotations

import fnmatch
import json
import typing
from typing import Annotated

from pydantic import Field

from arena_mcp.core import Domain, ToolError, cli


def _ready() -> str | None:
    try:
        cli.arena_dir()
    except ToolError as exc:
        return exc.message
    return None


domain = Domain(
    "assets",
    """Resolvable Arena data of every kind `arena asset` moves (assets_kinds lists them) through the same code the `arena asset` verbs run.
    find and ls show what each resolver answers, pull fetches into the local cache, push publishes to the kind's bucket after the preflight that refuses unresolvable references.""",
    check=_ready,
)

Kind = Annotated[str, Field(description="asset kind, assets_kinds lists them")]
Bucket = Annotated[str | None, Field(description="bucket to use instead of the kind's first configured one")]


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
