"""`arena mcp`: list and serve the arena_mcp domains, the in-container half of the MCP launcher."""

import importlib.util
import os
import subprocess
import sys

from arena_cli.common import CLIError, _exec, _host_path, make_verb
from arena_cli.complete import Flags, Static, Union

DESCRIPTION = """Serve the Arena MCP domains to an MCP client over stdio.

Usage:
  arena mcp                           installed domains, served or unavailable with the reason, and how to register the server
  arena mcp ls [DOMAIN ...] [--json]  the same list, with the tools of each named domain, --json for every field
  arena mcp serve [DOMAIN ...]        serve the named domains, every served one by default, over stdio

On the host an MCP client runs _meta/tools/arena-mcp, which attaches to the running
arena container (a fork with --fork, another machine with --host) and serves from
there. The .mcp.json at the Arena root registers it as the server 'arena'."""


def domains() -> list[str]:
    """Domain names registered under the arena_mcp entry-point group, without importing them."""
    if importlib.util.find_spec("arena_mcp") is None:
        return []
    from importlib.metadata import entry_points

    from arena_mcp import GROUP

    return sorted({point.name for point in entry_points(group=GROUP)})


def _require() -> None:
    if importlib.util.find_spec("arena_mcp") is None:
        raise CLIError("arena_mcp is not importable here, run 'arena build arena_mcp' and re-source")


def _python_mcp(*args: str) -> list[str]:
    return [sys.executable, "-m", "arena_mcp", *args]


def status() -> int:
    """Print the domain table and the lines that register the server with a client."""
    _require()
    sys.stdout.flush()
    rc = subprocess.run(_python_mcp("ls"), check=False).returncode
    launcher = _host_path(os.path.join(os.environ.get("ARENA_DIR", ""), "_meta", "tools", "arena-mcp"))
    print()
    print("register the server with an MCP client as a stdio command:")
    print(f"  host       {launcher} serve [DOMAIN ...]   (the Arena root's .mcp.json registers it as 'arena')")
    print("  container  python3 -m arena_mcp serve [DOMAIN ...]   (from a sourced shell)")
    print("'arena mcp ls DOMAIN' lists a domain's tools, 'arena mcp serve' serves them")
    return rc


def ls(argv: list[str]) -> None:
    """List the domains, with the tools of each named one.

    `arena mcp ls [DOMAIN ...] [--json]`. Each domain shows its tool count and
    whether it is served, or why it is unavailable (a missing feature or
    module). Named domains also list every tool with its one-line summary,
    --json prints the full description of each domain and tool. An unknown
    name lists the installed domains.
    """
    _require()
    _exec(*_python_mcp("ls", *argv))


def serve(argv: list[str]) -> None:
    """Serve domains over stdio, every served one when none is named.

    `arena mcp serve [DOMAIN ...]`. Stdout carries only the protocol, so run it
    from an MCP client, not a terminal. Unavailable domains are skipped and
    listed under arena://domains. An unknown name lists the installed domains.
    """
    _require()
    _exec(*_python_mcp("serve", *argv))


DOMAINS = Static(domains)

COMMANDS = {
    v.name: v
    for v in (
        make_verb("ls", ls, passthrough=True, complete=Union(DOMAINS, Flags({"--json": "every field of each domain and tool"}))),
        make_verb("serve", serve, passthrough=True, complete=DOMAINS),
    )
}
