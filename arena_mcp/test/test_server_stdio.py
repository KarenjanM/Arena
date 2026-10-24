from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters, stdio_client


async def _session(domains, body, env):
    params = StdioServerParameters(command=sys.executable, args=["-m", "arena_mcp", "serve", *domains], env={**os.environ, **env})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return await body(session)


def test_server_lists_and_calls_jobs_tools(tmp_path):

    async def body(session):
        tools = {tool.name for tool in (await session.list_tools()).tools}
        listed = await session.call_tool("jobs_list", {})
        missing = await session.call_tool("jobs_status", {"job_id": "nope"})
        domains = await session.read_resource("arena://domains")
        return tools, listed, missing, domains

    tools, listed, missing, domains = asyncio.run(_session(["jobs"], body, {"ARENA_DATA_DIR": str(tmp_path)}))
    assert {"jobs_list", "jobs_status", "jobs_log", "jobs_wait", "jobs_cancel"} <= tools
    assert not listed.is_error
    assert missing.is_error and "jobs_list shows the known jobs" in missing.content[0].text
    described = json.loads(domains.contents[0].text)
    assert [entry["domain"] for entry in described] == ["jobs"]


def test_unknown_domain_fails_with_the_installed_list():
    import subprocess

    done = subprocess.run([sys.executable, "-m", "arena_mcp", "serve", "no_such_domain"], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert done.returncode != 0
    assert "installed:" in done.stderr and "jobs" in done.stderr


def test_ls_reports_every_installed_domain():
    import subprocess

    done = subprocess.run([sys.executable, "-m", "arena_mcp", "ls", "--json"], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    names = {entry["domain"] for entry in json.loads(done.stdout)}
    assert "jobs" in names
