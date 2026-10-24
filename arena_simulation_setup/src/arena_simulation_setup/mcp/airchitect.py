from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Annotated, TypedDict, cast

import pydantic

from arena_mcp.core import JOBS, Domain, JobHandle, ToolError

from .worlds import level_preview, path_component, world_verdict

DEFAULT_URL = "http://localhost:3000"
DEPLOY_HINT = "deploy the stack on the host with `cd ~/airchitect && ./airchitect deploy up --cpu --arena text_kg`, or point AIRCHITECT_URL at a running webapp"
DOMAINS = ("residential", "hospital", "office")
STAGES = ("prompt", "floorplan", "decoration", "scenario")
ROUTES = {"domain": "domain", "prompt": "3dsg/prompt", "floorplan": "floor-plan", "decoration": "decoration", "scenario": "scenario"}
DEADLINES = {"prompt": 600.0, "floorplan": 900.0, "decoration": 1500.0, "scenario": 1500.0, "edit": 600.0}
POLL_S = 3.0
REQUEST_S = 30.0
EXPORT_S = 600.0
COST_UNAVAILABLE = "unavailable: the stack reports cost only in its SSE events, and only when it runs with COST_DEBUG=1"


def base_url() -> str:
    """The airchitect webapp, AIRCHITECT_URL or localhost:3000."""
    return (os.environ.get("AIRCHITECT_URL") or DEFAULT_URL).rstrip("/")


domain = Domain(
    "airchitect",
    """The airchitect generation stack over its webapp HTTP API (AIRCHITECT_URL, default http://localhost:3000).
    Sessions live in the stack's Redis for one hour per uuid, stages run prompt, floorplan, decoration, scenario.
    airchitect_generate runs the whole pipeline as a job and saves the world under the writable world root, checked with strict parsing, NAV@r and a preview.
    Stage responses of every job are kept as JSON under $ARENA_DATA_DIR/mcp/airchitect/<run id>/.""",
)


def runs_root() -> Path:
    """$ARENA_DATA_DIR/mcp/airchitect, where each run keeps its stage responses."""
    return Path(os.environ.get("ARENA_DATA_DIR") or Path.home() / "arena_ws" / "data") / "mcp" / "airchitect"


def http(method: str, path: str, *, body: object = None, data: bytes | None = None, content_type: str = "application/json", timeout: float = REQUEST_S) -> tuple[int, bytes]:
    """One request to the webapp, its status and body, a ToolError when the webapp is unreachable."""
    payload = json.dumps(body).encode() if body is not None else data
    request = urllib.request.Request(f"{base_url()}/api/{path}", data=payload, method=method, headers={"Content-Type": content_type} if payload is not None else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        raise ToolError(f"airchitect webapp at {base_url()} did not answer {method} /api/{path}: {reason}", hint=DEPLOY_HINT) from exc


def call(method: str, path: str, body: object = None, *, timeout: float = REQUEST_S) -> object:
    """A JSON request to the webapp and its decoded answer, a ToolError with the body on a non-2xx status."""
    status, raw = http(method, path, body=body, timeout=timeout)
    if not 200 <= status < 300:
        text = raw.decode(errors="replace").strip()[:2000]
        if status == 404 and "expired" in text:
            raise ToolError(f"{method} /api/{path}: the session is gone ({text})", hint="sessions expire one hour after their last write, start a new one")
        raise ToolError(f"{method} /api/{path} answered HTTP {status}: {text}")
    if not raw.strip():
        return None
    return json.loads(raw)


async def acall(method: str, path: str, body: object = None, *, timeout: float = REQUEST_S) -> object:
    return await asyncio.to_thread(call, method, path, body, timeout=timeout)


def stack_models() -> dict[str, list[dict[str, object]]]:
    """The stack's models per stage, keyed by lowercase stage name."""
    listing = call("GET", "models")
    if not isinstance(listing, dict):
        raise ToolError(f"GET /api/models answered {listing!r}")
    return {stage.lower(): entries for stage, entries in listing.items()}


def defaults(listing: dict[str, list[dict[str, object]]]) -> dict[str, str]:
    """The default model name of each stage."""
    out: dict[str, str] = {}
    for stage, entries in listing.items():
        chosen = next((entry for entry in entries if entry.get("default")), entries[0] if entries else None)
        if chosen is not None:
            out[stage] = str(chosen["name"])
    return out


def pick_models(listing: dict[str, list[dict[str, object]]], wanted: dict[str, str] | None, required: tuple[str, ...] = STAGES) -> dict[str, str]:
    """The stack's defaults per stage with wanted overrides, each checked against the stage's models, failing when a required stage has none."""
    chosen = defaults(listing)
    for stage, name in (wanted or {}).items():
        if stage not in STAGES:
            raise ToolError(f"no stage {stage!r}", hint=f"stages: {', '.join(STAGES)}")
        names = [str(entry["name"]) for entry in listing.get(stage, [])]
        if name not in names:
            raise ToolError(f"stage {stage} has no model {name!r}", hint=f"models: {', '.join(names)}")
        chosen[stage] = name
    missing = [stage for stage in required if stage not in chosen]
    if missing:
        raise ToolError(f"the stack offers no model for {', '.join(missing)}", hint="airchitect_status shows what the stack serves")
    return chosen


def _check_domain(name: str) -> str:
    if name not in DOMAINS:
        raise ToolError(f"no domain {name!r}", hint=f"domains: {', '.join(DOMAINS)}")
    return name


def _check_stage(stage: str) -> str:
    if stage not in STAGES:
        raise ToolError(f"no stage {stage!r}", hint=f"stages: {', '.join(STAGES)}")
    return stage


class FloorplanScore(TypedDict):
    dropped_edges: list[str]
    auto_doors: list[str]


FloorplanTry = TypedDict("FloorplanTry", {"try": int, "dropped_edges": list[str], "auto_doors": list[str]})


def floorplan_score(graph: dict[str, object], record: dict[str, object]) -> FloorplanScore:
    """Graph edges the floor plan realized no door for, and the doors it added on its own (door_auto_*)."""
    plan = cast("dict[str, dict[str, object]]", record["plan"])
    levels = cast("dict[str, dict[str, list[dict[str, object]] | None]]", plan["world"]["levels"])
    doors = list(dict.fromkeys(str(door["name"]) for level in levels.values() for door in level.get("doors") or []))
    have = {door.removeprefix("door_") for door in doors}
    edges = cast("list[dict[str, object]]", graph["edges"])
    return {"dropped_edges": [str(edge["id"]) for edge in edges if str(edge["id"]) not in have], "auto_doors": [door for door in doors if door.startswith("door_auto_")]}


def pick_floorplan(graph: dict[str, object], tries: list[dict[str, object]]) -> tuple[int, list[FloorplanTry]]:
    """Index of the try with the fewest dropped edges, then the fewest auto doors, then the earliest, and every try's score."""
    scores: list[FloorplanTry] = [{"try": index + 1, **floorplan_score(graph, record)} for index, record in enumerate(tries)]
    best = min(range(len(scores)), key=lambda index: (len(scores[index]["dropped_edges"]), len(scores[index]["auto_doors"]), index))
    return best, scores


def target_dir(name: str, overwrite: bool) -> Path:
    """The writable world directory for name, refusing an existing world unless overwrite."""
    from arena_simulation_setup.tree.World import WorldIdentifier

    target = Path(WorldIdentifier(path_component(name)).resolve_write_sync().path)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise ToolError(f"world {name!r} already exists at {target}", hint="pass overwrite=True to replace it, or pick another name")
    return target


def install_zip(data: bytes, name: str, *, overwrite: bool) -> Path:
    """Unpack an exported world zip into the writable world root as world name."""
    target = target_dir(name, overwrite)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ToolError(f"the export is not a zip archive: {exc}") from exc
    with archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        for info in members:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ToolError(f"the export holds an unsafe path {info.filename!r}")
        if "world.yaml" not in {PurePosixPath(info.filename).as_posix() for info in members}:
            raise ToolError("the export holds no world.yaml at its root", hint="the stack's export changed, inspect the saved world.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=target.parent))
        try:
            for info in members:
                destination = staging / info.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info))
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return target


def check_world(path: Path, radius: float | None, out: Path) -> dict[str, object]:
    """The strict verdict, NAV@r per level and a preview PNG per level written to out."""
    from arena_simulation_setup.metrics.nav_at_r import DEFAULT_RADIUS, nav_at_r
    from arena_simulation_setup.tree.World import WorldView

    out.mkdir(parents=True, exist_ok=True)
    view = WorldView(path)
    verdict = world_verdict(view, None)
    report: dict[str, object] = {"validation": {key: verdict[key] for key in ("format", "valid", "reason", "warnings") if key in verdict}}
    value = DEFAULT_RADIUS if radius is None else float(radius)
    try:
        results = nav_at_r(view, radius=value, image=out / "nav_at_r.png")
    except Exception as exc:
        report["nav_at_r"] = {"radius": value, "error": f"{type(exc).__name__}: {exc}"}
    else:
        report["nav_at_r"] = {
            "radius": value,
            "levels": {level: {"value": result.asdict()["value"], "components": result.components, "largest_share": result.largest_share, "cut_off_zones": list(result.cut_off_zones), "traversable_area": result.traversable_area} for level, result in results.items()},
        }
    previews: list[str] = []
    try:
        world = view.load(validate=False)
        for level_id, level in world.levels.items():
            rendered = level_preview(level)
            if rendered is None:
                continue
            preview = out / f"preview_{level_id}.png"
            preview.write_bytes(rendered[0])
            previews.append(str(preview))
    except Exception as exc:
        report["preview_error"] = f"{type(exc).__name__}: {exc}"
    report["previews"] = previews
    return report


class Run:
    """One run directory: run.json plus a JSON file per stage request and response."""

    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.dir = runs_root() / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def meta(self) -> dict[str, object]:
        path = self.dir / "run.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    def update(self, **fields: object) -> None:
        meta = {**self.meta, **fields}
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(self.dir / "run.json")

    def save(self, tag: str, data: object) -> None:
        (self.dir / f"{tag}.json").write_text(json.dumps(data, indent=1))

    def load(self, tag: str) -> dict[str, object] | None:
        path = self.dir / f"{tag}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) and data.get("status") == "completed" else None


async def run_stage(handle: JobHandle, run: Run, stage: str, body: dict[str, object], tag: str) -> dict[str, object]:
    """POST a stage's deferred request and poll the stage until the callback stored completed or error."""
    route = ROUTES[stage]
    run.save(f"{tag}_request", body)
    handle.log(f"POST /api/{route}/rr ({tag})")
    started = time.monotonic()
    accepted = await acall("POST", f"{route}/rr", body)
    run.save(f"{tag}_accepted", accepted)
    uuid = str(body.get("uuid") or cast("dict[str, object]", accepted or {}).get("uuid") or "")
    if not uuid:
        raise ToolError(f"{stage}: the webapp returned no session uuid: {accepted!r}")
    return await poll(handle, run, stage, uuid, tag, started)


async def poll(handle: JobHandle, run: Run, stage: str, uuid: str, tag: str, started: float) -> dict[str, object]:
    """Poll GET /api/<stage route> until the stage record is completed or error."""
    last = None
    while time.monotonic() < started + DEADLINES[stage]:
        record = await acall("GET", f"{ROUTES[stage]}?uuid={urllib.parse.quote(uuid)}")
        status = record.get("status") if isinstance(record, dict) else None
        if status != last:
            handle.log(f"{tag}: status {status} after {time.monotonic() - started:.0f}s")
            last = status
        if isinstance(record, dict) and status in ("completed", "error"):
            run.save(tag, record)
            if status == "error":
                raise ToolError(f"{stage} failed in the stack: {record.get('error')}", hint=f"the stage records are in {run.dir}")
            return record
        await asyncio.sleep(POLL_S)
    raise ToolError(f"{stage}: no result after {DEADLINES[stage]:.0f}s", hint=f"the stack may still be working, airchitect_session {uuid} shows its state")


def stamps(records: dict[str, dict[str, object] | None]) -> dict[str, object]:
    """The timestamp of every pipeline stage record, None where a stage holds nothing."""
    return {stage: (records.get(stage) or {}).get("timestamp") for stage in STAGES}


async def export_world(handle: JobHandle, run: Run, uuid: str, plan: dict[str, object], scenario: object, name: str, overwrite: bool, radius: float | None) -> dict[str, object]:
    """Export a plan as a zip, unpack it into the writable world root and check the result."""
    body: dict[str, object] = {"plan": plan, "model": "zip", "uuid": uuid, "world_name": name}
    if scenario is not None:
        body["scenario"] = scenario
    run.save("export_request", body)
    handle.progress(stage="export")
    handle.log("POST /api/export/zip")
    status, raw = await asyncio.to_thread(http, "POST", "export/zip", body=body, timeout=EXPORT_S)
    if status != 200:
        (run.dir / "export_error.txt").write_bytes(raw)
        raise ToolError(f"export answered HTTP {status}: {raw.decode(errors='replace')[:2000]}")
    (run.dir / "world.zip").write_bytes(raw)
    handle.progress(stage="install")
    path = await asyncio.to_thread(install_zip, raw, name, overwrite=overwrite)
    handle.log(f"unpacked into {path}")
    handle.progress(stage="check")
    checks = await asyncio.to_thread(check_world, path, radius, run.dir)
    return {"world": name, "path": str(path), "uuid": uuid, **checks, "run_dir": str(run.dir), "cost": COST_UNAVAILABLE}


def latest_plan(session: dict[str, dict[str, object] | None]) -> tuple[dict[str, object], object, str]:
    """The newest completed plan of a session, its scenario when the scenario stage finished, and the stage it came from."""
    for stage in ("scenario", "decoration", "floorplan"):
        record = session.get(stage)
        if isinstance(record, dict) and record.get("status") == "completed" and isinstance(plan := record.get("plan"), dict):
            return plan, record.get("scenario") if stage == "scenario" else None, stage
    raise ToolError("the session has no completed floor plan to export", hint="run airchitect_stage for floorplan first")


async def session_records(uuid: str) -> dict[str, dict[str, object] | None]:
    """The stored record of every stage of a session, None where a stage never ran."""
    out: dict[str, dict[str, object] | None] = {}
    for stage, route in ROUTES.items():
        status, raw = await asyncio.to_thread(http, "GET", f"{route}?uuid={urllib.parse.quote(uuid)}")
        if status == 404 and stage == "domain" and b"Domain not found" in raw:
            out[stage] = None
            continue
        if status == 404:
            raise ToolError(f"no session {uuid}: {raw.decode(errors='replace').strip()}", hint="sessions expire one hour after their last write")
        if status != 200:
            raise ToolError(f"GET /api/{route} answered HTTP {status}: {raw.decode(errors='replace')[:500]}")
        out[stage] = json.loads(raw) if raw.strip() else None
    return out


def summarize(record: dict[str, object] | None) -> dict[str, object] | None:
    """A stage record without its bulk: status, model, timestamp, error and counts."""
    if record is None:
        return None
    model = record.get("model")
    plan = cast("dict[str, object]", record.get("plan") or {})
    out: dict[str, object] = {key: record[key] for key in ("status", "domain", "timestamp", "error", "prompt", "source", "graph_source") if record.get(key) is not None}
    if isinstance(model, dict):
        out["model"] = model.get("name")
    graph = record.get("graph") or plan.get("graph")
    if isinstance(graph, dict):
        out["graph"] = {"nodes": len(graph.get("nodes") or []), "edges": len(graph.get("edges") or [])}
    world = plan.get("world")
    if isinstance(world, dict):
        levels = world.get("levels") or {}
        out["levels"] = {level_id: {key: len(level.get(key) or []) for key in ("zones", "doors", "elevators", "objects", "humans")} for level_id, level in levels.items()}
    if record.get("scenario") is not None:
        scenario = record["scenario"]
        out["scenario"] = sorted(scenario) if isinstance(scenario, dict) else type(scenario).__name__
    if record.get("kg") is not None:
        out["has_kg"] = True
    return out


def session_domain(session: dict[str, dict[str, object] | None]) -> str:
    """The domain a session was started with."""
    for stage in ("domain", "prompt", "floorplan", "decoration", "scenario"):
        record = session.get(stage)
        if isinstance(record, dict) and record.get("domain") in DOMAINS:
            return str(record["domain"])
    return DOMAINS[0]


Stage = Annotated[str, pydantic.Field(description="prompt, floorplan, decoration or scenario")]
DomainName = Annotated[str, pydantic.Field(description="residential, hospital or office")]
WorldName = Annotated[str, pydantic.Field(description="world name under the writable world root, one path component")]


@domain.tool(read_only=True, open_world=True)
async def airchitect_status() -> dict[str, object]:
    """Whether the webapp answers, its models per stage with defaults, and backend availability from models/score."""
    listing = await asyncio.to_thread(stack_models)
    out: dict[str, object] = {"url": base_url(), "reachable": True, "defaults": defaults(listing), "models": {stage: [entry.get("name") for entry in entries] for stage, entries in listing.items()}}
    try:
        scores = await acall("POST", "models/score", {"domain": DOMAINS[0]})
    except ToolError as exc:
        out["availability_error"] = exc.message
    else:
        stages = scores.get("stages", {}) if isinstance(scores, dict) else {}
        out["availability"] = {stage.lower(): [{key: entry.get(key) for key in ("name", "available", "eligible", "score", "auto", "reasons")} for entry in entries] for stage, entries in stages.items()}
    return out


@domain.tool(read_only=True, open_world=True)
async def airchitect_models(stage: Annotated[str | None, pydantic.Field(description="prompt, floorplan, decoration, scenario or export, all when omitted")] = None) -> dict[str, object]:
    """The stack's models per stage with description, paper and the default flag."""
    listing = await asyncio.to_thread(stack_models)
    if stage is None:
        return {**listing}
    if stage not in listing:
        raise ToolError(f"no stage {stage!r}", hint=f"stages: {', '.join(listing)}")
    return {stage: listing[stage]}


@domain.tool(read_only=True, open_world=True)
async def airchitect_inspire(domain: DomainName, kind: Annotated[str, pydantic.Field(description="prompt for a world description, scenario for a scenario description")] = "prompt") -> dict[str, object]:
    """An example prompt from the stack for a domain."""
    if kind not in ("prompt", "scenario"):
        raise ToolError(f"no inspiration kind {kind!r}", hint="kinds: prompt, scenario")
    answer = await acall("POST", f"inspire/{kind}", {"domain": _check_domain(domain)})
    return {"domain": domain, "kind": kind, "prompt": answer.get("prompt") if isinstance(answer, dict) else answer}


@domain.tool(destructive=True, open_world=True)
async def airchitect_generate(
    name: WorldName,
    prompt: Annotated[str, pydantic.Field(description="description of the world to build")],
    domain: DomainName,
    models: Annotated[dict[str, str] | None, pydantic.Field(description="model name per stage, the stack's defaults for the rest")] = None,
    floorplan_tries: Annotated[int, pydantic.Field(ge=1, le=10, description="floor plans to request, the one with the fewest dropped graph edges, then the fewest auto doors, is kept")] = 3,
    scenario_prompt: Annotated[str | None, pydantic.Field(description="run the scenario stage with this description and export its scenario")] = None,
    overwrite: bool = False,
    nav_radius: Annotated[float | None, pydantic.Field(description="robot radius for NAV@r, the nav_at_r default when omitted")] = None,
    resume: Annotated[str | None, pydantic.Field(description="run id of an earlier airchitect_generate job: its completed stages are reused in the same session")] = None,
) -> dict[str, object]:
    """Run prompt, floor plan tries, decoration, optional scenario and zip export as a job, then save and check the world."""
    path_component(name)
    _check_domain(domain)
    await asyncio.to_thread(target_dir, name, overwrite)
    listing = await asyncio.to_thread(stack_models)
    chosen = pick_models(listing, models, required=("prompt", "floorplan", "decoration", *(("scenario",) if scenario_prompt else ())))
    if resume is not None and not (runs_root() / path_component(resume, "run id") / "run.json").is_file():
        raise ToolError(f"no airchitect run {resume!r} under {runs_root()}", hint="the run id is the job id of the earlier airchitect_generate")

    async def body(handle: JobHandle) -> dict[str, object]:
        run = Run(resume or handle.job.id)
        run.update(jobs=[*cast("list[str]", run.meta.get("jobs", [])), handle.job.id], name=name, prompt=prompt, domain=domain, models=chosen, scenario_prompt=scenario_prompt)
        handle.progress(stage="domain", run_dir=str(run.dir))
        uuid = str(run.meta.get("uuid") or "")
        if not uuid:
            uuid = str(cast("dict[str, object]", await acall("POST", "domain", {"domain": domain}))["uuid"])
            run.update(uuid=uuid)
        handle.log(f"session {uuid}, run dir {run.dir}")

        handle.progress(stage="prompt", uuid=uuid)
        prompted = run.load("01_prompt") or await run_stage(handle, run, "prompt", {"prompt": prompt, "domain": domain, "model": chosen["prompt"], "uuid": uuid, "supporting_examples": []}, "01_prompt")
        graph = cast("dict[str, object]", prompted["graph"])

        floorplan = run.load("02_floorplan")
        if floorplan is None:
            tries: list[dict[str, object]] = []
            for index in range(1, floorplan_tries + 1):
                handle.progress(stage="floorplan", attempt=index, attempts=floorplan_tries)
                try:
                    tries.append(await run_stage(handle, run, "floorplan", {"graph": graph, "domain": domain, "model": chosen["floorplan"], "uuid": uuid, "score_based_kwargs": {}}, f"02_floorplan_try{index}"))
                except ToolError as exc:
                    handle.log(f"floor plan try {index} failed: {exc.message}")
            if not tries:
                raise ToolError(f"all {floorplan_tries} floor plan tries failed", hint=f"see the job log and {run.dir}")
            best, scores = pick_floorplan(graph, tries)
            floorplan = tries[best]
            run.save("02_floorplan_choice", {"chosen": scores[best]["try"], "tries": scores})
            handle.log(f"floor plan: kept try {scores[best]['try']} of {len(tries)}, dropped {scores[best]['dropped_edges']}, auto doors {len(scores[best]['auto_doors'])}")
            if best != len(tries) - 1:
                kept = cast("dict[str, object]", floorplan["plan"])
                await acall("POST", f"callback/floorplan/{urllib.parse.quote(uuid)}", {"world": kept["world"], "graph": kept["graph"], "kg": floorplan.get("kg")})
            run.save("02_floorplan", floorplan)

        handle.progress(stage="decoration")
        decorated = run.load("03_decoration") or await run_stage(handle, run, "decoration", {"plan": floorplan["plan"], "domain": domain, "model": chosen["decoration"], "uuid": uuid}, "03_decoration")
        plan, scenario = cast("dict[str, object]", decorated["plan"]), None
        if scenario_prompt:
            handle.progress(stage="scenario")
            staged = run.load("04_scenario") or await run_stage(handle, run, "scenario", {"plan": plan, "domain": domain, "model": chosen["scenario"], "uuid": uuid, "prompt": scenario_prompt}, "04_scenario")
            plan, scenario = cast("dict[str, object]", staged["plan"]), staged.get("scenario")

        result = await export_world(handle, run, uuid, plan, scenario, name, overwrite, nav_radius)
        choice = run.dir / "02_floorplan_choice.json"
        if choice.is_file():
            result["floorplan_choice"] = json.loads(choice.read_text())
        run.update(result=result)
        return result

    job = JOBS.start_task("airchitect", f"generate {name}", body)
    return job.as_dict()


@domain.tool(read_only=True, open_world=True)
async def airchitect_session(uuid: str, full: Annotated[bool, pydantic.Field(description="return every stage record whole instead of a summary")] = False) -> dict[str, object]:
    """The state of each stage of a session: status, model, timestamp, error and graph and world counts."""
    records = await session_records(uuid)
    if full:
        return {"uuid": uuid, "stages": records}
    return {"uuid": uuid, "stages": {stage: summarize(record) for stage, record in records.items()}}


@domain.tool(destructive=True, open_world=True)
async def airchitect_stage(
    uuid: str,
    stage: Stage,
    model: Annotated[str | None, pydantic.Field(description="model for the stage, the stack's default when omitted")] = None,
    prompt: Annotated[str | None, pydantic.Field(description="prompt for the prompt or scenario stage, the session's own when omitted")] = None,
) -> dict[str, object]:
    """Rerun one stage of a session on its predecessor's stored result as a job."""
    _check_stage(stage)
    session = await session_records(uuid)
    listing = await asyncio.to_thread(stack_models)
    chosen = pick_models(listing, {stage: model} if model is not None else None, required=(stage,))[stage]
    domain_name = session_domain(session)
    request: dict[str, object] = {"domain": domain_name, "model": chosen, "uuid": uuid}
    if stage == "prompt":
        text = prompt or (session.get("prompt") or {}).get("prompt")
        if not text:
            raise ToolError("the session has no prompt to rerun", hint="pass prompt")
        request.update(prompt=text, supporting_examples=(session.get("prompt") or {}).get("supporting_examples") or [])
    elif stage == "floorplan":
        graph = (session.get("prompt") or {}).get("graph")
        if not isinstance(graph, dict):
            raise ToolError("the session has no prompt graph", hint="rerun the prompt stage first")
        request.update(graph=graph, score_based_kwargs={})
    else:
        source = session.get("floorplan" if stage == "decoration" else "decoration") or {}
        if source.get("status") != "completed" or not isinstance(source.get("plan"), dict):
            raise ToolError(f"the session has no completed {'floorplan' if stage == 'decoration' else 'decoration'} to build on", hint="rerun the stage before it first")
        request["plan"] = source["plan"]
        if stage == "scenario":
            text = prompt or (session.get("scenario") or {}).get("prompt")
            if not text:
                raise ToolError("the scenario stage needs a prompt", hint="pass prompt, airchitect_inspire with kind=scenario suggests one")
            request["prompt"] = text

    async def body(handle: JobHandle) -> dict[str, object]:
        run = Run(handle.job.id)
        run.update(uuid=uuid, stage=stage, model=chosen)
        handle.progress(stage=stage, run_dir=str(run.dir))
        record = await run_stage(handle, run, stage, request, stage)
        return {"uuid": uuid, "stage": stage, "model": chosen, "result": summarize(record), "run_dir": str(run.dir), "cost": COST_UNAVAILABLE}

    return JOBS.start_task("airchitect", f"{stage} {uuid}", body).as_dict()


@domain.tool(destructive=True, open_world=True)
async def airchitect_kg_edit(uuid: str, stage: Stage, instruction: Annotated[str, pydantic.Field(min_length=1, description="what to change in the knowledge graph, in words")]) -> dict[str, object]:
    """Edit a session's knowledge graph at a stage as a job. The stack drops later stages when the edit rewinds them."""
    _check_stage(stage)
    records = await session_records(uuid)
    record = records.get(stage)
    if not isinstance(record, dict) or record.get("kg") is None:
        raise ToolError(f"stage {stage} of session {uuid} holds no knowledge graph", hint="stages with a graph show has_kg in airchitect_session")
    before = stamps(records)
    await acall("POST", f"{ROUTES['prompt']}/edit/rr", {"uuid": uuid, "stage": stage, "instruction": instruction})

    async def body(handle: JobHandle) -> dict[str, object]:
        run = Run(handle.job.id)
        run.update(uuid=uuid, stage=stage, instruction=instruction)
        handle.progress(stage="edit", run_dir=str(run.dir))
        started = time.monotonic()
        while time.monotonic() < started + DEADLINES["edit"]:
            after = await session_records(uuid)
            prompt_record = after.get("prompt") or {}
            if prompt_record.get("status") == "error" and prompt_record.get("timestamp") != before.get("prompt"):
                run.save("edit", after)
                raise ToolError(f"the knowledge graph edit failed in the stack: {prompt_record.get('error')}")
            if stamps(after) != before:
                run.save("edit", after)
                handle.log(f"edit applied after {time.monotonic() - started:.0f}s")
                return {"uuid": uuid, "stage": stage, "stages": {name: summarize(record) for name, record in after.items()}, "run_dir": str(run.dir), "cost": COST_UNAVAILABLE}
            await asyncio.sleep(POLL_S)
        raise ToolError(f"the knowledge graph edit did not land within {DEADLINES['edit']:.0f}s", hint=f"airchitect_session {uuid} shows the stages")

    return JOBS.start_task("airchitect", f"kg edit {uuid}", body).as_dict()


@domain.tool(destructive=True, open_world=True)
async def airchitect_export(uuid: str, name: WorldName, overwrite: bool = False, nav_radius: Annotated[float | None, pydantic.Field(description="robot radius for NAV@r, the nav_at_r default when omitted")] = None) -> dict[str, object]:
    """Export a session's newest plan (scenario, else decoration, else floor plan) into the writable world root as a job, with the generate checks."""
    path_component(name)
    await asyncio.to_thread(target_dir, name, overwrite)
    plan, scenario, source = latest_plan(await session_records(uuid))

    async def body(handle: JobHandle) -> dict[str, object]:
        run = Run(handle.job.id)
        run.update(uuid=uuid, name=name, source=source)
        handle.progress(stage="export", run_dir=str(run.dir))
        return {**(await export_world(handle, run, uuid, plan, scenario, name, overwrite, nav_radius)), "source_stage": source}

    return JOBS.start_task("airchitect", f"export {name}", body).as_dict()


def bundle(world: str) -> bytes:
    """A world as the zip airchitect ingests: world.yaml with linked levels written out, its KG and its graph.yaml."""
    from arena_simulation_setup.tree.World import WorldIdentifier, save_snapshot

    view = WorldIdentifier(world).resolve_sync()
    loaded = view.load(validate=False)
    with tempfile.TemporaryDirectory(prefix="airchitect_ingest_") as tmp:
        root = save_snapshot(loaded, Path(tmp) / "world")
        graph = Path(view.path) / "graph.yaml"
        if graph.is_file():
            shutil.copyfile(graph, root / "graph.yaml")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
    return buffer.getvalue()


@domain.tool(open_world=True)
async def airchitect_ingest(world: Annotated[str, pydantic.Field(description="Arena world name, resolved like the runtime does")]) -> dict[str, object]:
    """Seed a new airchitect session from an Arena world and return its uuid."""
    try:
        data = await asyncio.to_thread(bundle, world)
    except FileNotFoundError as exc:
        raise ToolError(f"world {world!r} does not load: {exc}", hint="assets_ls with kind world lists the resolvable worlds") from exc
    status, raw = await asyncio.to_thread(http, "POST", "world/ingest", data=data, content_type="application/zip", timeout=120.0)
    if status != 200:
        raise ToolError(f"world/ingest answered HTTP {status}: {raw.decode(errors='replace')[:2000]}")
    answer = json.loads(raw)
    return {"world": world, "uuid": answer.get("uuid"), "domain": answer.get("domain"), "graph_source": answer.get("graph_source"), "bytes": len(data)}
