# Generative scenarios

The scenario generator writes plain format 2 scenarios for a world: four base populations, prompt-driven edge cases on top of them, and a benchmark suite over the result. A generated case needs no runtime of its own. The stock `scenario` task modes run it, its pedestrians are behavior trees that switch on a need, and the scenario `timeline` raises those needs.

```
world + zone roles --populate--> bases --promptgen--> <base>__<prompt_id> --suite--> suite.yaml --benchmark--> episodes
```

## CLI

`generate_scenario <verb>`, or `python3 -m arena_simulation_setup.generative.scenario.cli <verb>`. A world is read and written in the first source root that has it: the `ARENA_WORLD_PATH` roots, then the source tree's `$ARENA_DIR/arena_simulation_setup/worlds`. A world in no source root is read where `WorldIdentifier` resolves it (the installed share, a composed `a,b` world, a world bucket), and a writing verb refuses it and names where it resolved, so copy it into a source root to write it. `--worlds DIR` reads and writes `DIR/<world>` only. A write under an install prefix (`AMENT_PREFIX_PATH`, `COLCON_PREFIX_PATH`) or `$ARENA_ASSETS_DIR` is refused. A world written outside the `ARENA_WORLD_PATH` roots reaches the runtime after a rebuild of arena_simulation_setup. Writing verbs are dry runs without `--write`.

| Verb | Writes | Purpose |
|---|---|---|
| `zones <world>` | - | each zone's role (work, break, transit, private, unknown) |
| `classify <world...>`, `classify --all` | `kg.yaml` of each world | ask a model for the roles no KG fact and no substring rule covers |
| `populate <world>` | `scenarios/{normal,break,night,empty}/` | the four base scenarios, on a base route the planner finds clear by `--robot-radius` (default 0.35 m) |
| `promptgen <world>` | `scenarios/<base>__<prompt_id>/` | realize every prompt on every base (`--ids` picks prompts, `--oracle` needs no model) |
| `suite <world...>` | `<name>.yaml` under `$ARENA_DIR/arena_evaluation/arena_evaluation/configs/benchmark/suites`, the source tree the benchmark runner reads after the installed share (an installed suite of the same name wins until arena_evaluation is rebuilt, and `suite` says so), or `--out` | one stage per (world, scenario) |

`promptgen` also takes the robot's drive: `--robot-speed` (the speed every interception release is designed at, default 0.5 m/s), `--traversals` (N round trips are 2N-1 legs, default 1), `--hold` (seconds at the start before the first leg, default none) and `--tolerance` (goal radius of every leg, default the runtime's goal tolerance).
Its prompts are the sets in [`promptgen/prompts/`](promptgen/prompts/), every `universal*.yaml` and the world's own `<world>.yaml`, plus the `--prompts` files. It exits 1 when a prompt crashed or no case came out `ok` or `partial`, a missing or unloadable base included.

```bash
generate_scenario populate hospital_1 --write --prune
generate_scenario promptgen hospital_1 --write -v
generate_scenario suite hospital_1 office_1 --robot jackal --episodes 3 --name edge_case_core --write
arena evaluation benchmark --suite edge_case_core --contest edge_case_differential
```

## Zone roles

Every zone has one role: `work`, `break`, `transit`, `private` or `unknown`. `populate` and `promptgen` place people by it, and `unknown` zones are populated as neither work nor break. A role comes from the world's [KG](../../../../AUTHORING.md#kg) first, as a `has_purpose` fact of the `space@1` package, and from the substring rules of [`zones.py`](zones.py) for a zone without one:

```yaml
format: arena-kg/1
uses: [core@2, space@1]
entities:
- {id: 'room:3_study'}
- {id: 'purpose:work'}
facts:
- subject: room:3_study
  verb: has_purpose
  object: purpose:work
  params: {by: llm, model: gemini-3.5-flash, prompt_sha: 12a42fdc3c28fdd0, why: A study or home office is used for working.}
```

`classify --write` asks a model about the zones neither covers and writes each confident answer into that world's `kg.yaml` as a fact with `params: {by: llm, model, prompt_sha, why}`, adding `space@1` to `uses` and creating the file when the world has none. A fact without `by: llm` is a human's: `classify` never replaces it, and it wins over a model's fact for the same zone. To accept or correct a proposal, set the role and drop `by: llm`. A stated `purpose:unknown` keeps a zone out of the model's queue. A zone with two human roles, or two model roles and no human one, is an error. A purpose that is no role (`purpose:storage`) is a valid KG fact for other consumers, and the generator reads it as no role. Linked levels bring their source world's facts along under their prefix.

## Outputs

A prompt scenario directory holds:

| File | Content |
|---|---|
| `scenario.yaml` | `arena-scenario/2`: `humans`, `robots`, `timeline`, checked by `parse_scenario` against the world's zones before it is written |
| `types/*.yaml` | the scenario's own HumanSim agent types, referenced as `./types/<name>.yaml` |
| `provenance.yaml` | `id`, `prompt_id`, `steps`, `note`, `decisions`, `owned` (`C`, `D`, `injected`), `formation` (`center`, `radius`, `variant`, `at`), `encounters` (`agent`, `point` per interception), `provenance` (prompt and answer hashes, generator version, status, dedup key). The runtime never reads it, the arena_evaluation metrics scope by it |
| `record.json` | the full decision record with warnings and failures |

Points in the timeline and in `provenance.yaml` are in the frame of the level the robot starts on, `robots[0].level`, which the generator always writes.

The robot's start and goal are pinned to the route the generator planned, and its `phases` are an optional hold phase, then the traversal legs, with `tolerance_radius` when `--tolerance` is set:

```yaml
robots:
- start: [3.33, 2.48, 1.131]
  phases:
  - {goto: [9.99, 17.0]}
  level: '0'
```

## Timeline

Every change a case makes at its onset is a timeline entry raising a need on named agents:

```yaml
timeline:
- at: 4.0
  stimulus:
  - {agents: [d_1, d_2, d_3], need: form}
- when: robot within 4.2 of 12.1,4.0
  stimulus:
  - {agents: [edge_0], need: go}
```

An onset `{at: s}` stays `at: s`. A proximity onset becomes `when: robot within <m> of <x>,<y>`, a point literal in the level frame. Without a point of its own it is measured to the centroid of the agents it scopes, a route fraction is measured to that point of the planned route within 1.5 m, and a `robot_beyond` onset is `not robot within`.

| Case | Agents | Need |
|---|---|---|
| A blackout, B intrinsic | a type-level `tunes:` entry held while the need is up | `blackout`, `intrinsic` |
| A evacuation, stampede | an `evacuate` or `rush` sequence to a point of the exit or work zone | `alarm` |
| B erratic | an `erratic` sequence of six points given up after `period` | `erratic` |
| D unbroken formation | a `form` sequence to the member's spot, standing with low repulsion | `form` |
| D formation break | `form` at 4 s, then `scatter` (a point `radius` from its spot, drawn from the seed) or `disperse` (a point of another zone) | `form`, `break` |
| C approach and blocking, D stubborn | `edge_<n>` waits at its spawn, then walks | `go` |

A plain walker a change reaches first becomes its own walking tree (`types/<type>__walk_<agent>`), its waypoints drawn once in each room where the runtime would draw them per episode, with the waypoint mode baked into the order.

## Interceptions

An interception is solved offline against the planned route ([`promptgen/encounter.py`](promptgen/encounter.py)): the encounter point at the requested route fraction, the spawn at most `LEAD_IN_M` = 4 m before it along the requested angle (180 head-on, 90 from the robot's right, 0 overtaking), and a lead-out of 3 m past it. When the walk or the spawn is blocked the solver tries other angles within 90 degrees, shorter lead-outs and nearby encounter fractions, and the achieved values are noted. The pedestrian stands at its spawn and walks on `go`, which the timeline raises when the robot is `speed * (walk time - offset)` from the encounter point, at least half a second out.

| `after` | Walk |
|---|---|
| `continue` | straight through the encounter to the lead-out point |
| `stop_facing` | to the encounter, facing the robot for `duration`, then the lead-out |
| `follow` | to the encounter, then along the robot's route for `duration` at its own speed, then the lead-out |
| `veer` | to the encounter, then to a point of the `veer_to` zone |
| `stand` | stands on the route, steps on `duration` s after the robot comes within 3 m |

## Grid and route

[`occupancy.py`](occupancy.py) renders the generator's level with `Level.render_grid` at 0.05 m, the runtime map's resolution, and stamps every object's footprint (`Obstacle.footprint()` with the world's assets resolvable through `world_assets`, else the 2 m box `WorldManager` stamps). A point is valid at 0.35 m clearance, the runtime's `task.episode.spawn.obstacle_clearance`, checked over the same square window the runtime validates a spawn in. The robot's route is planned on that grid by [`shared/pathing.py`](../../shared/pathing.py), furniture included, from endpoints snapped 0.7 m clear, cut to a 20 m leg and lengthened to the 15 m floor where the zones allow.

## Files

| Path | Purpose |
|---|---|
| `cli.py` | `generate_scenario` |
| `formats.py` | worlds in the working shape (`levels`, `first_level`), scenarios read (`read_scenario`) and rendered for writing (`scenario_text`) |
| `zones.py` | zone name to role: the world's `has_purpose` facts, then substring rules |
| `occupancy.py`, `zone_screen.py` | the level grid, and which zones may be homes (every admissible cell 1.5 m off the route) |
| `route.py`, `populate.py`, `materialize.py` | the base route, the four bases, the base writer and `GENERATED_MARKER` |
| `agent_types/talker.yaml` | the behavior-tree type copied beside every base with talkers |
| `suite.py` | scenario directories to a suite of `scenario` stages |
| `propose/` | the model call path through `arena_simulation_setup.utils.llm`, and model classification of zone names |
| `promptgen/` | prompt + world + base to a case: steps A, D, B+C decide, `realize.py` turns the design into pedestrians, trees and the timeline |

`promptgen/system_*.md` are build output: edit the `.tmpl.md` and run `python3 -m arena_simulation_setup.generative.scenario.promptgen.build_kb`, a test fails when the rendering is stale.

## Invariants

- **People are written in rooms, never coordinates.** `pose: laboratory`, `waypoints: [...]`, resolved at run time. The exceptions are the robot's pinned start and goal and the pedestrians a case injects.
- **Normal until the onset.** A case changes the crowd only through its timeline, except a parameter HumanSim has no `tune:` for (radius, need decay, step duration), which is pinned into the agent's type from the start.
- **Every agent block carries its type's radius.** HumanSim's adapter overrides an undeclared `radius` with 0.35 m, so `scenario_text` writes the type's mean.
- **The model runs at temperature 0**, every answer cached by content hash under `$ARENA_DATA_DIR/promptgen/llm_cache` (`$ARENA_WS_DIR/data`, else `/opt/arena_ws/data`, when `ARENA_DATA_DIR` is unset), so regeneration is byte-identical and free. An entry is written to a temporary file and renamed into place, and an unreadable one is asked again.
- **Generated files carry `GENERATED_MARKER`.** `populate --write` refuses to replace a scenario without it unless `--force`, and `--prune` removes only marked bases this run did not write.
- **Same world and seed reproduce byte-identically.** World variance, scatter points and waypoint draws come from RNGs keyed on world, agent, profile and seed.
