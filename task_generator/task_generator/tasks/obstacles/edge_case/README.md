# `edge_case` obstacle task mode

Runs the **effects** a scenario's `edge_case` block declares, on top of the population the
scenario file already carries. The block is `edge_case.yaml` beside the format 2 `scenario.yaml`
(format 2 rejects keys it does not know); a format 1 `scenario.yaml` may still carry it as its own
`edge_case:` key (`scenario_block.read`). The offline
generator that writes these scenarios is
[`arena_evaluation.generation.promptgen`](../../../../../arena_evaluation/arena_evaluation/arena_evaluation/generation/promptgen/README.md).

Who is present is baked into `scenario.yaml` by the generator; until a case runs, the
population behaves exactly as the base scenario wrote it. Everything the case *changes* -
where people go, how they are tuned, where a formation stands - is an effect with an onset,
and what depends on the live robot is solved at reset:

| Effect | What it does | Runtime |
|---|---|---|
| `intercept` | one pedestrian placed a short lead-in from a designed point on the robot's **planned** route, standing until the robot is due there within its walk time (`robot_eta`) | [`geometry.py`](geometry.py), [`pathing.py`](pathing.py) |
| `rally` | send the agents in scope to a zone or point, once | [`waypoint_driver.py`](waypoint_driver.py) |
| `shuffle` / `scatter` | throw each agent a new point near where it started, every `period` s / once | same |
| `track_robot` | re-issue the robot's position as the goal | same |
| `hold` | pin the agents in scope, then release them to their own goal | same |
| `stimulus` | raise a `need` (to 100 x `intensity`, default 1) on the agents in scope, once, through HumanSim's `notify_stimulus`; each agent reacts after its own `reaction_time`. A behaviour-tree type with a `transitions` entry keyed on that need switches sequence: behaviour changes without a respawn, parameters do not | same, [`impl.py`](impl.py) `_stimulate_agents` |
| `objects:` | a hand-authored object timeline (not written by the generator) | [`object_driver.py`](object_driver.py) |

Every effect carries a `when`: `{at: s}`, `{robot_within: m, of: zone|[x,y]}`,
`{robot_beyond: m, of: ...}`, or `{route_fraction: f}` (fires when the robot comes within 1.5 m of that point of its route).

Requires `human:=arena`; registered from `ArenaHumanSimulator._register_task_modes`.

---

## Quick start

```bash
arena launch sim:=gazebo world:=hospital_1 robot:=jackal human:=arena \
      task.robots:=edge_case task.obstacles:=edge_case headless:=true \
      task.episode.auto_reset:=true task.scenario.file:=normal__u_004
```

`normal__u_004` stands for any generated case (`<base>__<prompt id>`); the repo ships none.

Write `task.scenario.file:=`; `task.scenario:=` is a deprecated alias of it.
A scenario without a block runs the base population alone (the control arm). Records:

```bash
cat $ARENA_DATA_DIR/edge_case/cases.jsonl | python3 -m json.tool     # what was built
cat $ARENA_DATA_DIR/edge_case/scores.jsonl | python3 -m json.tool    # what happened
```

Newly generated scenario directories need `colcon build --packages-select
arena_simulation_setup --symlink-install` before the runtime sees them.

---

## The block

`edge_case.yaml`, the block's keys at the top level:

```yaml
id: normal__u_004
prompt_id: u_004
steps: [A:busy, D:interaction_unbroken, C:approach]
note: It is visiting hours and the place is packed. ...
effects:
- {type: intercept, label: C:approach, profile: cart, model: male_adult_construction_01,
   angle: 180.0, route_fraction: 0.3, speed: 0.9, after: continue, waypoint_mode: once}
- {type: rally, label: A:evacuation, scope: [normal_1, normal_2], target: central_hallway, when: {at: 8.0}}
- {type: scatter, label: D:formation_break, scope: [d_1, d_2, d_3], radius: 2.0, when: {robot_within: 3.0, of: [12.1, 4.0]}}
- {type: stimulus, label: A:evacuation, scope: [break_1, break_2], need: alarm, when: {at: 8.0}}
formation: {centre: [12.1, 4.0], radius: 1.8, variant: circle, at: {at: 8.0}}   # where Step D's group forms
decisions: {...}      # the generator's typed decisions, for review
provenance: {...}     # prompt hash, model, response hashes, generator version
```

Grammar: [`effects.py`](effects.py). Reader: [`scenario_block.py`](scenario_block.py). Both are
strict - an unknown key, an unknown effect type or a scope naming an agent the scenario did
not place **aborts the episode with the reason recorded**, because a case that silently did
nothing is indistinguishable from a case with no effect.

### `intercept`

| Field | Default | Meaning |
|---|---|---|
| `profile` | adopt from the population | agent type: a builtin (`adult`, `elder`, …) or `./types/x.yaml` beside the scenario |
| `model` | adopt from the population | mesh; `arenian` when the population is empty |
| `angle` | 180 | travel direction relative to the robot's heading at the encounter: 180 head-on, 90 crossing from its right, 0 overtaking |
| `route_fraction` | 0.5 | where on the robot's planned route the meeting is designed |
| `speed` | the profile's nominal | m/s |
| `offset` | 0 | seconds the pedestrian arrives after the robot |
| `lead_out` | 3.0 | metres it continues past the encounter |
| `search` | 90 | max angular deviation when hunting free space; 0 aborts instead |
| `after` | `continue` | `stop_facing` (halt at the encounter for `duration`), `follow` (track the robot for `duration`), `veer` (turn off to `veer_to`), `stand` (placed *on* the route, stationary, released after `duration`) |
| `waypoint_mode` | `reverse` | `once` for exactly one designed crossing |

The geometry is solved at reset against the robot's route as planned through the occupancy
grid ([`pathing.py`](pathing.py)); requested and achieved angle/fraction are both recorded.
Injected agents are named `edge_0`, `edge_1`, … in block order.

### Scopes

`all` (every placed agent), `injected` (the `edge_*` agents), or an explicit list of names.

### `stimulus`

Every agent in scope must be a `mode: behavior_tree` type that declares the `need`, or the case
**aborts**: HumanSim ignores an undeclared need and a `mode: simple` agent has no tree to switch,
so either would be a silent no-op. The generator's evacuation writes this pairing: talkers get a
derived `./types/talker__evacuate.yaml` (an `alarm` need latched at 0, an `evacuate` sequence
walking to the exit, a transition into it from every other sequence) and the stimulus, while the
plain walkers get the `rally`. A `rally` on a behaviour-tree agent swaps its movement for a
waypoint route its still-ticking tree then writes commands onto, which can crash HumanSim; keep
trees out of route rewrites.

---

## Parameters

Namespace `task.edge_case.*`, declared in [`__init__.py`](__init__.py).

| Parameter | Default | Meaning |
|---|---|---|
| `robot_speed` | 1.0 | Speed in m/s the encounter geometry and object events are designed against; the intercept release waits for the measured arrival time (`robot_eta`). |
| `read_scenario_block` | `true` | Off runs the base population alone. |
| `prompt` | `""` | **The RViz input.** A natural-language situation; when set, the scenario is generated from it at reset by the same generator as `evaluation generate promptgen` (model answers cached by content, so a repeated prompt is instant), written into the world's installed `scenarios/` as `<base>__rviz_<hash>` and run instead of `task.scenario.file`. The RViz task-mode panel shows this as a text field under `edge_case`. |
| `prompt_base` | `""` | Base the prompt is realised on; empty uses `task.scenario.file` (or its base when that is itself a prompt scenario). |
| `record_dir` | `""` | Where `cases.jsonl` / `scores.jsonl` go. Empty → `$ARENA_DATA_DIR/edge_case`. |
| `score`, `score_rate_hz`, `score_scope` | `true`, `10`, `auto` | Criticality panel per episode; `auto` scores the case's own agents - injected ones and the encounter (C) and formation (D) agents the block's `owned` names - when it has any, everyone otherwise (scope `injected`). |
| `objects`, `objects_rate_hz` | `""`, `5` | Object timeline override. |

There is no knob surface any more: a case is its scenario file. To sweep, list scenarios.
To switch case on a running stack, set the scenario and reset; the next episode builds it:

```bash
ros2 param set /arena/env_0/task_generator_node task.scenario.file normal__u_021
ros2 service call /arena/env_0/task_generator_node/lifecycle/reset_episode task_generator_msgs/srv/ResetEpisode "{}"
```

---

## Records

`cases.jsonl` (written at build): identity (`scenario`, `case_id`, `prompt_id`, `steps`), the
effects as designed, the resolved plans (owned agents, targets), the first intercept's
designed and achieved geometry (`spawn_pose`, `ped_goal`, `encounter_point`, `t_encounter`,
`approach_angle_requested/achieved`, …), `release_eta` (the walker stands at most `LEAD_IN_M` = 4 m from its encounter point - less when the lead-in does not fit in free space or would start inside the discs reserved around the robot's endpoints (`spawn_valid`), at least 3 m - and walks when the robot, at the speed it drove over the last `SPEED_WINDOW_S` = 2 s, is due at the encounter point within its walk time: no robot speed is assumed. Released open-loop from the full lead-in, walkers crossed the encounter point 8-28 s before the robot in the 2026-09 campaign; a standing blocker's `duration` likewise starts when the robot is `STAND_TRIGGER_M` = 3 m from it), `robot_hold` (sim seconds `tm_robots:=edge_case` holds the robot at its start - informational: every plan's clock, and so every onset and interception time, starts when the robot is **under way**, `ROBOT_DEPART_M` = 0.3 m from where it stood at arming, so neither the hold nor nav2's first-plan delay shifts a case), `invalid_spawns`, and `status` (`ok` / `baseline` /
`aborted` with `reason`).

`scores.jsonl` (written when the episode ends, at the next reset or at teardown): the panel
from [`criticality.py`](criticality.py) plus `waypoint_outcomes` - per plan, when the gate
fired (`t_trigger`; `None` with a non-empty `trigger` means the robot never came near, i.e.
the effect **did not run**), when the first rewrite went out, and when a hold released. When the
block names the agents Step D owns (`owned: {D: [...]}`), the row also carries `formation`: whether
those members were seen, when every one of them was first within the block's `formation.radius`
of its `formation.centre` (`formed_at`; None if they never all arrived; the centre is realised into the map frame first), how long that lasted
(`held_s`), when the first one left (`broke_at`), and `arrived` - the most members within the radius at
once (a line five of six reached is not "never formed") - the Type D success criterion,
positional because the cast is plain profiles walked into place by a rally.

Pair with [`tm_robots:=edge_case`](../../robots/edge_case/README.md): it assigns the goal
during reset (so the route exists when effects are built), traverses repeatedly, and reports
*blocked* as an outcome.

---

## Layout

| File | Role |
|---|---|
| [`impl.py`](impl.py) | `TM_EdgeCase` - population → `build` → arm → record |
| [`build.py`](build.py) | the case built against one episode's `Scene` (route, zones, free space, agent types): intercepts and armed plans |
| [`effects.py`](effects.py) | the block grammar |
| [`scenario_block.py`](scenario_block.py) | reading the block from the active scenario |
| [`waypoints.py`](waypoints.py), [`waypoint_driver.py`](waypoint_driver.py) | resolved route-rewrite plans and the driver that fires them, ticked on sim time from the node's event loop |
| [`triggers.py`](triggers.py) | proximity gates |
| [`geometry.py`](geometry.py), [`pathing.py`](pathing.py) | the interception solver and the route planner |
| [`objects.py`](objects.py), [`object_driver.py`](object_driver.py), [`timelines/`](timelines/README.md) | object timelines |
| [`agent_types.py`](agent_types.py) | agent types through HumanSim's loader |
| [`scoring.py`](scoring.py), [`criticality.py`](criticality.py) | the per-episode panel: `score_scope`, `score_episode` and the sampling node |
| [`provenance.py`](provenance.py) | `cases.jsonl` |

Tests: `tests/test_edge_case_*.py` and `tests/ros/test_edge_case_*.py` call these functions on real
inputs (`test_edge_case_build.py` builds cases from parsed blocks and real `WorldMap`s). The mode's
wiring - reset, the driver loops, the scorer node - runs only in `tests/sim/test_edge_case_in_sim.py`.
