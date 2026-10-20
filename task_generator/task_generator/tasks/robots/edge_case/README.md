# `edge_case` robot task mode

Routed traversals with obstruction as a first-class outcome. Built to pair with
[`task.obstacles:=edge_case`](../../obstacles/edge_case/README.md).

```bash
arena launch sim:=gazebo world:=hospital_1 robot:=jackal human:=arena headless:=true \
      task.robots:=edge_case task.obstacles:=edge_case task.scenario.file:=<case>
```

Unlike its obstacle-side namesake this mode has **no HumanSim dependency**, so it registers
at import and works under any `human:=` backend. It is useful on its own wherever you want
repeated traversals or blocked-detection, not only for edge cases.

---

## Why not `explore`

`task.robots:=explore` actively fights the edge-case design in three ways:

| | `explore` | `edge_case` |
|---|---|---|
| Goal set | lazily, in `done` | during `reset()`, before obstacles are generated |
| Episode end | never (`done` is always `False`) | on completion, obstruction, or timeout |
| Route | random | the scenario's declared route, deterministic |

The first matters most. `Task._reset_episode` awaits `tm_robots.reset()` **before**
`tm_obstacles.reset()`, so the obstacle mode can design its effects against this episode's
route: the last goal of the request in `RobotManager.runner`, from the `start_pos` it was submitted with. Under `explore` the goal is still unset at that moment and
every route-designed effect aborts the case.

`task.robots:=scenario` fixes the first two and is a reasonable fallback. What it does not do
is repeat the crossing or treat obstruction as an outcome.

---

## What it does

**Route** — inherited from the robots scenario mode: the `robots:` block of whichever
scenario `task.scenario.file` names. Deterministic, and authored next to the crowd it has
to cross. Zone names work (`start: reception`), as do coordinates.

**Traversals** — the single start→goal can be expanded into `2N-1` alternating legs (out,
back, out, …) so the robot passes through the population repeatedly within one episode.
The default is 1 since 2026-08-29: every prompt case is designed against the single
start→goal leg (onsets, interceptions, formations along that route), a return leg only
replays the population with the design behind it, and the recordings want the episode to
end when the robot arrives. The legs replace the scenario's phases in the one request the
scenarios mode submits (`TM_Scenario._plan`), so its `start`, `conditions` and declared
pedestrians are kept.

**Obstruction watchdog** — tracks each robot's closest-ever approach to its current leg's
goal. If that high-water mark does not improve by `blocked_distance` for `blocked_timeout`
seconds, the robot counts as obstructed and the episode ends as FAILED with a reason.

Distance-to-goal rather than raw speed, deliberately: a robot legitimately pauses, rotates
in place, or backs out of a doorway, and none of that is obstruction. Failing to get any
closer for a sustained window is.

---

## Outcomes

Three terminal states, each mapping to a real episode result:

| Situation | Outcome | Reason |
|---|---|---|
| every traversal completed | `SUCCESS` | — |
| obstructed past the timeout | `FAILED` | `robot 'jackal' blocked: no progress toward (…) for 15s, still 2.31m away` |
| `task.episode.timeout` reached | `FAILED` | `timeout` (from `Task.is_done`, before this mode is asked) |

Because episodes now end on their own, an external driver can wait for the episode outcome —
it waits for the task mode's own outcome instead of a wall-clock window.

---

## Parameters

Namespace `task.edge_case_robot.*`.

| Parameter | Default | Meaning |
|---|---|---|
| `traversals` | 1 | Round trips through the population. N traversals is `2N-1` legs; 1 = the single start→goal leg. |
| `retries` | 1 | Attempts per leg. nav2 gives up on a leg it cannot make progress on and the phase policy is `continue`, so with 1 a blocked leg is skipped; with 3 the same goal is submitted three times before the route moves on - what a recording wants. |
| `blocked_distance` | 0.25 | Metres of improvement that count as progress. Jitter below this is not progress. |
| `blocked_timeout` | 15.0 | Seconds without progress before the robot counts as obstructed. |
| `hold` | 0 | Sim seconds the robot stands at its start before the first leg (a pose-less first phase; the watchdog ignores a robot inside its leg's tolerance). Rarely needed since 2026-08-29: the obstacles mode's case clock starts when the robot is under way (0.3 m from its start), so onsets count from the drive, not the reset; recordings run with 0. |
| `accept_within` | 0.6 | Goal tolerance of every leg (`tolerance_radius`), when wider than `task.episode.goto_pose.tolerance.radius`. nav2 refuses to plan into a goal cell inside the robot's inflation - a goal pinned 0.35 m from a desk left mpo700 "blocked" 0.23 m short for a whole recording - and the robot is as good as there. 0 = the robot's own tolerance. |
| `on_blocked` | `abort` | `abort` ends the episode as FAILED; `continue` logs and carries on. |

The **scenario file is read from `task.scenario.file`**, not from this namespace — so the
route and the crowd always come from the same scenario. See the note in `impl.__init__`:
this is why `TM_Scenario.__init__` is deliberately skipped.

---

## Gotchas

**The route endpoints are `forbid`den.** Inherited from the scenario mode: a disc of
`robot.safe_distance` around the start and every goal is excluded from obstacle placement, so the crowd cannot spawn on top of the robot. On a small zone this is significant — it is
what pushed the generated worlds' `default` scenario to stop spawning anyone in
`reception`, which is only 20 m² and holds up to 8 pieces of furniture.

**The start is the robot's last placement.** The obstacle mode reads `RobotManager.start_pos`,
which `move` sets after `tm_obstacles.reset()`. On the first episode after launch the robot has
not been placed yet and route-designed effects abort; after a scenario switch the first case
is designed from the previous scenario's start.

**`traversals` interacts with `Robot.TIMEOUT`.** Several traversals of a long route may not
fit in the default timeout, which surfaces as a `timed out` failure rather than a blocked
one. Raise the timeout or lower `traversals` if you see that.

---

## Layout

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | registration + `declare_schema` |
| [`impl.py`](impl.py) | `TM_EdgeCase(TM_Scenario)` — route expansion, watchdog, outcomes |

Tests: `tests/test_edge_case_robot.py` (pure: phase expansion, watchdog) and
`tests/ros/test_edge_case_robot_registry.py` (registration, schema, param binding).
