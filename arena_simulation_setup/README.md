# arena_simulation_setup

Worlds, assets, environment configs, and the `Identifier` registry for Arena.
Simulators and `task_generator` resolve world layouts, 3D models, materials,
pedestrian agents, and environment templates through the types defined here.

## Guides

- [Identifier registry](src/arena_simulation_setup/tree/README.md): `Identifier`
  ABC, resolver hierarchy, shipped Identifier types, adding a new one.
- [Worlds](worlds/README.md): per-world directory layout, `world.yaml` schema,
  scenario schema, local asset overrides, `WorldIdentifier` resolution.
- [Environment configs](configs/environment/README.md): what an environment
  YAML binds; obstacle group schema; `EnvironmentIdentifier` resolution.
- [Wall presets](configs/walls/README.md): `WallDescription` YAML schema;
  sub-wall types; how `world.yaml` references a wall preset by `kind:`.
- [Authoring a world](AUTHORING.md) — end-to-end guide: create dir, author
  `world.yaml`, generate map, add scenario, validate.
- [Generative worlds](src/arena_simulation_setup/generative/world/README.md):
  the generator registry, the layout IR they compile to, the sketch and letter
  grammars, and the ROS surface the rviz panel drives.
- [Generative scenarios](src/arena_simulation_setup/generative/scenario/README.md):
  the `generate_scenario` verbs, zone roles from the world's KG, the base
  populations, the prompt-driven timeline cases and the benchmark suites over them.

## CLI

| Script | Usage | Effect |
|---|---|---|
| `generate_scenario` | `generate_scenario {zones,classify,populate,promptgen,suite} <world> [--write] ...` | Prints a world's zone roles, asks a model for the missing ones, writes the four base scenarios, realizes prompts on them as timeline cases and writes a benchmark suite over the result. Writing verbs are dry runs without `--write`. See the [generator README](src/arena_simulation_setup/generative/scenario/README.md). |
| `generate_world` | `generate_world "<prompt>" [-e endpoint] [-o outdir]` | Posts a natural-language prompt to a generation server; extracts the returned zip into `worlds/<outdir>/` |
| `model_staging` | `model_staging <install_dir>` | Creates symlinks in `<install_dir>` for all known robot models and writes a `deps` file |
| `nav_at_r` | `nav_at_r <world name or dir> [--robot NAME\|DIR \| --radius M] [--walls-only] [--start X,Y] [--image PNG] [--json]` | Prints the NAV@r reachability of a world per level for a footprint radius: value, traversable components, largest share and cut-off zones. See [AUTHORING.md](AUTHORING.md#reachability-navr). |
| `preload_world` | `preload_world [<world_name>] [--no-scenarios] [--dry-run] [--sounds]` | Resolves every identifier the world and its scenarios reference, downloading what is missing. `--dry-run` reports without transferring. `--sounds` adds the default sound of every sound kind, alone or with a world, and `arena launch` passes it under `acoustics:=arena`. Reached from the CLI as `arena preload`. |
| `touch_world` | `touch_world <world_name> [--all \| --migrate] [--resolution N] [--assets color] ...` | Renders a preview `map.png` + `map.yaml` into the world dir for inspection; `--all` re-saves `world.yaml` as canonical format 2 with `kg.yaml` and `map/<level id>.png`, `--migrate` rewrites a format 1 world in place. The runtime renders its own map in-process, so this is an authoring aid, not required after editing. |

Scripts live under [scripts/](scripts). To fetch a single asset by identifier, use `arena asset pull <kind> <name>`. `arena asset expire [<kind> <name>]` marks cached assets expired, all of them when bare, so each is downloaded again on its next use.

## Human assets and skeletal articulation

### arenian actor

`arenian` is the default pedestrian. Like every other human it is an `arena_humans`
bundle (MakeHuman skin, CMU clips, `cmu_mb` rig) fetched from the asset bucket, not
shipped in the repo. The SDF `<actor>` form is required so Gazebo registers the skinned
mesh and the clips that `PedSkeletonPlugin` poses.

**gpu_lidar implication.** Arena's lidar sensor is `gpu_lidar`, a rendering
sensor that forces a server-side render scene even in headless mode.  Because
`PedSkeletonPlugin` positions and animates the actor in that scene, the robot's
lidar perceives a moving, articulated body, not a static box or capsule, making
the in-sim animation perceptually load-bearing in headless RL training, not only
cosmetic.

`PedSkeletonPlugin` (package `arena_gz_plugins`) is the gz-sim 8 C++ plugin that
subscribes `arena_peds`, positions each actor via `TrajectoryPose`, and poses its
skeleton per bone via `BoneTransforms` from the wire `joint_state`, through the
same bone map as the Isaac provider (`utils/msgs/arena_people_msgs/bone_map.json`), so
both simulators render the same articulated gait.  `BoneTransforms` is an Arena
addition to gz-sim carried by the `gz-sim` overlay in `_meta/repos/gazebo.repos`
and built by `arena feature gazebo update`.  Without the overlay or a bone map the
plugin falls back to scrubbing the `walk` clip via `AnimationTime`.  Enabled by
default via the `ped_skeleton_enabled` launch arg in `arena_bringup`.  The plugin
lives in its own repo, [arena_gz_plugins](https://github.com/voshch/arena_gz_plugins),
pulled into the workspace via `_meta/repos/gazebo.repos`.

## Internals

Key `Identifier` types from `arena_simulation_setup.tree`:

- **`WorldIdentifier`** ([tree/World/view.py](src/arena_simulation_setup/tree/World/view.py)):
  resolves a world name to `ASS_DIR / 'worlds' / <name>`; `.load()` returns
  a `WorldView` that exposes `view.load()` (the `World`), `view.map_yaml(level_id)`
  and `view.scenario`.
- **`ObjectIdentifier`** ([tree/assets/Object.py](src/arena_simulation_setup/tree/assets/Object.py)):
  resolves `<domain>/Object/<name>` to a `ModelWrapper` (SDF + USD).
- **`HumanIdentifier`** ([tree/assets/Human.py](src/arena_simulation_setup/tree/assets/Human.py)):
  resolves `<domain>/Human/<name>` to a `ModelWrapper` (SDF only).
- **`MaterialIdentifier`** ([tree/assets/Material.py](src/arena_simulation_setup/tree/assets/Material.py)):
  resolves `<domain>/Material/<name>` to a `Material`; supports `tint`
  modifier.
- **`WallIdentifier`** ([tree/Wall.py](src/arena_simulation_setup/tree/Wall.py)):
  resolves `<domain>/Wall/<name>/<name>.yaml` to a `WallDescription`;
  realizes to `(WallSegment[], Obstacle[])`.
- **`EnvironmentIdentifier`** ([tree/configs/environment.py](src/arena_simulation_setup/tree/configs/environment.py)):
  resolves `configs/environment/<name>.yaml` to an `EnvironmentDescription`
  (untyped dict).
- **`ParametrizedIdentifier`** ([tree/configs/parametrized.py](src/arena_simulation_setup/tree/configs/parametrized.py)):
  resolves an XML file in `arena_bringup/configs/parametrized/` to a
  `ParametrizedConfig` with `STATIC`, `INTERACTIVE`, and `DYNAMIC` obstacle lists.

See [tree/README.md](src/arena_simulation_setup/tree/README.md) for resolver
order, the `DynamicPaths` singleton, and how to add a new Identifier.

[utils/llm.py](src/arena_simulation_setup/utils/llm.py) holds the LLM client:
`from arena_simulation_setup.utils.llm import genai` (or `openai`) imports the
library on first access, redirected to a local endpoint when `LLM_API_ENDPOINT`
is set and passed through unchanged otherwise. `DEFAULT_MODEL` is
`$ARENA_PROMPT_LLM`, default `gemini-3.5-flash`, the model of
`tm_obstacles:=prompt`.
