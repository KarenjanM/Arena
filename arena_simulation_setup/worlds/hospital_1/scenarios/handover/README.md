# Hospital 1 handover

The Jackal drives to a fixed stop 0.81 m in front of a waiting receiver in the central hallway. The UR5 presents its tool, holds for 1.2 seconds, then returns to stow. The bottle profile shapes the right-hand pose. The scene contains the robot and receiver without a physical bottle.

## Run in Arena

Use two shells in the same sourced Arena Isaac environment. Build `arena_simulation_setup` to install the new scenario if the workspace uses the installed worlds. The workspace also needs the build of `arena_people_msgs` that provides `Handover`.

In each shell, set the path to the scenario directory from the workspace root.

```bash
handover_dir="$PWD/src/Arena/arena_simulation_setup/worlds/hospital_1/scenarios/handover"
```

Start the one-attempt driver first so it can see the approach.

```bash
python3 "$handover_dir/driver.py" --ros-args -r __ns:=/arena/env_0 -p use_sim_time:=true
```

In the other shell, launch the scenario in a fresh Arena runtime so the environment is `env_0`.

```bash
arena launch sim:=isaac world:=hospital_1 human:=arena 'robot:=jackal[arm=ur5]' \
  robot.mobile:=nav2 robot.arm:=moveit robot.mobile.params_overlay:="$handover_dir/nav2.yaml" \
  task.robots:=scenario task.obstacles:=scenario task.scenario.file:=handover task.episode.count:=1
```

The driver reads `arena_peds` and TF in the stream's map frame. It publishes `attend`, `ready`, `reach`, `take` and `retract` to `human/handover` under the environment namespace. `take` requires the tool to be near its authored goal and within 9.5 cm of the palm for 0.20 seconds. A phase that exceeds the receiver's timeout goes to `retract`. The driver logs each transition with the measured gap. Restart it for a new episode.

The progress video in `data/gesture_qa/handover_task9/hospital_1_handover.mp4` uses Arena's receiver animation and an offline Jackal/UR5 kinematic replay. Live Nav2, MoveIt, ROS transport and Isaac rendering still need an integration run.
