# arena_runtime

Workspace folder containing two co-located ROS packages:

- [arena_runtime/](arena_runtime/README.md): Python package; owns `arena_node` (the runtime binary), `EnvRegistry`, `HoldRegistry`, `CleanupManager`, and the per-simulator adapters (gazebo, isaac, flatsim, dummy).
- [arena_runtime_msgs/](arena_runtime_msgs/README.md): rosidl interfaces consumed by `arena_node` and per-env consumers.

`task_generator` (the per-env episode loop) depends on both; the dep graph is one-way.
