from __future__ import annotations

import importlib
import threading
import time

ENV = 7
PLAIN_ENV = 8
ROBOT = "jackal_0"
WORLDS = ["map_empty", "small_warehouse"]
SCENARIOS = {"map_empty": ["crowd", "default"], "small_warehouse": ["aisles"]}


def fqn(env_id: int) -> str:
    return f"/arena/env_{env_id}/task_generator_node"


def _latched(depth: int = 1):
    import rclpy.qos

    return rclpy.qos.QoSProfile(depth=depth, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)


def _peer_class():
    import rclpy
    import rclpy.action
    from arena_humansim_msgs import srv as humansim_srv
    from arena_humansim_msgs.msg import AgentMeta
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from arena_people_msgs.srv import MovePedestrians
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_simulation_setup.shared.task import TaskRequest
    from rclpy.callback_groups import ReentrantCallbackGroup
    from std_msgs.msg import Bool, String
    from std_srvs.srv import Empty, Trigger
    from task_generator_msgs import srv
    from task_generator_msgs.action import RunEpisode
    from task_generator_msgs.msg import EpisodeRecord, RobotFleet, RobotState, SemanticEntityState, SemanticSnapshot

    from task_generator.constants.runtime import Configuration
    from task_generator.tasks.registry import MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES, walk_schemas

    for package in ("robots", "obstacles", "modules"):
        importlib.import_module(f"task_generator.tasks.{package}")

    class TaskGeneratorPeer(ROSParamServer):
        """Serves the task generator's interfaces with the real Configuration and mode schemas declared."""

        def __init__(self, env_id: int, human: str) -> None:
            super().__init__("task_generator_node", namespace=f"/arena/env_{env_id}", parameter_overrides=[rclpy.Parameter("world", value="map_empty"), rclpy.Parameter("human", value=human)])
            self.conf = Configuration(self)
            self.declare_parameter("task.episode.auto_reset", False)
            self.declare_parameter("task.episode.fail_on_collision", False)
            walk_schemas(self)
            self.fqn = fqn(env_id)
            self.env = f"/arena/env_{env_id}"
            self.group = ReentrantCallbackGroup()
            self.lock = threading.Lock()
            self.episode_id = 0
            self.resets: list[tuple[str, int]] = []
            self.queued: list[object] = []
            self.spawns: list[tuple[str, object]] = []
            self.moves: list[object] = []
            self.despawns: list[str] = []
            self.semantic_writes: list[tuple[str, str, str]] = []
            self.sounds: list[object] = []
            self.submitted: list[tuple[str, str]] = []
            self.phase_count = 0
            self.streamed: list[object] = []
            self.ped_moves: list[object] = []
            self.humansim: list[tuple[str, object]] = []
            self.running_goal = False
            self.cancelled = False
            self.pub_episode = self.create_publisher(EpisodeRecord, f"{self.fqn}/state/episode", _latched(20))
            self.pub_queue = self.create_publisher(EpisodeRecord, f"{self.fqn}/state/queue", _latched(10))
            self.pub_resetting = self.create_publisher(Bool, f"{self.fqn}/state/resetting", _latched())
            self.pub_world = self.create_publisher(String, f"{self.fqn}/state/world", _latched())
            self.pub_semantics = self.create_publisher(SemanticSnapshot, f"{self.fqn}/state/semantics", _latched())
            self.pub_robots = self.create_publisher(RobotFleet, f"{self.fqn}/state/robots", _latched())
            self.pub_peds = self.create_publisher(Pedestrians, f"{self.env}/arena_peds", 10)
            self.pub_resetting.publish(Bool(data=False))
            self.pub_world.publish(String(data="map_empty"))
            fleet = RobotFleet()
            state = RobotState()
            state.descriptor.name = ROBOT
            state.descriptor.model = "jackal"
            state.descriptor.ns = f"{self.env}/{ROBOT}"
            fleet.robots.append(state)
            self.pub_robots.publish(fleet)
            snapshot = SemanticSnapshot(env_id=env_id, world="map_empty")
            snapshot.entities.append(SemanticEntityState(entity=f"env_{env_id}/door_1", kind="door", discrete_names=["state"], discrete_values=["closed"], predicate_names=["locked"], predicate_values=[False], members=["adult_1"]))
            self.pub_semantics.publish(snapshot)
            self.pub_queue.publish(self.record(EpisodeRecord.QUEUED, episode_id=0))
            self.publish_record(EpisodeRecord.RUNNING, 1)

            def service(srv_type: type, name: str, callback: object) -> None:
                self.create_service(srv_type, f"{self.fqn}/{name}", callback, callback_group=self.group)

            service(srv.ResetEpisode, "lifecycle/reset_episode", self.on_reset)
            service(Empty, "lifecycle/wait_for_world", lambda request, response: response)
            service(srv.QueryWorlds, "query/worlds", lambda request, response: srv.QueryWorlds.Response(ids=WORLDS))
            service(srv.QueryScenarios, "query/scenarios", lambda request, response: srv.QueryScenarios.Response(ids=SCENARIOS[request.world or "map_empty"]))
            service(srv.QueryRobots, "query/robots", lambda request, response: srv.QueryRobots.Response(ids=["jackal", "burger"]))
            service(srv.QueryStaticObstacles, "query/static_obstacles", lambda request, response: srv.QueryStaticObstacles.Response(ids=["shelf"]))
            service(srv.QueryDynamicObstacles, "query/dynamic_obstacles", lambda request, response: srv.QueryDynamicObstacles.Response(ids=["adult"]))
            service(srv.QueryEnvironments, "query/environments", lambda request, response: srv.QueryEnvironments.Response(ids=[]))
            service(srv.QueryParametrizeds, "query/parametrizeds", lambda request, response: srv.QueryParametrizeds.Response(ids=["crowd"]))
            service(srv.QueryTaskModes, "query/task_modes", self.on_task_modes)
            service(srv.GetTaskModes, "config/get_task_modes", self.on_get_task_modes)
            service(srv.QueueEpisode, "config/queue_episode", self.on_queue)
            service(srv.SpawnStatic, "runtime/spawn_static", self.on_spawn("static"))
            service(srv.SpawnDynamic, "runtime/spawn_dynamic", self.on_spawn("dynamic"))
            service(srv.SpawnRobot, "runtime/spawn_robot", self.on_spawn_robot)
            service(srv.MoveEntity, "runtime/move", self.on_move)
            service(srv.DespawnRobot, "runtime/despawn_robot", self.on_despawn)
            service(srv.SubmitTask, "runtime/submit_task", self.on_submit)
            service(srv.SetSemantic, "semantics/set", self.on_set_semantic)
            service(srv.SpawnSound, "runtime/spawn_sound", self.on_spawn_sound)
            service(srv.RemoveSound, "runtime/remove_sound", lambda request, response: srv.RemoveSound.Response(success=request.entity.startswith("runtime_"), error_msg="" if request.entity.startswith("runtime_") else f"unknown sound {request.entity!r}"))
            service(Trigger, "runtime/require_map", lambda request, response: Trigger.Response(success=True))
            self.create_service(MovePedestrians, f"{self.env}/human/move", self.on_move_peds, callback_group=self.group)
            self.create_subscription(Pedestrians, f"{self.env}/human/stream", self.streamed.append, 10, callback_group=self.group)
            self.create_timer(0.1, self.publish_peds, callback_group=self.group)
            self.action = rclpy.action.ActionServer(
                self,
                RunEpisode,
                f"{self.fqn}/lifecycle/run_episode",
                execute_callback=self.on_execute,
                goal_callback=lambda goal: rclpy.action.GoalResponse.REJECT if self.running_goal else rclpy.action.GoalResponse.ACCEPT,
                cancel_callback=lambda handle: rclpy.action.CancelResponse.ACCEPT,
                callback_group=self.group,
            )
            if human == "arena":
                self.serve_humansim(humansim_srv, AgentMeta)

        def record(self, state: int, episode_id: int | None = None, seed: int = 0, world: str = "map_empty") -> EpisodeRecord:
            msg = EpisodeRecord(episode_id=self.episode_id if episode_id is None else episode_id, world=world, seed=seed, tm_robots="random", tm_obstacles="random", tm_modules=["sounds"], robots=["jackal"], outcome_state=state)
            msg.path_length = 3.5
            msg.goal_dist_start = 4.0
            msg.phases = '{"jackal_0": {"phases": [{"goto": [1.0, 2.0, 0.0]}]}}'
            msg.conditions = '[{"op": "never", "p": "collision", "q": null, "text": "no collisions"}]'
            msg.robots_params = [rclpy.Parameter("n", value=3).to_parameter_msg()]
            msg.start_time.sec = 12
            return msg

        def publish_record(self, state: int, episode_id: int, seed: int = 0, world: str = "map_empty", info: str = "") -> None:
            with self.lock:
                self.episode_id = episode_id
            msg = self.record(state, episode_id, seed, world)
            msg.outcome_info = info
            self.pub_episode.publish(msg)

        def burst(self, outcomes: list[int]) -> None:
            for outcome in outcomes:
                current = self.episode_id
                self.publish_record(outcome, current, info="ended")
                self.publish_record(EpisodeRecord.RUNNING, current + 1)

        def on_reset(self, request: object, response: object) -> object:
            self.resets.append((request.world, request.seed))
            current = self.episode_id

            def finish() -> None:
                self.publish_record(EpisodeRecord.SKIPPED, current, info="reset")
                self.pub_resetting.publish(Bool(data=True))
                time.sleep(0.3)
                self.pub_resetting.publish(Bool(data=False))
                self.publish_record(EpisodeRecord.RUNNING, current + 1, seed=request.seed, world=request.world or "map_empty")

            threading.Thread(target=finish, daemon=True).start()
            response.success = True
            return response

        def on_task_modes(self, request: object, response: object) -> object:
            response.robots = sorted(key.value for key in ROBOTS_MODES.keys())
            response.obstacles = sorted(key.value for key in OBSTACLES_MODES.keys())
            response.modules = sorted(key.value for key in MODULE_MODES.keys())
            return response

        def on_get_task_modes(self, request: object, response: object) -> object:
            response.tm_robots = self.conf.TaskMode.TM_ROBOTS.value.value
            response.tm_obstacles = self.conf.TaskMode.TM_OBSTACLES.value.value
            response.tm_modules = sorted(module.value for module in self.conf.TaskMode.TM_MODULES.value)
            return response

        def on_queue(self, request: object, response: object) -> object:
            self.queued.append(request)
            msg = self.record(EpisodeRecord.QUEUED, episode_id=0, world=request.world or "map_empty")
            msg.tm_robots = request.tm_robots or "random"
            msg.tm_obstacles = request.tm_obstacles or "random"
            msg.robots_params = list(request.robots_params)
            msg.obstacles_params = list(request.obstacles_params)
            self.pub_queue.publish(msg)
            response.success = True
            return response

        def on_spawn(self, kind: str) -> object:
            def callback(request: object, response: object) -> object:
                self.spawns.append((kind, request))
                response.success = request.model != "missing"
                response.error_msg = "" if response.success else "model missing is not resolvable"
                response.id = f"env_7/ext_{request.model}"
                return response

            return callback

        def on_spawn_robot(self, request: object, response: object) -> object:
            self.spawns.append(("robot", request))
            response.success = True
            response.name = request.name or "burger_1"
            return response

        def on_move(self, request: object, response: object) -> object:
            self.moves.append(request)
            response.success = request.entity in (ROBOT, "env_7/ext_shelf")
            response.error_msg = "" if response.success else f"unknown entity {request.entity!r}"
            return response

        def on_despawn(self, request: object, response: object) -> object:
            self.despawns.append(request.name)
            response.success = True
            return response

        def on_submit(self, request: object, response: object) -> object:
            import yaml

            self.submitted.append((request.robot, request.task_yaml))
            if request.robot != ROBOT:
                response.success = False
                response.error_msg = f"unknown robot {request.robot!r}"
                return response
            task = TaskRequest.parse(yaml.safe_load(request.task_yaml))
            first = self.phase_count
            self.phase_count += len(task.phases)
            response.phase_indices = list(range(first, self.phase_count))
            response.success = True
            return response

        def on_set_semantic(self, request: object, response: object) -> object:
            self.semantic_writes.append((request.entity, request.field, request.value))
            response.success = request.field in ("state", "locked")
            response.error_msg = "" if response.success else f"field {request.field!r} is not writable"
            return response

        def on_spawn_sound(self, request: object, response: object) -> object:
            self.sounds.append(request)
            response.success = request.kind == "music"
            response.error_msg = "" if response.success else f"unknown environment sound kind {request.kind!r}"
            response.entity = "runtime_music_1"
            return response

        def publish_peds(self) -> None:
            msg = Pedestrians()
            msg.header.frame_id = "map"
            msg.pedestrians.append(Pedestrian(name="adult_1", id=1, pose=_pose(2.0, 3.0), animation_state=Pedestrian.WALKING, agent_type="adult"))
            self.pub_peds.publish(msg)

        def on_move_peds(self, request: object, response: object) -> object:
            self.ped_moves.append(request)
            response.results = [MovePedestrians.Response.SUCCESS if ped.name == "adult_1" else MovePedestrians.Response.NOT_FOUND for ped in request.pedestrians]
            return response

        def on_execute(self, handle: object) -> object:
            self.running_goal = True
            try:
                current = self.episode_id + 1
                world = handle.request.world
                if world == "busy_result":
                    handle.abort()
                    return RunEpisode.Result(state=RunEpisode.Result.SKIPPED, info="busy")
                self.publish_record(EpisodeRecord.RUNNING, current, seed=handle.request.seed)
                deadline = time.monotonic() + (60.0 if world == "forever" else 0.3)
                while time.monotonic() < deadline:
                    if handle.is_cancel_requested:
                        self.cancelled = True
                        self.publish_record(EpisodeRecord.SKIPPED, current, seed=handle.request.seed, info="cancelled")
                        handle.canceled()
                        return RunEpisode.Result(state=RunEpisode.Result.SKIPPED, info="cancelled", episode_id=current)
                    time.sleep(0.02)
                self.publish_record(EpisodeRecord.SUCCESS, current, seed=handle.request.seed, info="finished after 0.3s")
                handle.succeed()
                return RunEpisode.Result(state=RunEpisode.Result.SUCCESS, info="finished after 0.3s", episode_id=current)
            finally:
                self.running_goal = False

        def serve_humansim(self, humansim_srv: object, agent_meta: type) -> None:
            def recorder(name: str, srv_type: type) -> None:
                def callback(request: object, response: object) -> object:
                    self.humansim.append((name, request))
                    if "success" in srv_type.Response.get_fields_and_field_types():
                        response.success = True
                    if "name" in srv_type.Response.get_fields_and_field_types():
                        response.name = f"{name}_1"
                    return response

                self.create_service(srv_type, f"{self.fqn}/{name}", callback, callback_group=self.group)

            recorder("set_waypoints", humansim_srv.SetWaypoints)
            recorder("set_flow", humansim_srv.SetFlow)
            recorder("notify_stimulus", humansim_srv.NotifyStimulus)
            recorder("add_source", humansim_srv.AddSource)
            recorder("remove_source", humansim_srv.RemoveSource)
            recorder("add_sink", humansim_srv.AddSink)
            recorder("remove_sink", humansim_srv.RemoveSink)

            def profile(request: object, response: object) -> object:
                self.humansim.append(("get_profile", request))
                return humansim_srv.GetProfile.Response(phase_names=["social_force"], phase_means_ms=[0.5], phase_p95s_ms=[0.9], n_ticks=40, n_agents=3)

            self.create_service(humansim_srv.GetProfile, f"{self.fqn}/get_profile", profile, callback_group=self.group)
            publisher = self.create_publisher(agent_meta, f"{self.fqn}/agent_meta", _latched())
            publisher.publish(agent_meta(policies=["social_force"], agent_id=[4, 9], name=["anna", ""], handedness=["r", "l"], agent_type=["adult", "child"]))
            self.agent_meta_publisher = publisher

    return TaskGeneratorPeer


def _pose(x: float, y: float) -> object:
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x, pose.position.y = x, y
    pose.orientation.w = 1.0
    return pose


class Peers:
    """The two env peers and the runtime registry node, spun on one executor thread."""

    def __init__(self) -> None:
        import rclpy
        from arena_runtime_msgs.msg import EnvRecord, EnvRegistry
        from rclpy.executors import MultiThreadedExecutor

        peer = _peer_class()
        self.humansim = peer(ENV, "arena")
        self.plain = peer(PLAIN_ENV, "dummy")
        self.runtime = rclpy.create_node("arena_mcp_tg_test_runtime")
        publisher = self.runtime.create_publisher(EnvRegistry, "/arena/state/envs", _latched())
        publisher.publish(EnvRegistry(envs=[EnvRecord(env_id=env_id, fqn=fqn(env_id), ready=True) for env_id in (ENV, PLAIN_ENV)]))
        self.registry_publisher = publisher
        self.executor = MultiThreadedExecutor(num_threads=4)
        for node in (self.humansim, self.plain, self.runtime):
            self.executor.add_node(node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.executor.shutdown()
        for node in (self.humansim, self.plain, self.runtime):
            node.destroy_node()
