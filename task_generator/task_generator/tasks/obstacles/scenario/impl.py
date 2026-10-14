import asyncio

from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier
from arena_simulation_setup.tree.World import WorldIdentifier, anchor_light
from arena_simulation_setup.tree.World.Scenario import Scenario, ScenarioView

from task_generator.shared import Region
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles
from task_generator.tasks.obstacles._validation import make_is_valid
from task_generator.tasks.registry import default_scenario


class TM_Scenario(TM_Obstacles):
    _config: ROSParamT[str]

    async def reset(self, *, seed: int) -> Obstacles:
        scenario_name = self._config.value
        await self._ctx.environment_manager.configure_contact(self._contact_mode.value, self._standing_distance.value)
        await self._ctx.environment_manager.configure_gestures(self._gesture_mode.value)
        world_description = self._ctx.world_manager.world_compacted()

        is_valid = make_is_valid(self._ctx.world_manager.map, self.node.conf.Obstacles.SAFE_DIST.value)

        zone_conv = world_description.zone_converter(
            self.node.conf.General.RNG.stream("obstacles", "scenario"),
            is_valid=is_valid,
        )

        scenario_view = WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().scenario(scenario_name).resolve_sync()
        scenario = scenario_view.load(converter=zone_conv)
        self.scenario = scenario
        self.lights = [anchor_light(light, scenario.objects) for light in scenario.lights]
        await self._check_clips(scenario_view, scenario)

        regions = [
            Region(
                name=name,
                type=r.type,
                polygon=list(r.polygon),
                config=r.config,
                included_from=scenario_view.path,
            )
            for name, r in scenario.regions.items()
        ]
        # Recorded, not pushed: `Task._reset_episode` applies these once, after every mode
        # that wraps this one has had a chance to see them. See TM_Obstacles.pending_regions.
        self.pending_regions[:] = regions

        self.node.register_timeline(scenario.timeline, seed)
        self.node.register_conditions(scenario.conditions)

        return scenario.objects, scenario.humans

    async def _check_clips(self, scenario_view: ScenarioView, scenario: Scenario) -> None:
        """Resolve the clips the scenario's agents play and warn about the ones that will not."""
        try:
            names = sorted({name for obstacle in scenario.humans for name in scenario_view.agent_clips(obstacle.agent)})
        except OSError as e:
            self._logger.warning(f"scenario {scenario_view.path.name}: could not read an agent file to check its clips: {e}")
            return
        found = await asyncio.gather(*(AnimationIdentifier.parse(name).resolve_path() for name in names), return_exceptions=True)
        missing = [name for name, path in zip(names, found, strict=True) if isinstance(path, Exception)]
        if missing:
            self._logger.warning(f"scenario {scenario_view.path.name}: animation clips not found, agents will skip them: {missing} (arena asset find animation <name>)")

    def __init__(self, **kwargs: object) -> None:
        TM_Obstacles.__init__(self, **kwargs)
        self.scenario: Scenario | None = None

        self._config = self.node.ROSParam[str](
            self.namespace("file"),
            default_scenario(self._ctx.world_manager.loaded_world),
        )
        self._contact_mode = self.node.ROSParam[str](self.namespace("contact_mode"), "enabled")
        self._standing_distance = self.node.ROSParam[float](self.namespace("standing_distance"), 1.2)
        self._gesture_mode = self.node.ROSParam[str](self.namespace("gesture_mode"), "enabled")
