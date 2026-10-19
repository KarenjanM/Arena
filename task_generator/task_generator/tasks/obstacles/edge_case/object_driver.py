from __future__ import annotations

from collections.abc import Callable, Coroutine, Sequence
from typing import TYPE_CHECKING, Any

import attrs

from .objects import DESPAWN, SPAWN, ResolvedEvent

if TYPE_CHECKING:
    from rclpy.impl.rcutils_logger import RcutilsLogger

Work = Coroutine[None, None, object]

DEFAULT_RATE_HZ = 5.0


@attrs.define
class EventOutcome:
    entity: str
    action: str
    t_fired: float | None = None
    t_designed: float = 0.0
    pose: tuple[float, float] | None = None
    reveal_distance_m: float | None = None
    entity_id: str = ""
    status: str = "pending"
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)


class ObjectDriver:
    """Fires a resolved object timeline against sim time; `tick` is called by the mode's sim-time loop."""

    def __init__(
        self,
        *,
        spawn: Callable[[ResolvedEvent], Work],
        despawn: Callable[[ResolvedEvent, str], Work],
        submit: Callable[[Work], None],
        logger: RcutilsLogger,
    ) -> None:
        self._spawn = spawn
        self._despawn = despawn
        self._submit = submit

        self._events: list[ResolvedEvent] = []
        self._outcomes: list[EventOutcome] = []
        self._ids: dict[str, str] = {}
        self._next = 0
        self._t0: float | None = None
        self._logger = logger

    def arm(self, events: Sequence[ResolvedEvent]) -> None:
        """The clock starts on the next tick, not here: the simulator may still be paused at the end of a reset."""
        self._events = list(events)
        self._outcomes = [EventOutcome(entity=e.entity, action=e.action, t_designed=e.t) for e in self._events]
        self._ids = {}
        self._next = 0
        self._t0 = None

    def outcomes(self) -> list[EventOutcome]:
        return list(self._outcomes)

    def pending(self) -> int:
        return sum(1 for o in self._outcomes if o.status == "pending")

    def note_result(self, index: int, *, entity_id: str = "", ok: bool = True, reason: str = "") -> None:
        if not 0 <= index < len(self._outcomes):
            return
        outcome = self._outcomes[index]
        outcome.status = "ok" if ok else "failed"
        outcome.reason = reason
        if entity_id:
            outcome.entity_id = entity_id
            self._ids[outcome.entity] = entity_id

    def tick(self, now: float) -> None:
        if self._next >= len(self._events):
            return
        if self._t0 is None:
            if now <= 0.0:
                return
            self._t0 = now
            return

        elapsed = now - self._t0
        # a loop: a pause that outlasts several events must fire all of them
        while self._next < len(self._events) and self._events[self._next].t <= elapsed:
            index = self._next
            self._next += 1
            self._fire(index, self._events[index], elapsed)

    def _fire(self, index: int, event: ResolvedEvent, elapsed: float) -> None:
        outcome = self._outcomes[index]
        outcome.t_fired = elapsed
        outcome.pose = event.pose
        outcome.reveal_distance_m = event.reveal_distance_m

        coroutine: Work
        if event.action == SPAWN:
            coroutine = self._spawn(event)
        elif event.action == DESPAWN:
            entity_id = self._ids.get(event.entity, "")
            if not entity_id:
                outcome.status = "skipped"
                outcome.reason = "nothing was spawned under this entity"
                return
            coroutine = self._despawn(event, entity_id)
        else:  # pragma: no cover - parse_event rejects anything else
            outcome.status = "failed"
            outcome.reason = f"unknown action {event.action!r}"
            return

        self._submit(self._run(index, coroutine))

    async def _run(self, index: int, coroutine: Work) -> None:
        try:
            result = await coroutine
        except Exception as exc:
            self.note_result(index, ok=False, reason=repr(exc))
            self._logger.warning(f"edge_case objects: event {index} failed ({exc!r})")
            return
        if isinstance(result, str):
            self.note_result(index, entity_id=result, ok=bool(result), reason="" if result else "spawn returned no id")
        else:
            self.note_result(index, ok=bool(result), reason="" if result else "the entity was not there")
