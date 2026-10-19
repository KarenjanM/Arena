from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arena_humansim.core.agents import AgentType


class AgentTypeError(ValueError):
    pass


def load(agent_type: str) -> AgentType:
    """A builtin by name or a file by path, through HumanSim's loader; callers resolve a path against the obstacle's ``included_from`` first."""
    from arena_humansim.core.agents import BUILTIN_AGENTS  # noqa: PLC0415
    from arena_humansim.core.agents.loader import load_agent_type_from_file  # noqa: PLC0415

    candidate = Path(agent_type)
    if candidate.suffix == ".yaml" or candidate.is_file():
        if not candidate.is_file():
            raise AgentTypeError(f"agent type file not found: {agent_type}")
        try:
            return load_agent_type_from_file(candidate)
        except Exception as exc:  # noqa: BLE001 - any parse or schema failure is one unusable type
            raise AgentTypeError(f"{agent_type}: {exc}") from None
    if agent_type not in BUILTIN_AGENTS:
        raise AgentTypeError(f"unknown builtin agent type {agent_type!r}; known: {sorted(BUILTIN_AGENTS)}")
    return BUILTIN_AGENTS[agent_type]


def agent_block(agent_type: str, definition: AgentType, speed: float) -> dict[str, Any]:
    """The scenario `agent:` block for a spawned agent: the adapter overrides an undeclared radius with its own default."""
    return {"agent_type": agent_type, "desired_velocity": speed, "radius": definition.agent_radius.mean}
