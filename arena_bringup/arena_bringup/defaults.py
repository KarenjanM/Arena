"""Per-sim launch defaults shared by the supervisor and task_generator.launch.py."""

DEFAULT_HUMAN: dict[str, str] = {
    "dummy": "dummy",
    "flatsim": "arena",
    "gazebo": "arena",
    "isaac": "arena",
    "mujoco": "arena",
}


GZ_DISCOVERY_BASE_PORT = 5500


def gz_discovery_env(domain_id: int) -> dict[str, str]:
    """gz-transport discovery ports of one ROS domain: a private pair per domain, below the DDS port blocks."""
    msg_port = GZ_DISCOVERY_BASE_PORT + 2 * domain_id
    return {"GZ_DISCOVERY_MSG_PORT": str(msg_port), "GZ_DISCOVERY_SRV_PORT": str(msg_port + 1)}


def default_human(sim: str) -> str:
    """Human-backend default for a sim, unknown sims fall back to dummy."""
    return DEFAULT_HUMAN.get(sim, "dummy")
