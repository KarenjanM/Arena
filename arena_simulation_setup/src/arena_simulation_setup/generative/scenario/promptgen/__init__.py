"""Prompt-driven edge cases: steps A, D and B+C decide, then the design is realized into a scenario timeline."""

from __future__ import annotations

from .pipeline import Result, RobotPlan, generate, write_result
from .profiles import PROFILES, TIERS, Profile, derive, profile, tier_value
from .spec import Record, SchemaError

__all__ = ["PROFILES", "TIERS", "Profile", "Record", "Result", "RobotPlan", "SchemaError", "derive", "generate", "profile", "tier_value", "write_result"]
