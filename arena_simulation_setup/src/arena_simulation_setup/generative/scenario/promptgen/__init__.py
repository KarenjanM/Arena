"""Prompt-driven edge-case generation: Type A by lexicon or model, then Type D, then Type B+C by model.
Static changes go into the population; what the case changes at its onset is a behavior-tree switch the scenario timeline raises."""

from __future__ import annotations

from .pipeline import Result, RobotPlan, generate, write_result
from .profiles import PROFILES, TIERS, Profile, derive, profile, tier_value
from .spec import Record, SchemaError

__all__ = ["PROFILES", "TIERS", "Profile", "Record", "Result", "RobotPlan", "SchemaError", "derive", "generate", "profile", "tier_value", "write_result"]
