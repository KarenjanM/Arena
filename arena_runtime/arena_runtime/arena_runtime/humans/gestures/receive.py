"""``receive``: reach with a straight wrist, orienting either the palm or the thumb side upward."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from arena_runtime.humans.pointing import HoldPose, PointAtGenerator
from arena_runtime.humans.pointing import skeleton as S
from arena_runtime.humans.pointing.contract import ROS_JOINT_ORDER, SPINE_SEGMENTS, arm_dofs, clamp, wrap_pi, wrist_dofs
from arena_runtime.humans.support import skeleton

from . import BODY_HEIGHT, GestureClip, resample
from .point import PointGesture

if typing.TYPE_CHECKING:
    from arena_runtime.humans.support import BodyMeasurements


@dataclass(frozen=True)
class _TwoHandHold:
    left: HoldPose
    right: HoldPose


class ReceiveGesture(PointGesture):
    SLOT_SIDE = {"receive_l": "l", "receive_r": "r", "receive_both": "both"}

    def __init__(self, orientation: str = "palm_up", *, body: BodyMeasurements | None = None) -> None:
        if orientation not in ("palm_up", "thumb_up"):
            raise ValueError(f"unknown receive orientation {orientation!r}; expected 'palm_up' or 'thumb_up'")
        self._body = skeleton(body.legs, body.proportions) if body is not None else S.Body(BODY_HEIGHT)
        super().__init__(PointAtGenerator(body=self._body) if body is not None else None)
        self._orientation = orientation

    def bind(self, slot: str, local: np.ndarray, opts: dict) -> dict:
        """Freeze l/r/both from the channel, resolving one hand for the general receive channel."""
        side = self.SLOT_SIDE.get(slot) or self.resolve_hand(local, opts)
        return {**opts, "side": side}

    def joints(self, side: str, moving: bool) -> set[str]:
        if side == "both":
            return self.joints("l", moving) | self.joints("r", moving)
        return super().joints(side, moving) | set(wrist_dofs(side))

    def breathing(self, side: str) -> dict[str, float]:
        if side == "both":
            return {**super().breathing("l"), **super().breathing("r")}
        return super().breathing(side)

    def start(self, local: np.ndarray, opts: dict) -> GestureClip:
        side = opts.get("side", opts.get("hand", "auto"))
        if side == "auto":
            side = self.resolve_hand(local, opts)
        if side not in ("l", "r", "both"):
            raise ValueError("receive side must be l, r or both")
        if side == "both":
            # Both arm solutions use the same upright torso. Independent torso
            # assists cannot be merged without changing the other arm's aim.
            left = self._start_one(local, {**opts, "hand": "l", "moving": True})
            right = self._start_one(local, {**opts, "hand": "r", "moving": True})
            return self._combine(left, right, local)
        return self._start_one(local, {**opts, "hand": side})

    def _start_one(self, local: np.ndarray, opts: dict) -> GestureClip:
        clip = super().start(local, opts)
        twist = self._palm_twist(self._forearm(clip.frames[clip.hold_end], clip.side, opts), clip.side)
        return self._orient_hand(clip, opts, lambda _fore: twist, ramp=True)

    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip:
        if isinstance(hold, _TwoHandHold):
            if opts.get("side", "both") != "both":
                raise ValueError("receive side changed")
            left = self._retarget_one(hold.left, local, {**opts, "hand": "l", "moving": True})
            right = self._retarget_one(hold.right, local, {**opts, "hand": "r", "moving": True})
            return self._combine(left, right, local)
        assert isinstance(hold, HoldPose)
        side = opts.get("side", hold.side)
        if side == "both":
            raise ValueError("two-handed receive requires a two-arm hold")
        return self._retarget_one(hold, local, {**opts, "hand": side})

    def _retarget_one(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip:
        clip = super().retarget(hold, local, opts)
        side = clip.side
        prev = [self._palm_twist(self._forearm(clip.frames[0], side, opts), side)]

        def track(fore: np.ndarray) -> float:
            prev[0] += float(wrap_pi(self._palm_twist(fore, side) - prev[0]))
            return prev[0]

        return self._orient_hand(clip, opts, track, ramp=False)

    def _combine(self, left: GestureClip, right: GestureClip, local: np.ndarray) -> GestureClip:
        """Synchronize both arm timelines into one clip with one shared, upright torso."""
        if left.fps != right.fps:
            raise ValueError("receive arm clips must have the same frame rate")
        clips = (left, right)
        ramp = max(clip.hold_start for clip in clips)
        settled = max(clip.hold_end - clip.hold_start + 1 for clip in clips)
        tracks = [resample(clip.frames[: clip.hold_start], ramp) + resample(clip.frames[clip.hold_start : clip.hold_end + 1], settled) for clip in clips]

        def merge(tracks: Sequence[Sequence[dict]]) -> list[dict]:
            merged = []
            for i, pair in enumerate(zip(*tracks, strict=True)):
                angles = dict.fromkeys(ROS_JOINT_ORDER, 0.0)
                for side, frame in zip(("l", "r"), pair, strict=True):
                    for name in (*arm_dofs(side), *wrist_dofs(side)):
                        angles[name] = frame["angles"][name]
                merged.append({**pair[0], "angles": angles, "t": i / left.fps})
            return merged

        frames = merge(tracks)
        release = None
        if left.release is not None and right.release is not None:
            count = max(len(left.release), len(right.release))
            release = merge([resample(left.release, count), resample(right.release, count)])
        elif left.release is not None or right.release is not None:
            raise ValueError("receive arm clips must both supply a release arc")

        pos, fore = S.fk(frames[-1]["angles"], self._body)
        clearance = min(value for side in ("l", "r") for value in S.collision_clearances(pos, self._body, side).values())
        errors = []
        for side in ("l", "r"):
            want = S.unit(np.asarray(local) - pos[f"{side}_wrist"])
            errors.append(math.degrees(math.acos(float(np.clip((fore[f"{side}_forearm"] @ S.DOWN) @ want, -1.0, 1.0)))))
        report = {
            "bimanual": True,
            "upright": True,
            "ok": all(clip.report.get("ok", True) for clip in clips) and clearance >= 0.0,
            "near_target_fallback": any(clip.report.get("near_target_fallback", False) for clip in clips),
            "relaxed": any(clip.report.get("relaxed", False) for clip in clips),
            "clearance_m": clearance,
            "aim_error_deg": max(errors),
            "arms": {"l": left.report, "r": right.report},
        }
        assert isinstance(left.hold, HoldPose) and isinstance(right.hold, HoldPose)
        return GestureClip(frames=frames, fps=left.fps, hold_start=ramp, hold_end=ramp + settled - 1, side="both", hold=_TwoHandHold(left.hold, right.hold), report=report, release=release)

    def _forearm(self, frame: dict, side: str, opts: dict) -> np.ndarray:
        """Ped-local forearm frame, with an upright torso when the torso is not blended."""
        pose = {**dict.fromkeys(ROS_JOINT_ORDER, 0.0), **frame["angles"]}
        if opts.get("moving", False):
            for seg in SPINE_SEGMENTS:
                pose[f"r_{seg}"] = pose[f"y_{seg}"] = pose[seg] = 0.0
        _, frames = S.fk(pose, self._body)
        return frames[f"{side}_forearm"]

    def _palm_twist(self, fore: np.ndarray, side: str) -> float:
        """Wrist twist that turns the selected hand direction as upward as the forearm allows."""
        axis = fore @ S.WRIST_AXES[side][0]
        local = S.FWD if self._orientation == "thumb_up" else np.array([0.0, -S.REFLECT[side], 0.0])
        direction = fore @ local
        up = S.perp_component(fore @ S.DOWN, S.UP)
        return math.atan2(float(axis @ np.cross(direction, up)), float(direction @ up))

    def _orient_hand(self, clip: GestureClip, opts: dict, twist: Callable[[np.ndarray], float], *, ramp: bool) -> GestureClip:
        """Orient the hand with a straight wrist, ramping the twist on a fresh reach and unwinding on release."""
        side = clip.side
        tw, ext = wrist_dofs(side)
        park = self._forearm(clip.frames[clip.hold_end], side, opts)
        park_twist = self._palm_twist(park, side)

        def palm(frames: Sequence[dict], twist: Callable[[np.ndarray], float], rest: dict | None) -> list[dict]:
            """Blend wrist twist with the forearm's rise from ``rest`` to the park pose, applying it fully without a rest frame."""
            lo, hi = (self._rise(self._forearm(rest, side, opts)), self._rise(park)) if rest is not None else (0.0, 0.0)
            out = []
            for frame in frames:
                fore = self._forearm(frame, side, opts)
                w = min(1.0, max(0.0, (self._rise(fore) - lo) / (hi - lo))) if hi - lo > 1e-6 else 1.0
                angles = dict(frame["angles"])
                angles[tw] = clamp(tw, w * twist(fore))
                angles[ext] = 0.0
                out.append({**frame, "angles": angles})
            return out

        frames = palm(clip.frames, twist, clip.frames[0] if ramp else None)
        release = None if clip.release is None else palm(clip.release, lambda _fore: park_twist, clip.release[-1])
        return GestureClip(frames=frames, fps=clip.fps, hold_start=clip.hold_start, hold_end=clip.hold_end, side=side, hold=clip.hold, report=clip.report, release=release, joints=clip.joints, loop=clip.loop)

    @staticmethod
    def _rise(fore: np.ndarray) -> float:
        """Angle of the forearm off hanging straight down."""
        return math.acos(float(np.clip(S.DOWN @ (fore @ S.DOWN), -1.0, 1.0)))
