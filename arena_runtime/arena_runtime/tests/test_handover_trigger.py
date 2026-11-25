from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from arena_runtime.humans.animation_manager import AnimationManager
from arena_runtime.humans.clips import NoClips
from arena_runtime.humans.gestures import Channel, GestureLayer, GestureRequest
from arena_runtime.humans.handover import PHASE_TIMEOUT_S, HandoverTrigger
from arena_runtime.humans.pointing import skeleton as S
from arena_runtime.humans.support import PALM_CENTER_FRACTION, TYPICAL_BODY, skeleton

POSE = (4.0, -2.0, math.pi / 2)
TARGET = (4.0, -1.5, 1.25)


def test_external_phases_choose_their_overlay_and_keep_the_reach_on_take() -> None:
    trigger = HandoverTrigger()
    head = Channel("head", (6.0, -2.0, 1.5))
    arm = Channel("halt_r", TARGET)
    background = (head, arm)

    def apply(phase: str, now: float) -> tuple[Channel, ...]:
        trigger.command("alice", phase, TARGET, "bottle", now)
        active = trigger.active("alice", now)
        assert active is not None
        return active.apply(background, POSE, TYPICAL_BODY, "l")

    assert apply("attend", 0.0) == (arm, Channel("head", TARGET))
    kept, ready = apply("ready", 1.0)
    assert kept == Channel("head", TARGET)
    assert ready.slot == "receive_l" and ready.orientation == "thumb_up"
    assert ready.at[0] < POSE[0] and POSE[1] < ready.at[1] < TARGET[1]
    assert ready.at[2] < TARGET[2]
    assert ready.grip == 0.0
    reach_head, reach = apply("reach", 2.0)
    take_head, take = apply("take", 3.0)
    assert reach_head == take_head == kept
    assert reach.at == take.at == TARGET
    assert ready.slot == reach.slot == take.slot
    assert ready.opts == reach.opts == take.opts
    assert reach.grip == 0.0 and take.grip == 0.6
    assert apply("retract", 4.0) == (head,)


@pytest.mark.parametrize("phase", ["attend", "ready", "reach", "take"])
def test_timeout_is_per_phase_and_duplicate_commands_do_not_extend_it(phase: str) -> None:
    trigger = HandoverTrigger()
    deadline = PHASE_TIMEOUT_S[phase]
    updated = (4.1, -1.5, 1.25)
    trigger.command("alice", phase, TARGET, "box", 0.0)
    trigger.command("bob", "ready", TARGET, "bottle", deadline / 2)
    trigger.command("alice", phase, updated, "box", deadline / 2)
    active = trigger.active("alice", deadline - 0.01)
    assert active is not None and active.at == updated
    if phase in ("reach", "take"):
        head, receive = active.apply((), POSE, TYPICAL_BODY, "r")
        assert head == Channel("head", updated)
        assert receive.slot == "receive_both" and receive.at == updated

    # A take received after the deadline, even before the next publish, cannot revive the old attempt.
    trigger.command("alice", "take", TARGET, "box", deadline)
    retract = trigger.active("alice", deadline)
    assert retract is not None and retract.name == "retract"
    assert retract.apply((Channel("receive_both", TARGET),), POSE, TYPICAL_BODY, "r") == ()
    other = trigger.active("bob", deadline)
    assert other is not None and other.name == "ready"
    assert trigger.active("alice", deadline + PHASE_TIMEOUT_S["retract"]) is None

    trigger.command("alice", "ready", TARGET, "box", deadline + PHASE_TIMEOUT_S["retract"])
    retry = trigger.active("alice", deadline + PHASE_TIMEOUT_S["retract"])
    assert retry is not None and retry.name == "ready"
    _, receive = retry.apply((), POSE, TYPICAL_BODY, "l")
    assert receive.slot == "receive_both" and receive.orientation == "palm_up" and receive.grip == 0.0


def test_invalid_commands_preserve_the_last_phase_and_departed_peds_are_forgotten() -> None:
    trigger = HandoverTrigger()
    trigger.command("alice", "reach", TARGET, "bottle", 0.0)
    with pytest.raises(ValueError, match="unknown handover object"):
        trigger.command("alice", "take", TARGET, "missing", 1.0)
    with pytest.raises(ValueError, match="must be finite"):
        trigger.command("alice", "take", (float("nan"), 0.0, 0.0), "bottle", 1.0)
    active = trigger.active("alice", 1.0)
    assert active is not None and active.name == "reach"
    trigger.retain(["bob"])
    assert trigger.active("alice", 1.0) is None


@pytest.mark.parametrize(("kind", "side", "expected_grip"), [("bottle", "l", 0.6), ("box", "both", 0.2)])
def test_ready_retargets_then_take_grips_the_same_clip(kind: str, side: str, expected_grip: float) -> None:
    log = logging.getLogger(__name__)
    manager = AnimationManager(logger=log, fps=20.0, clips=NoClips())
    layer = GestureLayer(manager, log)
    manager.gesture_hook = layer
    trigger = HandoverTrigger()
    pose = (0.0, 0.0, 0.0)

    def tick(start: int, stop: int) -> None:
        for frame in range(start, stop):
            active = trigger.active("alice", frame / 20)
            channels = active.apply((), pose, TYPICAL_BODY, "l") if active else ()
            manager.compute(1, 0, 0.0, 0.05, gesture=GestureRequest(channels, pose, False, TYPICAL_BODY))

    target = (0.48, 0.18 if side == "l" else 0.0, 1.17)
    trigger.command("alice", "ready", target, kind, 0.0)
    tick(0, 25)
    slot = layer._agents[1].slots["arm"]
    assert slot.phase == "hold" and layer.sides(1)["arm"] == side
    ready_clip = slot.clip
    assert ready_clip is not None
    body = skeleton(TYPICAL_BODY.legs, TYPICAL_BODY.proportions)
    palm_length = PALM_CENTER_FRACTION * TYPICAL_BODY.proportions.hand

    trigger.command("alice", "reach", target, kind, 1.25)
    tick(25, 50)
    assert slot.clip is not ready_clip
    assert slot.local == pytest.approx(target)
    reached_clip = slot.clip
    assert reached_clip is not None
    assert reached_clip.frames[0]["angles"] == ready_clip.frames[-1]["angles"]
    for arm in reached_clip.sides:
        path = []
        for frame in reached_clip.frames:
            pos, axes = S.fk(frame["angles"], body)
            path.append(pos[f"{arm}_wrist"] + palm_length * (axes[f"{arm}_forearm"] @ S.DOWN))
        samples = np.stack(path)
        assert samples[-1, 0] > samples[0, 0] + 0.10
        assert samples[-1, 2] > samples[0, 2] + 0.06
        assert np.all(np.diff(samples[:, (0, 2)], axis=0) >= -1e-8)
        lateral, z = slot.bound["handover"]
        assert samples[-1] == pytest.approx((target[0], target[1] + S.REFLECT[arm] * lateral, target[2] + z), abs=0.005)

    trigger.command("alice", "take", target, kind, 2.5)
    tick(50, 70)
    assert layer._agents[1].slots["arm"] is slot and slot.clip is reached_clip
    grip_side, weight = layer.grip_weights(1)["arm"]
    assert grip_side == side and weight == pytest.approx(expected_grip)

    trigger.command("alice", "retract", target, kind, 3.5)
    tick(70, 71)
    assert {slot: phase[2] for slot, phase in layer.phases(1).items()} == {"head": "release", "arm": "release"}
    tick(71, 100)
    assert layer.phases(1) == {}


def test_early_retract_lowers_the_arm_and_releases_gaze_together() -> None:
    log = logging.getLogger(__name__)
    manager = AnimationManager(logger=log, fps=20.0, clips=NoClips())
    layer = GestureLayer(manager, log)
    manager.gesture_hook = layer
    trigger = HandoverTrigger()
    pose = (0.0, 0.0, 0.0)

    def tick(t: float) -> None:
        active = trigger.active("alice", t)
        channels = active.apply((), pose, TYPICAL_BODY, "r") if active else ()
        manager.compute(1, 0, 0.0, 0.05, gesture=GestureRequest(channels, pose, False, TYPICAL_BODY))

    target = (0.48, -0.18, 1.17)
    trigger.command("alice", "ready", target, "bottle", 0.0)
    for frame in range(4):
        tick(frame / 20)
    assert layer.phases(1)["arm"][2] == "ramp"
    before = {slot: weight for slot, _, _, weight in manager.overlays(1)}
    previous_arm = manager.overlay_pose(1, "arm")
    trigger.command("alice", "retract", target, "bottle", 0.20)
    tick(0.20)
    assert {slot: phase[2] for slot, phase in layer.phases(1).items()} == {"head": "release", "arm": "release"}
    after = {slot: weight for slot, _, _, weight in manager.overlays(1)}
    assert after["head"] < before["head"]
    assert after["arm"] <= before["arm"]
    arm = manager.overlay_pose(1, "arm")
    assert arm is not None and previous_arm is not None
    assert 0 <= arm["r_elbow"] < previous_arm["r_elbow"]
    for frame in range(5, 40):
        tick(frame / 20)
    assert layer.phases(1) == {}
