from __future__ import annotations

from types import SimpleNamespace

import attrs
import numpy as np
import pytest

from arena_runtime.humans.gestures import FADE_S, Channel, GestureClip, GestureLayer, GestureRequest
from arena_runtime.humans.hands import DIGITS, GRIP, receive_hand


class Logger:
    def info(self, message):
        pass

    def warning(self, message):
        raise AssertionError(message)


class Manager:
    def __init__(self):
        self.animations = {}
        self.installs = 0

    def register_transient(self, name, frames, **kwargs):
        self.animations[name] = SimpleNamespace(frames=frames)
        return name

    def set_ped_blend(self, *args, **kwargs):
        self.installs += 1


class Receive:
    """Fixed arm clip so these state-machine tests need neither assets nor IK."""

    def __init__(self):
        self.starts = 0
        self.retargets = 0

    def bind(self, slot, local, opts):
        return {**opts, "hand": "l" if slot.endswith("_l") else "r"}

    def joints(self, side, moving):
        return {f"{side}_elbow"}

    def breathing(self, side):
        return {}

    def clip(self, opts):
        side = opts["hand"]
        frames = [{"angles": {f"{side}_elbow": 0.4}}]
        return GestureClip(frames=frames, fps=20.0, hold_start=0, hold_end=0, side=side, hold=None, report={}, release=frames * 2)

    def start(self, local, opts):
        self.starts += 1
        return self.clip(opts)

    def retarget(self, hold, local, opts):
        self.retargets += 1
        return self.clip(opts)


@pytest.fixture
def rig():
    manager, receive = Manager(), Receive()
    layer = GestureLayer(manager, Logger())
    layer._gestures["receive"] = receive
    return manager, layer, receive


def request(channel):
    return GestureRequest(channels=(channel,), pose=(0.0, 0.0, 0.0), moving=False)


def test_grip_is_separate_from_generator_options():
    channel = Channel("receive", (2.0, 0.0, 1.2), hand="l")
    assert channel.grip == 0.0
    assert attrs.evolve(channel, grip=0.4).opts == channel.opts
    assert "grip" not in channel.opts


@pytest.mark.parametrize("grip", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_grip_is_rejected(grip):
    with pytest.raises(ValueError, match="grip must be between"):
        Channel("receive_r", (2.0, 0.0, 1.2), grip=grip)


@pytest.mark.parametrize("grip,expected", [(False, 0.0), (True, 1.0), (0, 0.0), (1, 1.0)])
def test_grip_converts_existing_boolean_and_integer_inputs(grip, expected):
    channel = Channel("receive_r", (2.0, 0.0, 1.2), grip=grip)
    assert channel.grip == expected
    assert isinstance(channel.grip, float)


@pytest.mark.parametrize("side", ["l", "r"])
def test_grip_changes_preserve_the_running_slot_and_clip(rig, side):
    manager, layer, receive = rig
    channel = Channel(f"receive_{side}", (2.0, 0.0, 1.2))
    layer(1, request(channel), 0.05)
    slot = layer._agents[1].slots["arm"]
    clip = slot.clip
    for grip in (0.0, 0.4, 0.4, 0.8, 1.0, 0.0):
        old_time = slot.t
        layer(1, request(attrs.evolve(channel, grip=grip)), 0.05)
        assert layer._agents[1].slots["arm"] is slot
        assert slot.clip is clip
        assert slot.phase == "hold"
        assert slot.t == pytest.approx(old_time + 0.05)
        assert layer.grips(1) == {"arm": (side, grip)}
    assert receive.starts == manager.installs == 1
    assert receive.retargets == 0


@pytest.mark.parametrize("side", ["l", "r"])
def test_grip_can_be_set_on_the_first_request(rig, side):
    _, layer, _ = rig
    layer(1, request(Channel(f"receive_{side}", (2.0, 0.0, 1.2), grip=0.4)), 0.05)
    assert layer.grips(1) == {"arm": (side, 0.4)}
    assert layer.grips(99) == {}
    layer.forget(1)
    assert layer.grips(1) == {}


def test_target_and_grip_can_change_together_without_restarting(rig):
    manager, layer, receive = rig
    channel = Channel("receive_r", (2.0, 0.0, 1.2))
    layer(1, request(channel), 0.05)
    slot = layer._agents[1].slots["arm"]
    updated = attrs.evolve(channel, at=(2.0, 0.5, 1.2), grip=0.4)
    layer(1, request(updated), 0.05)
    assert layer._agents[1].slots["arm"] is slot
    assert layer.grips(1) == {"arm": ("r", 0.4)}
    assert np.allclose(slot.local, updated.at)
    assert receive.starts == 1
    assert receive.retargets == 1
    assert manager.installs == 2


def test_other_gestures_do_not_force_a_receive_hand_shape(rig):
    _, layer, receive = rig
    layer._gestures["point"] = receive
    layer(1, request(Channel("arm_r", (2.0, 0.0, 1.2), grip=1.0)), 0.05)
    assert layer.grips(1) == {}


@pytest.mark.parametrize("side", ["l", "r", "both"])
@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_receive_hand_interpolates_each_digit(side, weight):
    pose = receive_hand(side, weight)
    sides = ("l", "r") if side == "both" else (side,)
    for hand in sides:
        assert tuple(pose[f"{hand}_curl_{digit}"] for digit in DIGITS) == pytest.approx(tuple(weight * value for value in GRIP))
        assert pose[f"{hand}_spread"] == 0.0
    assert len(pose) == 6 * len(sides)


@pytest.mark.parametrize("grip", [0.4, 1.0])
@pytest.mark.parametrize("side", ["l", "r"])
def test_grip_fades_over_fade_s_without_restarting(rig, side, grip):
    manager, layer, receive = rig
    channel = Channel(f"receive_{side}", (2.0, 0.0, 1.2))
    layer(1, request(channel), 0.05)
    closed = attrs.evolve(channel, grip=grip)
    layer(1, request(closed), 0.0)
    assert layer.grip_weights(1) == {"arm": (side, 0.0)}
    # Repeating the request must advance, rather than reset, the same fade.
    for fraction in (0.25, 0.5, 0.75, 1.0):
        layer(1, request(closed), FADE_S / 4)
        assert layer.grip_weights(1)["arm"][1] == pytest.approx(fraction * grip)
    layer(1, request(closed), FADE_S)
    assert layer.grip_weights(1)["arm"][1] == grip
    assert receive.starts == manager.installs == 1
    assert receive.retargets == 0


def test_reversing_a_grip_fade_starts_from_the_current_weight(rig):
    manager, layer, _ = rig
    opened = Channel("receive_r", (2.0, 0.0, 1.2))
    closed = attrs.evolve(opened, grip=1.0)
    layer(1, request(opened), 0.05)
    layer(1, request(closed), 0.0)
    layer(1, request(closed), FADE_S / 2)
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(0.5)
    layer(1, request(opened), 0.0)
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(0.5)
    layer(1, request(opened), FADE_S / 2)
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(0.25)
    layer(1, request(opened), FADE_S / 2)
    assert layer.grip_weights(1)["arm"][1] == 0.0
    assert manager.installs == 1


def test_changing_partial_grip_mid_fade_keeps_the_current_pose(rig):
    manager, layer, receive = rig
    channel = Channel("receive_r", (2.0, 0.0, 1.2), grip=0.8)
    layer(1, request(channel), 0.0)
    layer(1, request(channel), FADE_S / 2)
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(0.4)
    updated = attrs.evolve(channel, grip=0.3)
    layer(1, request(updated), 0.0)
    assert layer.grips(1)["arm"][1] == 0.3
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(0.4)
    layer(1, request(updated), FADE_S / 2)
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(0.35)
    layer(1, request(updated), FADE_S / 2)
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(0.3)
    assert receive.starts == manager.installs == 1
    assert receive.retargets == 0


@pytest.mark.parametrize("grip", [0.4, 1.0])
@pytest.mark.parametrize("side", ["l", "r"])
def test_initial_grip_fades_from_open_and_forget_removes_the_weight(rig, side, grip):
    _, layer, _ = rig
    channel = Channel(f"receive_{side}", (2.0, 0.0, 1.2), grip=grip)
    layer(1, request(channel), 0.05)
    assert layer.grip_weights(1) == {"arm": (side, 0.0)}
    layer(1, request(channel), FADE_S / 2)
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(grip / 2)
    layer(1, request(channel), FADE_S / 2)
    assert layer.grip_weights(1) == {"arm": (side, grip)}
    assert layer.grip_weights(99) == {}
    layer.forget(1)
    assert layer.grip_weights(1) == {}


def test_retarget_does_not_reset_the_grip_fade_clock(rig):
    _, layer, receive = rig
    channel = Channel("receive_r", (2.0, 0.0, 1.2), grip=0.4)
    layer(1, request(channel), 0.05)
    layer(1, request(channel), FADE_S / 2)
    updated = attrs.evolve(channel, at=(2.0, 0.5, 1.2))
    layer(1, request(updated), 0.0)
    assert receive.retargets == 1
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(0.2)
    layer(1, request(updated), FADE_S / 2)
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(0.4)


@pytest.mark.parametrize("weight,expected", [(-1.0, 0.0), (2.0, 1.0)])
def test_receive_hand_clamps_the_weight(weight, expected):
    pose = receive_hand("r", weight)
    assert tuple(pose[f"r_curl_{digit}"] for digit in DIGITS) == tuple(expected * value for value in GRIP)
