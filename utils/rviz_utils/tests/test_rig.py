from __future__ import annotations

import math

import numpy as np
import pytest

from rviz_utils.hri.rig import _rot_axis, semantic_to_rig

_HALF_PI = math.pi / 2


def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Raxis(axis: np.ndarray, a: float) -> np.ndarray:
    """Rodrigues' rotation formula, spec-independent of rig.py's own implementation."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(a), math.sin(a)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _semantic_rotation(y: float, p: float, r: float) -> np.ndarray:
    """JOINTS.md Section 1a: R = Rz(y) . Raxis((0,-1,0), p) . Raxis((0,0,-1), r),
    the twist intrinsic about the rest limb axis (equivalently the world-frame
    rotation about the current limb direction applied after yaw and flexion)."""
    partial = _Rz(y) @ _Raxis((0.0, -1.0, 0.0), p)
    return partial @ _Raxis((0.0, 0.0, -1.0), r)


def _left_chain(a: float, b: float, c: float) -> np.ndarray:
    return _Rz(-a) @ _Rx(b) @ _Rz(c)


def _right_chain(a: float, b: float, c: float) -> np.ndarray:
    return _Rz(a) @ _Rx(-b) @ _Rz(-c)


_SHOULDER_NAMES = (
    "l_y_shoulder",
    "l_p_shoulder",
    "l_r_shoulder",
    "r_y_shoulder",
    "r_p_shoulder",
    "r_r_shoulder",
)


def _translate_shoulders(y: float, p: float, r: float) -> dict[str, float]:
    """Feed the same wire triple through both sides."""
    positions = [y, p, r, y, p, r]
    out = semantic_to_rig(_SHOULDER_NAMES, positions)
    return dict(zip(_SHOULDER_NAMES, out, strict=True))


_P_SWEEP = np.linspace(-3.0, 3.0, 25)


@pytest.mark.parametrize("p", _P_SWEEP)
def test_sagittal_only_constants_left(p: float) -> None:
    """wire (0, p, 0) extracts to the constants (pi/2, p, pi/2)."""
    s = _translate_shoulders(0.0, p, 0.0)
    assert math.isclose(s["l_y_shoulder"], _HALF_PI, abs_tol=1e-9)
    assert math.isclose(s["l_p_shoulder"], p, abs_tol=1e-9)
    assert math.isclose(s["l_r_shoulder"], _HALF_PI, abs_tol=1e-9)


@pytest.mark.parametrize("p", _P_SWEEP)
def test_sagittal_only_constants_right(p: float) -> None:
    s = _translate_shoulders(0.0, p, 0.0)
    assert math.isclose(s["r_y_shoulder"], _HALF_PI, abs_tol=1e-9)
    assert math.isclose(s["r_p_shoulder"], p, abs_tol=1e-9)
    assert math.isclose(s["r_r_shoulder"], _HALF_PI, abs_tol=1e-9)


def _random_wire_triples(n: int, seed: int) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    ys = rng.uniform(-3.1, 3.1, size=n)
    ps = rng.uniform(-1.0, 3.3, size=n)
    rs = rng.uniform(-1.6, 1.6, size=n)
    return list(zip(ys, ps, rs, strict=True))


@pytest.mark.parametrize("triple", _random_wire_triples(200, seed=0))
def test_round_trip_left(triple: tuple[float, float, float]) -> None:
    y, p, r = triple
    s = _translate_shoulders(y, p, r)
    target = _semantic_rotation(y, p, r)
    got = _left_chain(s["l_y_shoulder"], s["l_p_shoulder"], s["l_r_shoulder"])
    assert np.allclose(got, target, atol=1e-9)


@pytest.mark.parametrize("triple", _random_wire_triples(200, seed=0))
def test_round_trip_right(triple: tuple[float, float, float]) -> None:
    y, p, r = triple
    s = _translate_shoulders(y, p, r)
    target = _semantic_rotation(y, p, r)
    got = _right_chain(s["r_y_shoulder"], s["r_p_shoulder"], s["r_r_shoulder"])
    assert np.allclose(got, target, atol=1e-9)


def _random_yr_pairs(n: int, seed: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    ys = rng.uniform(-3.1, 3.1, size=n)
    rs = rng.uniform(-1.6, 1.6, size=n)
    return list(zip(ys, rs, strict=True))


@pytest.mark.parametrize("pair", _random_yr_pairs(50, seed=1))
def test_degenerate_p_zero_folds_correctly(pair: tuple[float, float]) -> None:
    """p=0 is the degenerate branch (sin(b)~0). a/c individually are ambiguous, only
    their composition through the URDF chain has to reproduce the semantic rotation."""
    y, r = pair
    s = _translate_shoulders(y, 0.0, r)
    target = _semantic_rotation(y, 0.0, r)

    assert math.isclose(s["l_y_shoulder"], _HALF_PI, abs_tol=1e-9)
    assert math.isclose(s["r_y_shoulder"], _HALF_PI, abs_tol=1e-9)

    got_l = _left_chain(s["l_y_shoulder"], s["l_p_shoulder"], s["l_r_shoulder"])
    got_r = _right_chain(s["r_y_shoulder"], s["r_p_shoulder"], s["r_r_shoulder"])
    assert np.allclose(got_l, target, atol=1e-9)
    assert np.allclose(got_r, target, atol=1e-9)


def test_non_shoulder_joints_passthrough() -> None:
    names = ["l_r_hip", "l_elbow", "waist", "r_knee", "p_head"]
    positions = [0.45, 0.8, 0.03, -0.9, 0.1]
    out = semantic_to_rig(names, positions)
    assert out == positions


def test_torso_and_ankle_passthrough() -> None:
    names = ["r_waist", "y_waist", "l_ankle", "r_ankle"]
    positions = [0.1, -0.2, 0.3, -0.15]
    out = semantic_to_rig(names, positions)
    assert out == positions


def test_mixed_sequence_preserves_order_and_length() -> None:
    names = ["waist", "l_y_shoulder", "l_p_shoulder", "l_r_shoulder", "r_waist", "l_ankle"]
    positions = [0.02, 0.0, 0.5, 0.0, 0.1, 0.3]
    out = semantic_to_rig(names, positions)
    assert len(out) == len(names)
    assert out[0] == 0.02
    assert math.isclose(out[1], _HALF_PI, abs_tol=1e-9)
    assert math.isclose(out[2], 0.5, abs_tol=1e-9)
    assert math.isclose(out[3], _HALF_PI, abs_tol=1e-9)
    assert out[4] == 0.1
    assert out[5] == 0.3


@pytest.mark.parametrize("pair", _random_yr_pairs(50, seed=2))
@pytest.mark.parametrize("p", [math.pi, -math.pi])
def test_degenerate_arm_overhead_folds_correctly(pair: tuple[float, float], p: float) -> None:
    """p=+-pi points the limb up the twist axis (cos(b)=-1), the other degenerate half."""
    y, r = pair
    s = _translate_shoulders(y, p, r)
    target = _semantic_rotation(y, p, r)

    for side in ("l", "r"):
        assert math.isclose(s[f"{side}_y_shoulder"], _HALF_PI, abs_tol=1e-9)
        assert math.isclose(s[f"{side}_p_shoulder"], math.pi, abs_tol=1e-9)

    got_l = _left_chain(s["l_y_shoulder"], s["l_p_shoulder"], s["l_r_shoulder"])
    got_r = _right_chain(s["r_y_shoulder"], s["r_p_shoulder"], s["r_r_shoulder"])
    assert np.allclose(got_l, target, atol=1e-9)
    assert np.allclose(got_r, target, atol=1e-9)


@pytest.mark.parametrize("p", [math.pi, -math.pi])
def test_arm_overhead_keeps_sagittal_constants(p: float) -> None:
    s = _translate_shoulders(0.0, p, 0.0)
    for side in ("l", "r"):
        assert math.isclose(s[f"{side}_y_shoulder"], _HALF_PI, abs_tol=1e-9)
        assert math.isclose(s[f"{side}_p_shoulder"], math.pi, abs_tol=1e-9)
        assert math.isclose(s[f"{side}_r_shoulder"], _HALF_PI, abs_tol=1e-9)


def test_flexion_below_degenerate_threshold_holds_first_angle() -> None:
    s = _translate_shoulders(0.7, 1e-9, 0.3)
    target = _semantic_rotation(0.7, 1e-9, 0.3)
    assert s["l_y_shoulder"] == _HALF_PI
    assert s["r_y_shoulder"] == _HALF_PI
    assert np.allclose(_left_chain(s["l_y_shoulder"], s["l_p_shoulder"], s["l_r_shoulder"]), target, atol=1e-9)
    assert np.allclose(_right_chain(s["r_y_shoulder"], s["r_p_shoulder"], s["r_r_shoulder"]), target, atol=1e-9)


_BRANCH_CASES = [
    ((1.0, 0.5, 0.2), (_HALF_PI - 1.0, 0.5, _HALF_PI - 0.2), (_HALF_PI + 1.0, 0.5, _HALF_PI + 0.2)),
    ((0.5, -0.7, 0.1), (_HALF_PI - 0.5, -0.7, _HALF_PI - 0.1), (_HALF_PI + 0.5, -0.7, _HALF_PI + 0.1)),
    ((2.5, 1.2, -0.3), (3 * _HALF_PI - 2.5, -1.2, -_HALF_PI + 0.3), (2.5 - _HALF_PI, -1.2, -_HALF_PI - 0.3)),
]


@pytest.mark.parametrize(("wire", "left", "right"), _BRANCH_CASES)
def test_branch_flip_keeps_first_angle_in_half_turn(wire: tuple[float, float, float], left: tuple[float, float, float], right: tuple[float, float, float]) -> None:
    """Canonical extractions with a negative first angle fold to (a+pi, -b, c+pi), the others stay."""
    s = _translate_shoulders(*wire)
    got_left = (s["l_y_shoulder"], s["l_p_shoulder"], s["l_r_shoulder"])
    got_right = (s["r_y_shoulder"], s["r_p_shoulder"], s["r_r_shoulder"])
    assert np.allclose(got_left, left, atol=1e-9)
    assert np.allclose(got_right, right, atol=1e-9)
    for a in (got_left[0], got_right[0]):
        assert 0.0 <= a < math.pi
    target = _semantic_rotation(*wire)
    assert np.allclose(_left_chain(*got_left), target, atol=1e-9)
    assert np.allclose(_right_chain(*got_right), target, atol=1e-9)


@pytest.mark.parametrize("triple", _random_wire_triples(200, seed=3))
def test_first_angle_always_in_half_turn(triple: tuple[float, float, float]) -> None:
    s = _translate_shoulders(*triple)
    assert 0.0 <= s["l_y_shoulder"] < math.pi
    assert 0.0 <= s["r_y_shoulder"] < math.pi


def test_single_side_triple_translates_only_that_side() -> None:
    names = ["l_y_shoulder", "l_p_shoulder", "l_r_shoulder", "r_y_shoulder", "r_p_shoulder"]
    positions = [0.0, 0.5, 0.0, 0.4, 0.6]
    out = semantic_to_rig(names, positions)
    assert np.allclose(out[:3], [_HALF_PI, 0.5, _HALF_PI], atol=1e-9)
    assert out[3:] == [0.4, 0.6]


def test_right_triple_without_left_translates_right() -> None:
    names = ["r_r_shoulder", "r_p_shoulder", "r_y_shoulder"]
    positions = [0.0, -0.4, 0.0]
    out = semantic_to_rig(names, positions)
    assert np.allclose(out, [_HALF_PI, -0.4, _HALF_PI], atol=1e-9)


def test_mismatched_names_and_positions_raise() -> None:
    with pytest.raises(ValueError):
        semantic_to_rig(["l_elbow", "r_elbow"], [0.1])


def test_rot_axis_zero_axis_is_identity() -> None:
    assert np.array_equal(_rot_axis(np.zeros(3), 1.3), np.eye(3))


@pytest.mark.parametrize("axis", [(0.0, 0.0, 2.5), (1.0, -2.0, 0.5), (-3.0, 0.0, 0.0)])
def test_rot_axis_matches_rodrigues_for_non_unit_axis(axis: tuple[float, float, float]) -> None:
    got = _rot_axis(np.array(axis), 0.8)
    assert np.allclose(got, _Raxis(axis, 0.8), atol=1e-12)
    assert np.allclose(got @ np.array(axis), np.array(axis), atol=1e-12)
    assert np.allclose(got @ got.T, np.eye(3), atol=1e-12)
