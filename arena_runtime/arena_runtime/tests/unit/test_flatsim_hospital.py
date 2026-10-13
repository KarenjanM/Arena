"""FlatWorld on the walls of hospital_1: scans against analytic intersections, byte-identical reruns."""

from __future__ import annotations

import math
import pathlib

import typing

import numpy as np
import pytest

if typing.TYPE_CHECKING:
    from flatsim.world import FlatWorld, ScanSpec

DT = 1.0 / 30.0
SQUARE = [(0.2, 0.2), (0.2, -0.2), (-0.2, -0.2), (-0.2, 0.2)]


WORLDS = pathlib.Path(__file__).resolve().parents[4] / 'arena_simulation_setup' / 'worlds'


def _fan() -> ScanSpec:
    from flatsim.world import ScanSpec

    return ScanSpec(mount=(0.0, 0.0, 0.0), angle_min=-math.pi, angle_increment=2 * math.pi / 640, beams=640, range_min=0.0, range_max=12.0)


def _hospital() -> tuple[np.ndarray, np.ndarray]:
    from arena_simulation_setup.tree.World import WorldView

    level = WorldView(WORLDS / 'hospital_1').load().all_levels[0]
    segments = np.array([[w.start.x, w.start.y, w.end.x, w.end.y] for w in level.all_walls])
    hallway = next(zone for zone in level.zones if zone.name == 'central_hallway')
    return segments, np.mean([[p.x, p.y] for p in hallway.polygon], axis=0)


def _intersections(origin: np.ndarray, segments: np.ndarray, spec: ScanSpec) -> np.ndarray:
    angles = spec.angle_min + spec.angle_increment * np.arange(spec.beams)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    start, edge = segments[:, :2], segments[:, 2:] - segments[:, :2]
    rel = start - origin
    den = dirs[:, 0, None] * edge[None, :, 1] - dirs[:, 1, None] * edge[None, :, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (rel[None, :, 0] * edge[None, :, 1] - rel[None, :, 1] * edge[None, :, 0]) / den
        u = (rel[None, :, 0] * dirs[:, 1, None] - rel[None, :, 1] * dirs[:, 0, None]) / den
    t = np.where((t >= 0) & (u >= 0) & (u <= 1), t, np.inf).min(axis=1)
    return np.where(t <= spec.range_max, t, np.inf)


def test_scan_matches_analytic_intersections_with_hospital_walls():
    pytest.importorskip('flatsim')
    from flatsim.drive import Axis, DriveModel
    from flatsim.world import FlatWorld

    segments, origin = _hospital()
    world = FlatWorld(DT)
    for i, (x0, y0, x1, y1) in enumerate(segments):
        world.add_outline(f'wall_{i}', [[(x0, y0), (x1, y1)]], (0.0, 2.5))
    world.add_robot('robot', (origin[0], origin[1], 0.0), 0.0, DriveModel(linear=Axis(-1.0, 1.0), lateral=Axis(0.0, 0.0), angular=Axis(-2.0, 2.0)), radius=0.3)
    ranges = world.scan('robot', _fan())
    expected = _intersections(origin, segments, _fan())
    assert np.isfinite(expected).sum() > 500
    assert np.array_equal(np.isfinite(ranges), np.isfinite(expected))
    hit = np.isfinite(expected)
    assert np.abs(ranges[hit] - expected[hit]).max() < 1e-3


def _scripted_run() -> bytes:
    from flatsim.drive import Axis, DriveModel
    from flatsim.world import FlatWorld

    segments, origin = _hospital()
    world = FlatWorld(DT)
    for i, (x0, y0, x1, y1) in enumerate(segments):
        world.add_outline(f'wall_{i}', [[(x0, y0), (x1, y1)]], (0.0, 2.5))
    world.add_robot('robot', (origin[0], origin[1], 0.3), 0.0, DriveModel(linear=Axis(-0.5, 0.5, 2.5), lateral=Axis(0.0, 0.0), angular=Axis(-1.5, 1.5, 3.2)), footprint=SQUARE)
    world.add_disc('ped', 0.3, (origin[0] + 1.5, origin[1] + 0.2), (0.0, 1.8))
    trace = bytearray()
    for i in range(1, 240):
        world.command('robot', (0.5, 0.0, 0.8 * math.sin(i / 20.0)))
        world.move('ped', (origin[0] + 1.5 - i * 0.01, origin[1] + 0.2, 0.0), velocity=(-0.3, 0.0))
        world.advance(i * DT)
        trace += np.array(_state(world), dtype=np.float64).tobytes()
        if i % 3 == 0:
            trace += world.scan('robot', _fan()).tobytes()
    return bytes(trace)


def _state(world: FlatWorld) -> tuple[float, ...]:
    state = world.robot_state('robot')
    return (state.x, state.y, state.z, state.yaw, state.vx, state.vy, state.wz)


def test_same_inputs_give_identical_bytes():
    pytest.importorskip('flatsim')
    first = _scripted_run()
    assert len(first) > 100_000
    assert first == _scripted_run()
