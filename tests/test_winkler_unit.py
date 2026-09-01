# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Fast unit tests for the Winkler readout algorithm.

Unlike the scenario suite these never step a solver: models are finalized,
poses are written by hand into the state, and the sensor is a light stand-in
exposing exactly the surface :class:`WinklerReadout` reads. Each test pins one
property of the algorithm — signed distances per shape, uniform split under a
flat face, linear gradient under a tilt, patch symmetry under a sphere,
hovering-body exclusion, two-body ownership, passthrough — in milliseconds,
so the algorithm can change without waiting on the settling suite.

Backend note: ``make_readout`` is parameterized over implementation backends.
Today numpy is the only one; when a Warp kernel backend lands it joins
``BACKENDS`` and this module doubles as the numpy-vs-warp equivalence harness.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="warp not installed")
newton = pytest.importorskip("newton", reason="newton not installed")

from synaptics_tactile_newton import WinklerReadout

pytestmark = pytest.mark.unit

#: The readout's defaults, restated so the expectations below are explicit.
DELTA0 = 3.0e-4
#: Taxel spacing [m] for the hand-built rows (same scale as the CTS grid).
PITCH = 2.5e-3

#: Implementation backends the whole module runs against.
BACKENDS = ["numpy"]


@pytest.fixture(params=BACKENDS)
def make_readout(request):
    """Construct a :class:`WinklerReadout` on the parameterized backend."""

    def factory(sensor, model, **kwargs):
        assert request.param == "numpy"  # the only backend so far
        return WinklerReadout(sensor, model, **kwargs)

    return factory


class FakeSensor:
    """The exact surface ``WinklerReadout`` reads off a ``CTSSensor``.

    Static world taxels (body ``-1``) at hand-chosen positions, with
    hand-chosen "engine" forces — no USD asset, no ``SensorContact``.
    """

    def __init__(self, positions, forces, force_max: float = 100.0):
        positions = np.asarray(positions, dtype=np.float32)
        self.taxel_bodies = np.full(len(positions), -1, dtype=np.int32)
        self.force_max = float(force_max)
        self.sensing_axis = np.array([0.0, 0.0, 1.0])
        self.data = SimpleNamespace(
            force=wp.array(np.asarray(forces, dtype=np.float32), dtype=wp.float32),
            positions_w=wp.array(positions, dtype=wp.vec3),
        )

    def forces(self) -> np.ndarray:
        return self.data.force.numpy()


def row_taxels(n: int = 5, z: float = 0.0) -> np.ndarray:
    """``n`` taxel positions in a centred row along x at height ``z``."""
    xs = (np.arange(n) - (n - 1) / 2.0) * PITCH
    return np.stack([xs, np.zeros(n), np.full(n, z)], axis=1)


def build_pressers(specs):
    """Finalize a model of free bodies, one shape each; no solver, no ground.

    ``specs`` is a list of ``(kind, params)`` with kind in box/sphere/capsule/
    cylinder and params passed to the matching ``add_shape_*``. Returns
    ``(model, state, bodies)``; place the bodies with :func:`set_pose`.
    """
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 1000.0
    bodies = []
    for i, (kind, params) in enumerate(specs):
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            label=f"presser_{i}",
        )
        adder = getattr(builder, f"add_shape_{kind}")
        adder(body=body, cfg=cfg, label=f"presser_{i}_shape", **params)
        bodies.append(body)
    model = builder.finalize()
    return model, model.state(), bodies


def set_pose(state, body: int, pos, quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> None:
    """Write one body's world pose straight into the state."""
    q = state.body_q.numpy()
    q[body, 0:3] = pos
    q[body, 3:7] = quat_xyzw
    state.body_q.assign(q)


def quat_about_y(deg: float) -> tuple:
    half = math.radians(deg) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def gap_row(readout, body: int) -> np.ndarray:
    """The readout's last computed gaps for one body."""
    return readout.gaps[readout.bodies.index(body)]


# --------------------------------------------------------------------------- #
# Signed distances per shape
# --------------------------------------------------------------------------- #


def test_gap_box(make_readout):
    """Under the face: plane distance. Past the edge: corner distance."""
    probes = np.array([[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]])
    sensor = FakeSensor(probes, np.zeros(len(probes)))
    model, state, (body,) = build_pressers(
        [("box", dict(hx=0.005, hy=0.005, hz=0.005))]
    )
    set_pose(state, body, (0.0, 0.0, 0.006))  # bottom face at z = 1 mm

    readout = make_readout(sensor, model, aperture=None)
    readout.update(state)
    gaps = gap_row(readout, body)
    assert gaps[0] == pytest.approx(1.0e-3, rel=1e-6)
    # 5 mm past the face edge in x, 1 mm below it in z.
    assert gaps[1] == pytest.approx(math.hypot(5.0e-3, 1.0e-3), rel=1e-6)


def test_gap_sphere(make_readout):
    probes = np.array([[0.0, 0.0, 0.0]])
    sensor = FakeSensor(probes, np.zeros(1))
    model, state, (body,) = build_pressers([("sphere", dict(radius=0.004))])
    set_pose(state, body, (0.0, 0.0, 0.005))
    readout = make_readout(sensor, model, aperture=None)
    readout.update(state)
    assert gap_row(readout, body)[0] == pytest.approx(1.0e-3, rel=1e-6)


def test_gap_capsule(make_readout):
    """Beside the barrel: radial. Past the cap: distance to the end sphere."""
    # Upright capsule (axis = z): radius 2 mm, segment half-length 5 mm.
    probes = np.array([[0.003, 0.0, 0.010], [0.0, 0.0, 0.018]])
    sensor = FakeSensor(probes, np.zeros(len(probes)))
    model, state, (body,) = build_pressers(
        [("capsule", dict(radius=0.002, half_height=0.005))]
    )
    set_pose(state, body, (0.0, 0.0, 0.010))
    readout = make_readout(sensor, model, aperture=None)
    readout.update(state)
    gaps = gap_row(readout, body)
    assert gaps[0] == pytest.approx(1.0e-3, rel=1e-6)          # barrel side
    assert gaps[1] == pytest.approx(3.0e-3 - 2.0e-3, rel=1e-6)  # past the cap


def test_gap_cylinder_lying_down(make_readout):
    """A cylinder rotated onto its side: barrel below, rim corner past the end."""
    radius, half_len = 0.001, 0.010
    probes = np.array([[0.0, 0.0, 0.0], [0.012, 0.0, 0.0]])
    sensor = FakeSensor(probes, np.zeros(len(probes)))
    model, state, (body,) = build_pressers(
        [("cylinder", dict(radius=radius, half_height=half_len))]
    )
    # +90 deg about y maps the local z axis onto world x: a lying "wire".
    set_pose(state, body, (0.0, 0.0, 0.002), quat_about_y(90.0))
    readout = make_readout(sensor, model, aperture=None)
    readout.update(state)
    gaps = gap_row(readout, body)
    assert gaps[0] == pytest.approx(1.0e-3, rel=1e-5)  # under the barrel
    # 2 mm past the flat end, 1 mm below the rim.
    assert gaps[1] == pytest.approx(math.hypot(2.0e-3, 1.0e-3), rel=1e-5)


# --------------------------------------------------------------------------- #
# Distribution properties
# --------------------------------------------------------------------------- #


def test_flat_box_reads_uniform(make_readout):
    """The motivating case: engine reads 2,8,7,8,9 — corrected reads flat."""
    taxels = row_taxels(5)
    engine = [2.0, 8.0, 7.0, 8.0, 9.0]
    sensor = FakeSensor(taxels, engine)
    model, state, (body,) = build_pressers(
        [("box", dict(hx=0.010, hy=0.005, hz=0.005))]
    )
    set_pose(state, body, (0.0, 0.0, 0.005))  # face exactly on the taxel plane

    corrected = make_readout(sensor, model).update(state)
    assert corrected == pytest.approx(np.full(5, sum(engine) / 5.0), rel=1e-6)


def test_tilted_box_reads_linear_gradient(make_readout):
    """A pitched face: weights fall off linearly along the tilt."""
    taxels = row_taxels(5)
    sensor = FakeSensor(taxels, np.full(5, 2.0))
    model, state, (body,) = build_pressers(
        [("box", dict(hx=0.010, hy=0.005, hz=0.005))]
    )
    # Tilt so the gap grows ~delta0/8 per pitch: every taxel stays engaged
    # and the profile must come out as a straight line.
    angle = math.degrees(math.atan((DELTA0 / 8.0) / PITCH))
    set_pose(state, body, (0.0, 0.0, 0.005), quat_about_y(angle))

    corrected = make_readout(sensor, model).update(state)
    assert float(corrected.sum()) == pytest.approx(10.0, rel=1e-6)
    diffs = np.diff(corrected.astype(np.float64))
    assert (diffs > 0).all() or (diffs < 0).all(), corrected
    assert diffs == pytest.approx(np.full(4, diffs.mean()), rel=5e-2), corrected


def test_sphere_reads_symmetric_patch(make_readout):
    """A big sphere: peak under the centre, symmetric shoulders, cold ends.

    The engine reading is plausible-but-lopsided inside the patch and zero
    outside it — a taxel outside the patch that *did* read force would keep it
    (the passthrough rule, pinned by the hovering test), so cold ends here
    must stay exactly cold.
    """
    taxels = row_taxels(5)
    engine = [0.0, 1.0, 3.0, 1.0, 0.0]
    sensor = FakeSensor(taxels, engine)
    model, state, (body,) = build_pressers([("sphere", dict(radius=0.020))])
    set_pose(state, body, (0.0, 0.0, 0.020))  # touching the middle taxel

    corrected = make_readout(sensor, model).update(state).astype(np.float64)
    assert float(corrected.sum()) == pytest.approx(sum(engine), rel=1e-6)
    assert corrected.argmax() == 2
    assert corrected[1] == pytest.approx(corrected[3], rel=1e-4)
    assert 0.0 < corrected[1] < corrected[2]
    # sqrt(2 R delta0) ~ 3.5 mm patch radius: the outermost taxels at 5 mm
    # sit outside it and read nothing.
    assert corrected[0] == corrected[4] == 0.0


def test_hovering_body_is_excluded(make_readout):
    """A body above the surface claims nothing; unclaimed taxels pass through."""
    taxels = row_taxels(5)
    engine = [5.0, 5.0, 0.0, 0.0, 0.4]  # 0.4 N of contact the model can't explain
    sensor = FakeSensor(taxels, engine)
    half = dict(hx=0.002, hy=0.005, hz=0.002)
    model, state, (rest, hover) = build_pressers([("box", half), ("box", half)])
    set_pose(state, rest, (-3.75e-3, 0.0, 0.002))   # covers taxels 0-1, touching
    set_pose(state, hover, (+3.75e-3, 0.0, 0.007))  # covers taxels 3-4, 5 mm up

    readout = make_readout(sensor, model)
    corrected = readout.update(state)
    assert hover not in readout.owners, "hovering body claimed a taxel"
    assert corrected[0] == pytest.approx(5.0, rel=1e-6)
    assert corrected[1] == pytest.approx(5.0, rel=1e-6)
    # Unclaimed taxel keeps the engine's reading rather than losing it.
    assert corrected[4] == pytest.approx(0.4, rel=1e-6)


def test_two_resting_bodies_split_by_ownership(make_readout):
    """Each body's own engine total is redistributed over its own taxels."""
    taxels = row_taxels(5)
    engine = [3.0, 1.0, 0.0, 2.0, 6.0]
    sensor = FakeSensor(taxels, engine)
    half = dict(hx=0.002, hy=0.005, hz=0.002)
    model, state, (left, right) = build_pressers([("box", half), ("box", half)])
    set_pose(state, left, (-3.75e-3, 0.0, 0.002))
    set_pose(state, right, (+3.75e-3, 0.0, 0.002))

    readout = make_readout(sensor, model)
    corrected = readout.update(state)
    assert corrected == pytest.approx([2.0, 2.0, 0.0, 4.0, 4.0], rel=1e-6)
    assert readout.owners[2] == -1, "the cold middle taxel belongs to nobody"


def test_no_supported_presser_passes_through(make_readout):
    """With nothing to model, the readout must not touch the engine forces."""
    engine = [1.0, 2.0]
    sensor = FakeSensor(row_taxels(2), engine)
    model, state, _ = build_pressers([])

    readout = make_readout(sensor, model)
    assert not readout.active
    assert readout.update(state) == pytest.approx(engine)
    assert sensor.forces() == pytest.approx(engine)


def test_diagonal_wire_fills_every_column(make_readout):
    """The wire-drop failure: a thin diagonal line must not read as holes.

    A 2 mm wire lying at ~18 deg across a 5x5 grid passes up to ~1.8 mm from
    some covered cells' centroids — farther than the wire's Winkler patch
    (sqrt(2 R delta0) ~ 0.8 mm) — so point sampling reads holes in the line.
    The cell-aperture average (default) must light every column the wire
    crosses; point sampling is asserted to show at least one hole, which is
    exactly why the aperture exists.
    """
    n = 5
    xs = (np.arange(n) - (n - 1) / 2.0) * PITCH
    taxels = np.array([(x, y, 0.0) for y in xs for x in xs])
    heading = math.radians(18.0)
    y0 = 1.25e-3  # offset so the line passes between centroids in some columns
    radius = 1.0e-3

    # Engine forces the way the manifold really places them for a line: at
    # the ENDPOINTS only (the 10001 signature) — cells near where the wire
    # meets the grid edges. Anything filled in between must come from the
    # Winkler redistribution, not from passthrough.
    engine = np.zeros(len(taxels))
    for ex in (-xs[-1], xs[-1]):
        ey = y0 + math.tan(heading) * ex
        engine[np.argmin(np.linalg.norm(taxels[:, :2] - (ex, ey), axis=1))] = 1.0
    assert engine.sum() == 2.0

    model, state, (body,) = build_pressers(
        [("cylinder", dict(radius=radius, half_height=0.012))]
    )
    # Rotate the local z axis onto the in-plane heading, wire touching z=0.
    axis = np.array([-math.sin(heading), math.cos(heading), 0.0])
    half = math.pi / 4.0
    quat = (
        math.sin(half) * axis[0],
        math.sin(half) * axis[1],
        math.sin(half) * axis[2],
        math.cos(half),
    )
    set_pose(state, body, (0.0, y0, radius), quat)

    sensor = FakeSensor(taxels, engine)
    corrected = make_readout(sensor, model).update(state)
    col_sums = corrected.reshape(n, n).sum(axis=0)
    assert (col_sums > 0.005 * corrected.sum()).all(), (
        f"aperture readout leaves a hole: column sums {col_sums.round(4)}"
    )
    assert float(corrected.sum()) == pytest.approx(float(engine.sum()), rel=1e-6)

    sensor_pt = FakeSensor(taxels, engine)
    point = make_readout(sensor_pt, model, aperture=None).update(state)
    pt_cols = point.reshape(n, n).sum(axis=0)
    assert (pt_cols < 1e-6).any(), (
        f"expected point sampling to show a hole, got columns {pt_cols.round(4)}"
    )
