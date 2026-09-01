# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Winkler readout: a flat object at rest reads FLAT across the taxels.

The engine's raw per-taxel split under a flat resting plate is statically
indeterminate (an accident of contact-point placement), so it is uneven and
tuning-dependent. :class:`WinklerReadout` replaces the split with the analytic
Winkler-foundation distribution, renormalized to the engine's total — so a
level plate must read uniform across the covered taxels while the total stays
exactly the engine's. These tests settle a plate across the whole sensor and
check both properties, plus footprint locality for a small cube.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensor_rig import SENSOR_Z, DEFAULT_SENSOR_LABEL, SensorRigFactory, centroid_to_world
from synaptics_tactile_newton import WinklerReadout

pytestmark = pytest.mark.sim

PLATE_MASS = 0.05             # kg
PLATE_HALF_THICKNESS = 0.001  # m
PLATE_MARGIN = 0.002          # m overhang beyond the force-area footprint
DROP_GAP = 0.005              # m start clearance above the force-area tips

CUBE_MASS = 0.02              # kg
CUBE_HALF = 0.005             # m — covers a few taxels, not the whole array


def _footprint(sim_config):
    """World-frame extents of the force-area array (from the taxel map)."""
    base = SensorRigFactory(config=sim_config)().finalize(build_solver=False)
    centroids = base.taxel_centroids
    if centroids is None:
        pytest.skip("taxel map exposes no centroids")
    world = np.array([centroid_to_world(c) for c in centroids], dtype=np.float64)
    return world


@pytest.fixture(scope="module")
def settled_plate(sim_config):
    """A level plate covering every force area, settled, with its Winkler readout."""
    world = _footprint(sim_config)
    cx, cy = world[:, 0].mean(), world[:, 1].mean()
    hx = (world[:, 0].max() - world[:, 0].min()) / 2.0 + PLATE_MARGIN
    hy = (world[:, 1].max() - world[:, 1].min()) / 2.0 + PLATE_MARGIN
    tip_z = world[:, 2].max()

    rig = SensorRigFactory(config=sim_config)()
    height = tip_z + PLATE_HALF_THICKNESS + DROP_GAP - SENSOR_Z
    body = rig.add_box(
        mass=PLATE_MASS,
        half_extents=(hx, hy, PLATE_HALF_THICKNESS),
        xy=(float(cx), float(cy)),
        height=float(height),
        label="plate",
    )
    rig.finalize()
    rig.settle()

    raw = rig.forces().copy()
    winkler = WinklerReadout(rig.sensors[DEFAULT_SENSOR_LABEL], rig.model)
    corrected = winkler.update(rig.state_0).copy()
    return {"rig": rig, "raw": raw, "corrected": corrected, "winkler": winkler,
            "body": body}


def test_flat_plate_reads_flat(settled_plate):
    """Every covered taxel reads the same force — the Winkler flat profile."""
    corrected = settled_plate["corrected"]
    assert corrected.sum() > 0.0, "plate settled without loading the sensor"
    mean = corrected.mean()
    # A level plate over the full array puts every taxel at delta0, so the
    # profile is uniform up to the plate's residual settling tilt.
    assert corrected == pytest.approx(np.full_like(corrected, mean), rel=0.05), (
        f"spread {corrected.min():.4f}..{corrected.max():.4f} N "
        f"around mean {mean:.4f} N"
    )


def test_total_is_preserved(settled_plate):
    """Renormalization keeps the engine's total normal force exactly."""
    raw, corrected = settled_plate["raw"], settled_plate["corrected"]
    assert float(corrected.sum()) == pytest.approx(float(raw.sum()), rel=1.0e-5)


def test_raw_engine_split_was_the_problem(settled_plate):
    """Document the motivation: the raw split is (much) less flat than Winkler's.

    Skips rather than fails if the engine happens to produce a flat split — the
    fix must not depend on the engine misbehaving.
    """
    raw, corrected = settled_plate["raw"], settled_plate["corrected"]
    loaded = raw > 1.0e-4
    if not loaded.any():
        pytest.skip("engine reported no per-taxel forces")

    def spread(f):
        return float(f.max() - f.min()) / max(float(f.mean()), 1.0e-9)

    raw_spread = spread(raw[loaded])
    win_spread = spread(corrected)
    print(
        f"\nraw engine spread (max-min)/mean over loaded taxels: {raw_spread:.3f}"
        f"\nwinkler spread over all taxels:                      {win_spread:.3f}"
    )
    if raw_spread <= win_spread:
        pytest.skip("engine split was already flat here")
    assert win_spread < raw_spread


def test_small_cube_footprint_locality(sim_config):
    """A small cube loads only the taxels under it, uniformly, summing unchanged."""
    world = _footprint(sim_config)
    cx, cy = world[:, 0].mean(), world[:, 1].mean()
    target = int(np.argmin(np.linalg.norm(world[:, :2] - (cx, cy), axis=1)))
    drop_xy = (float(world[target, 0]), float(world[target, 1]))
    tip_z = float(world[target, 2])

    rig = SensorRigFactory(config=sim_config)()
    body = rig.add_box(
        mass=CUBE_MASS,
        half_extents=(CUBE_HALF, CUBE_HALF, CUBE_HALF),
        xy=drop_xy,
        height=tip_z + CUBE_HALF + 0.002 - SENSOR_Z,
        label="cube",
    )
    rig.finalize()
    rig.settle()

    raw_total = float(rig.forces().sum())
    if raw_total <= 1.0e-4:
        pytest.skip("cube settled without loading the sensor")

    winkler = WinklerReadout(rig.sensors[DEFAULT_SENSOR_LABEL], rig.model)
    assert winkler.active and body in winkler.bodies
    corrected = winkler.update(rig.state_0)

    positions = rig.sensors[DEFAULT_SENSOR_LABEL].data.positions_w.numpy()
    cube_xy = rig.body_position(body)[:2]
    dist_xy = np.linalg.norm(positions[:, :2] - cube_xy, axis=1)
    outside = dist_xy > CUBE_HALF * np.sqrt(2.0) + 1.0e-3

    assert float(corrected.sum()) == pytest.approx(raw_total, rel=1.0e-5)
    assert corrected[outside].max() == 0.0, "taxel outside the footprint is loaded"
    loaded = corrected > 1.0e-4
    assert loaded.any(), "no taxel under the cube is loaded"
    # The cell aperture integrates coverage: cells whose full aperture lies
    # under the face read uniform; cells the face edge crosses read less.
    interior = loaded & (
        np.abs(positions[:, 0] - cube_xy[0]) < CUBE_HALF - 1.3e-3
    ) & (np.abs(positions[:, 1] - cube_xy[1]) < CUBE_HALF - 1.3e-3)
    assert interior.any(), "no fully covered taxel under the cube"
    vals = corrected[interior]
    assert vals == pytest.approx(np.full_like(vals, vals.mean()), rel=0.10), (
        "fully covered taxels are not uniformly loaded"
    )
    edge = loaded & ~interior
    if edge.any():
        assert corrected[edge].max() < vals.mean(), (
            "a partially covered edge cell reads more than the interior"
        )


BALL_MASS = 0.02   # kg
BALL_RADIUS = 0.004  # m
# A dominant taxel must sit within this XY distance of the ball centre [m]
# (matches test_spatial's locality radius).
_BALL_LOCALITY = 6.0e-3


def test_ball_reads_a_local_patch(sim_config):
    """A resting sphere: peak under its centre, total preserved, distant taxels zero.

    Exercises the sphere signed-distance path — the readout is object-agnostic,
    so a curved presser must work through the same code that handles boxes.
    """
    world = _footprint(sim_config)
    cx, cy = world[:, 0].mean(), world[:, 1].mean()
    target = int(np.argmin(np.linalg.norm(world[:, :2] - (cx, cy), axis=1)))
    drop_xy = (float(world[target, 0]), float(world[target, 1]))
    tip_z = float(world[target, 2])

    rig = SensorRigFactory(config=sim_config)()
    body = rig.add_ball(
        mass=BALL_MASS,
        radius=BALL_RADIUS,
        xy=drop_xy,
        start_z=tip_z + BALL_RADIUS + 0.002,
    )
    rig.finalize()
    rig.settle()

    raw_total = float(rig.forces().sum())
    if raw_total <= 1.0e-4:
        pytest.skip("ball settled without loading the sensor")

    winkler = WinklerReadout(rig.sensors[DEFAULT_SENSOR_LABEL], rig.model)
    assert winkler.active and body in winkler.bodies
    corrected = winkler.update(rig.state_0)

    assert float(corrected.sum()) == pytest.approx(raw_total, rel=1.0e-5)

    positions = rig.sensors[DEFAULT_SENSOR_LABEL].data.positions_w.numpy()
    ball_xy = rig.body_position(body)[:2]
    peak = int(np.argmax(corrected))
    assert np.linalg.norm(positions[peak, :2] - ball_xy) <= _BALL_LOCALITY, (
        "peak taxel is not under the ball"
    )
    far = np.linalg.norm(positions[:, :2] - ball_xy, axis=1) > _BALL_LOCALITY
    assert corrected[far].max() == pytest.approx(0.0, abs=1.0e-6), (
        "taxel far from the ball is loaded"
    )


WIRE_MASS = 0.05        # kg
WIRE_RADIUS = 0.001     # m — 2 mm thick, like the wire-drop demo scene
WIRE_HALF_LENGTH = 0.010  # m — 20 mm long
# A loaded taxel must sit within this of the wire's axis line, laterally [m].
_WIRE_LATERAL = 2.0e-3


def test_wire_reads_a_uniform_line(sim_config):
    """A resting horizontal cylinder: a line of taxels, uniform, total preserved.

    Line contact is the cylinder signed-distance path — the wire-drop and
    rolling-cylinder demo scenes both rest a cylinder on the array.
    """
    world = _footprint(sim_config)
    # Rest the wire along a taxel row: pick the row (y) nearest the footprint
    # centre and centre the wire on it in x.
    cy = world[:, 1].mean()
    row_y = float(world[np.argmin(np.abs(world[:, 1] - cy)), 1])
    cx = float(world[:, 0].mean())
    tip_z = float(world[:, 2].max())

    rig = SensorRigFactory(config=sim_config)()
    body = rig.add_cylinder(
        mass=WIRE_MASS,
        radius=WIRE_RADIUS,
        half_height=WIRE_HALF_LENGTH,
        xy=(cx, row_y),
        start_z=tip_z + WIRE_RADIUS + 0.002,
    )
    rig.finalize()
    rig.settle()

    raw_total = float(rig.forces().sum())
    if raw_total <= 1.0e-4:
        pytest.skip("wire settled without loading the sensor")

    winkler = WinklerReadout(rig.sensors[DEFAULT_SENSOR_LABEL], rig.model)
    assert winkler.active and body in winkler.bodies
    corrected = winkler.update(rig.state_0)

    assert float(corrected.sum()) == pytest.approx(raw_total, rel=1.0e-5)

    positions = rig.sensors[DEFAULT_SENSOR_LABEL].data.positions_w.numpy()
    wire_y = float(rig.body_position(body)[1])
    loaded = corrected > 1.0e-4
    assert int(loaded.sum()) >= 3, "a 20 mm wire should load several taxels"
    assert np.abs(positions[loaded, 1] - wire_y).max() <= _WIRE_LATERAL, (
        "loaded taxel is off the wire's line"
    )
    # Interior of the line (away from the wire's ends) is uniform: line contact
    # puts every interior taxel at the same gap.
    wire_x = float(rig.body_position(body)[0])
    interior = loaded & (
        np.abs(positions[:, 0] - wire_x) <= WIRE_HALF_LENGTH - 2.0e-3
    )
    if int(interior.sum()) >= 2:
        vals = corrected[interior]
        assert vals == pytest.approx(np.full_like(vals, vals.mean()), rel=0.10), (
            "interior of the wire's line is not uniform"
        )
