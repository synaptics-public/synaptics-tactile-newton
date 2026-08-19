# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Quasi-static dead-weight loading.

Spawn a cube of known mass above the sensor, let it settle under gravity, and
check that the steady-state total normal force approaches ``m * g``. Sweeping
mass also exercises linearity and saturation.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensor_rig import GRAVITY, SENSOR_Z, SensorRigFactory, centroid_to_world

pytestmark = pytest.mark.sim

# Cube half-size [m] (matches add_weight's default) and the clearance to leave
# between the cube's underside and the force-area tip at spawn.
_CUBE_HALF = 0.005
_DROP_GAP = 0.002

# (mass [kg], relative tolerance, label). Off-nominal masses get looser tolerance
# because settling residual velocity and contact stiffness perturb the total.
_MASS_MATRIX = [
    (0.010, 0.30, "light_10g"),
    (0.050, 0.20, "medium_50g"),
    (0.200, 0.20, "heavy_200g"),
]


@pytest.fixture(scope="module")
def dead_weight_run(sim_config):
    """Settle each mass once; return {mass: rig-snapshot} for reuse by tests."""
    factory = SensorRigFactory(config=sim_config)

    # Drop over a real force area, not the world origin: the force-area array
    # leaves a gap at (0,0) (only the board sits under the origin), so a cube
    # dropped there lands on the board and never loads a taxel. Pick the area
    # nearest the footprint centre so the cube is well supported.
    base = factory()
    base.finalize(build_solver=False)
    centroids = base.taxel_centroids
    if centroids is None:
        pytest.skip("taxel map exposes no centroids")
    world = np.array([centroid_to_world(c) for c in np.asarray(centroids)])
    centre_xy = world[:, :2].mean(axis=0)
    target = int(np.argmin(np.linalg.norm(world[:, :2] - centre_xy, axis=1)))
    drop_xy = (float(world[target, 0]), float(world[target, 1]))

    # Start the cube a couple mm above that force area's tip, not 2 cm up: a
    # cube dropped from far above just bounces off the housing rim and never
    # loads a taxel. ``height`` is measured from SENSOR_Z (see add_weight).
    tip_z = float(world[target, 2])
    drop_height = tip_z + _CUBE_HALF + _DROP_GAP - SENSOR_Z

    out = {}
    for mass, _rel, _label in _MASS_MATRIX:
        rig = factory()
        rig.add_weight(mass=mass, xy=drop_xy, height=drop_height)
        rig.finalize()
        rig.settle()
        out[mass] = {
            "total": rig.total_force(),
            "max_taxel": float(rig.forces().max()),
            "active": rig.active_count(),
            "force_max": rig.config.force_max,
        }
    return out


def test_zero_load_baseline(make_rig):
    """A body present but not touching the sensor -> all taxels read ~0."""
    rig = make_rig()
    # High, non-contacting mass: gives the model a dynamic joint (required by the
    # MuJoCo solver) while keeping the sensor unloaded.
    rig.add_weight(mass=0.05, height=0.2)
    rig.finalize()
    rig.step(20)
    assert rig.total_force() == pytest.approx(0.0, abs=1.0e-3)


@pytest.mark.parametrize("mass,rel,label", _MASS_MATRIX, ids=[m[2] for m in _MASS_MATRIX])
def test_total_force_matches_mg(dead_weight_run, mass, rel, label):
    snap = dead_weight_run[mass]
    expected = mass * GRAVITY
    assert snap["total"] == pytest.approx(expected, rel=rel, abs=5.0e-3)


@pytest.mark.parametrize("mass,_rel,label", _MASS_MATRIX, ids=[m[2] for m in _MASS_MATRIX])
def test_no_taxel_exceeds_saturation(dead_weight_run, mass, _rel, label):
    snap = dead_weight_run[mass]
    assert snap["max_taxel"] <= snap["force_max"] + 1.0e-6


def test_monotonic_in_mass(dead_weight_run, artifact_csv):
    masses = [m for m, _, _ in _MASS_MATRIX]
    totals = [dead_weight_run[m]["total"] for m in masses]
    artifact_csv(
        "dead_weight",
        [
            {"mass_kg": m, "expected_N": m * GRAVITY, "measured_N": t}
            for m, t in zip(masses, totals)
        ],
    )
    assert all(b >= a - 1.0e-6 for a, b in zip(totals, totals[1:])), totals
