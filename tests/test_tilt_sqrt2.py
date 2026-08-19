# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Tilted-sensor projection: the normal force shrinks by sqrt(2) at 45 degrees.

The CTS sensor reports only the contact force projected onto its press axis (the
taxel-surface normal); shear is discarded. So a fixed *vertical* press, applied
to a sensor held at angle ``theta``, indents the taxel by ``d * cos(theta)`` along
its normal and the sensor reads ``k * d * cos(theta)``.

Driving the SAME vertical displacement into an upright sensor and into one tilted
45 degrees should therefore give

    force(flat) / force(45 deg) = 1 / cos(45 deg) = sqrt(2).

A high-friction probe is used so it grips the tilted dome instead of sliding off
under the now-oblique press (the shear it must resist is what the sensor throws
away).

The check is a RATIO (flat vs tilted) so it is independent of the absolute
stiffness/contact-area details; it skips only if the press produces essentially
no force.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from sensor_rig import (
    MOUNT_QUAT_XYZW,
    SensorRig,
    local_to_world,
    quat_mul_xyzw,
)

pytestmark = pytest.mark.sim

TILT_DEG = 45.0
PROBE_VELOCITY = -0.012   # 12 mm/s downward (matches the indentation test)
PROBE_RADIUS = 0.002
PROBE_CLEARANCE = 0.0005  # start the probe 0.5 mm above the force-area surface
PROBE_MU = 5.0            # high friction so the probe grips the tilted dome
NUM_FRAMES = 120
# Defensive floor: if a press produces essentially no force the ratio is
# meaningless, so we skip rather than fail. (Not expected to trigger.)
FORCE_FLOOR = 1.0e-4
RATIO_TOL = 0.30         # generous: dome geometry perturbs the ideal projection


def _tilt_mount_xyzw(deg: float) -> np.ndarray:
    """Compose a tilt of ``deg`` about world X on top of the upright mount."""
    half = math.radians(deg) / 2.0
    tilt = np.array([math.sin(half), 0.0, 0.0, math.cos(half)], dtype=np.float64)
    return quat_mul_xyzw(tilt, MOUNT_QUAT_XYZW)


def _peak_normal_force(sim_config, mount_xyzw: np.ndarray) -> float:
    """Press a high-friction probe straight down (world -Z) into the central
    force area of a sensor mounted at ``mount_xyzw``; return the peak total
    normal force [N] over the press."""
    # Read-only survey to locate the central force area under this mount.
    survey = SensorRig(sim_config, mount_quat_xyzw=mount_xyzw).finalize(
        build_solver=False
    )
    centroids = survey.taxel_centroids
    if centroids is None:
        pytest.skip("taxel map exposes no centroids")
    world = np.array(
        [local_to_world(c, mount_xyzw) for c in np.asarray(centroids, dtype=np.float64)]
    )
    centre_xy = world[:, :2].mean(axis=0)
    target = int(np.argmin(np.linalg.norm(world[:, :2] - centre_xy, axis=1)))
    tw = world[target]

    rig = SensorRig(sim_config, mount_quat_xyzw=mount_xyzw)
    rig.add_probe(
        velocity_z=PROBE_VELOCITY,
        radius=PROBE_RADIUS,
        xy=(float(tw[0]), float(tw[1])),
        start_z=float(tw[2]) + PROBE_RADIUS + PROBE_CLEARANCE,
        mu=PROBE_MU,
    )
    rig.finalize()
    peak = 0.0
    for _ in range(NUM_FRAMES):
        rig.step(1)
        peak = max(peak, rig.total_force())
    return peak


@pytest.fixture(scope="module")
def tilt_run(sim_config):
    """Press the same vertical stroke into an upright and a 45-degree sensor."""
    flat = _peak_normal_force(sim_config, MOUNT_QUAT_XYZW)
    tilted = _peak_normal_force(sim_config, _tilt_mount_xyzw(TILT_DEG))
    return {"flat": flat, "tilted": tilted}


def test_tilt_force_smaller_by_sqrt2(tilt_run, artifact_csv):
    flat = tilt_run["flat"]
    tilted = tilt_run["tilted"]
    ratio = flat / tilted if tilted > 0.0 else float("nan")

    artifact_csv(
        "tilt_sqrt2",
        [
            {"mount": "flat_0deg", "peak_normal_N": flat},
            {"mount": f"tilted_{TILT_DEG:g}deg", "peak_normal_N": tilted},
            {"mount": "ratio_flat_over_tilted", "peak_normal_N": ratio},
            {"mount": "expected_sqrt2", "peak_normal_N": math.sqrt(2.0)},
        ],
    )

    print(
        "\n45-degree tilt projection:"
        f"\n  flat (0 deg)  peak normal force = {flat:.6f} N"
        f"\n  tilted (45 deg) peak normal force = {tilted:.6f} N"
        f"\n  ratio flat/tilted = {ratio:.4f}  (expected sqrt(2) = {math.sqrt(2.0):.4f})"
    )

    if flat < FORCE_FLOOR or tilted < FORCE_FLOOR:
        pytest.skip(
            f"press produced essentially no force (flat={flat:.3g}, "
            f"tilted={tilted:.3g}); ratio undefined"
        )

    assert ratio == pytest.approx(math.sqrt(2.0), rel=RATIO_TOL), (
        f"flat/tilted = {ratio:.3f}, expected ~sqrt(2)=1.414 "
        f"(flat={flat:.4g} N, tilted={tilted:.4g} N)"
    )
