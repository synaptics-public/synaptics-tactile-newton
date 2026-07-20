# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.
#
# This material contains information that is proprietary to Synaptics
# Incorporated. The holder of this document shall treat all information
# contained herein as confidential, shall use the information only for its
# intended purpose and in conjunction with Synaptics products, and shall protect
# the information in whole or part from duplication, disclosure to any other
# party, or dissemination in any media without the written permission of
# Synaptics Incorporated.
#
# Information contained herein is provided AS-IS, with no express or implied
# warranties. SYNAPTICS HEREBY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED,
# INCLUDING, WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE, AND NONINFRINGEMENT. SYNAPTICS ASSUMES NO LIABILITY
# WHATSOEVER, INCLUDING NO LIABILTY FOR INTELLECTUAL PROPERTY INFRINGEMENT, FOR
# ANY DAMAGES, INCLUDING ANY SPECIAL, PUNITIVE, INCIDENTAL, OR CONSEQUENTIAL
# DAMAGES RESULTING FROM THE USE OF THE INFORMATION CONTAINED HEREIN. This
# material conveys no express or implied licenses to any intellectual property
# rights belonging to Synaptics or any other party. Synaptics may, from time to
# time and at its sole option, update the information contained herein without
# notice.
#
# Synaptics Incorporated
# 1109 McKay Drive
# San Jose, CA 95131
# (408) 904-1100

r"""Two static CTS sensors with a ball resting in a 90-degree V (``\o/``).

Exercises multiple independent sensor instances in a single environment. Two CTS
sensors are fixed to the world, each tilted 45 degrees so their sensing faces
meet at 90 degrees and form a V. A free ball settles into the valley, pressing
on both. Each sensor is scoped by a label-prefixed glob, so the two read their
forces independently from the shared contact buffer.

Expected reading (frictionless normal balance). The ball's weight ``mg`` is
carried by two faces at 90 degrees, symmetric about vertical, each 45 degrees
from horizontal. Force balance on the ball gives, per face,

    2 * N * cos(45 deg) = mg   =>   N = mg / sqrt(2),

and the two vertical components ``N * cos(45 deg)`` sum back to ``mg``. The
sensor reports only the force projected onto its press axis (its face normal);
here the contact is essentially normal (the ball is centred by symmetry, so
shear is negligible), so each sensor should read ``~ mg / sqrt(2)``.

The scene uses ``SensorRig``, so ``--viewer`` and ``--gui`` work like they do
for the other sensor tests.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from sensor_rig import (
    GRAVITY,
    MOUNT_QUAT_XYZW,
    SensorRig,
    quat_mul_xyzw,
)

pytestmark = pytest.mark.sim

# --- V geometry ------------------------------------------------------------ #
# Each sensor is tilted 45 deg about world Y, so the two sensing faces meet at
# 90 deg. The apex (valley bottom) runs along world Y; the ball rolls in x-z.
TILT_DEG = 45.0
# Sensor body origins: symmetric about x=0, lifted clear of the ground. The
# offset is chosen so the two ~27 mm pad arrays leave a small centre gap (no
# body-body overlap) yet the gap stays well under the ball diameter.
SENSOR_X0 = 0.013
SENSOR_Z0 = 0.020

# Free ball that settles into the V.
BALL_RADIUS = 0.010
BALL_MASS = 0.050
BALL_MU = 1.0
BALL_START_Z = 0.032  # drop centred at x=0, just above the tangent rest height

# Settling.
MAX_FRAMES = 900
SETTLE_VEL = 2.0e-3      # ball linear speed [m/s] considered "at rest"
SETTLE_PATIENCE = 8      # consecutive calm frames required
FORCE_FLOOR = 1.0e-4     # below this a sensor is treated as "not in contact"
FORCE_TOL = 0.30         # generous: pad discreteness / settling perturb the ideal


def _tilt_quat(deg: float, sign: float) -> np.ndarray:
    """Sensor pose: ``deg`` about world Y (signed) on top of the upright mount.

    The mount brings the authored +Y-facing pads to face +Z; the Y-rotation then
    tilts that face by ``sign * deg`` so the two sensors form the V.
    """
    half = math.radians(sign * deg) / 2.0
    ry = np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float64)
    return quat_mul_xyzw(ry, MOUNT_QUAT_XYZW)


@pytest.fixture(scope="module")
def vshape_run(sim_config):
    """Build the two-sensor V, settle the ball, and return the per-sensor readout."""
    config = replace(
        sim_config,
        settle_vel=SETTLE_VEL,
        settle_patience=SETTLE_PATIENCE,
        max_steps=MAX_FRAMES,
    )
    rig = SensorRig(config, add_default_sensor=False)

    # Left "\" and right "/" sensors, tilted +/-45 deg to meet at 90 deg.
    rig.add_sensor(
        (-SENSOR_X0, 0.0, SENSOR_Z0),
        _tilt_quat(TILT_DEG, +1.0),
        label="L",
    )
    rig.add_sensor(
        (SENSOR_X0, 0.0, SENSOR_Z0),
        _tilt_quat(TILT_DEG, -1.0),
        label="R",
    )
    ball = rig.add_ball(
        mass=BALL_MASS,
        radius=BALL_RADIUS,
        start_z=BALL_START_Z,
        mu=BALL_MU,
    )
    rig.finalize()
    frames = rig.settle()

    sensor_l = rig.sensors["L"]
    sensor_r = rig.sensors["R"]

    fl = sensor_l.data.total_force.numpy().reshape(3)
    fr = sensor_r.data.total_force.numpy().reshape(3)
    return {
        "total_l": fl,
        "total_r": fr,
        "mag_l": float(np.linalg.norm(fl)),
        "mag_r": float(np.linalg.norm(fr)),
        "active_l": int((sensor_l.data.force.numpy() > FORCE_FLOOR).sum()),
        "active_r": int((sensor_r.data.force.numpy() > FORCE_FLOOR).sum()),
        "taxels_l": sensor_l.num_taxels,
        "taxels_r": sensor_r.num_taxels,
        "unique_taxels_l": len(set(sensor_l.taxel_names)),
        "unique_taxels_r": len(set(sensor_r.taxel_names)),
        "ball_z": float(rig.body_position(ball)[2]),
        "frames": frames,
    }


def test_both_sensors_read_independently(vshape_run):
    """Each sensor scopes to its own body's pads and reports a non-zero force."""
    r = vshape_run
    # Each sensor matched one complete pad set, without duplicates from the
    # other sensor.
    assert r["taxels_l"] == r["taxels_r"] > 0
    assert r["taxels_l"] == r["unique_taxels_l"]
    assert r["taxels_r"] == r["unique_taxels_r"]
    # The ball loads both sides of the V.
    assert r["active_l"] > 0, "left sensor read no contact"
    assert r["active_r"] > 0, "right sensor read no contact"


def test_each_sensor_reads_mg_over_sqrt2(vshape_run, artifact_csv):
    """Each face carries the normal force N = mg / sqrt(2) of a 90-degree V."""
    r = vshape_run
    mg = BALL_MASS * GRAVITY
    expected = mg / math.sqrt(2.0)

    fl, fr = r["total_l"], r["total_r"]
    vertical_sum = abs(float(fl[2])) + abs(float(fr[2]))

    artifact_csv(
        "two_sensors_vshape",
        [
            {"quantity": "left_normal_N", "value": r["mag_l"]},
            {"quantity": "right_normal_N", "value": r["mag_r"]},
            {"quantity": "expected_mg_over_sqrt2_N", "value": expected},
            {"quantity": "vertical_sum_N", "value": vertical_sum},
            {"quantity": "expected_mg_N", "value": mg},
            {"quantity": "ball_rest_z_m", "value": r["ball_z"]},
            {"quantity": "settle_frames", "value": r["frames"]},
        ],
    )

    print(
        "\nTwo-sensor V (\\o/):"
        f"\n  left  normal force = {r['mag_l']:.4f} N (active {r['active_l']}/{r['taxels_l']})"
        f"\n  right normal force = {r['mag_r']:.4f} N (active {r['active_r']}/{r['taxels_r']})"
        f"\n  expected per sensor mg/sqrt(2) = {expected:.4f} N"
        f"\n  vertical sum = {vertical_sum:.4f} N (expected mg = {mg:.4f} N)"
        f"\n  ball rest z = {r['ball_z']:.4f} m, settled in {r['frames']} frames"
    )

    if r["mag_l"] < FORCE_FLOOR or r["mag_r"] < FORCE_FLOOR:
        pytest.skip("ball did not settle in contact with both sensors")

    # Symmetry: the two faces share the load equally.
    assert r["mag_l"] == pytest.approx(r["mag_r"], rel=FORCE_TOL)
    # Each face reads mg / sqrt(2).
    assert r["mag_l"] == pytest.approx(expected, rel=FORCE_TOL)
    assert r["mag_r"] == pytest.approx(expected, rel=FORCE_TOL)
    # Vertical components add back up to the ball's weight.
    assert vertical_sum == pytest.approx(mg, rel=FORCE_TOL)
