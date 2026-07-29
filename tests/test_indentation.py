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

"""Controlled indentation force-displacement curve.

Drive a kinematic probe down into the sensor at constant velocity and record
total force vs. probe displacement. The resulting force-displacement curve is the
primary calibration artifact (written to ``tests/_artifacts/indentation.csv``).

As an automated check we assert the curve is well-formed: force starts at ~0,
rises once the probe makes contact, and is non-trivial by the end of travel.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensor_rig import SensorRigFactory, centroid_to_world

pytestmark = pytest.mark.sim

PROBE_VELOCITY = -0.012   # 12 mm/s downward (controlled indenter)
PROBE_RADIUS = 0.002
PROBE_CLEARANCE = 0.0005   # start the probe 0.5 mm above the force-area surface
NUM_FRAMES = 120


@pytest.fixture(scope="module")
def indentation_curve(sim_config):
    """Run one indentation sweep; return list of (frame, probe_z, total_force).

    The probe is placed directly above a central force area (just clear of the
    surface) and driven straight down at constant velocity, so it indents the
    sensor progressively rather than free-falling and bouncing off it.
    """
    factory = SensorRigFactory(config=sim_config)

    # Locate a central force area to indent into: build a read-only rig to read
    # the taxel centroids, then target the one nearest the lateral centre so the
    # probe stays clear of the housing edges.
    survey = factory().finalize(build_solver=False)
    centroids = survey.taxel_centroids
    if centroids is None:
        pytest.skip("taxel map exposes no centroids")
    world = np.array(
        [centroid_to_world(c) for c in np.asarray(centroids, dtype=np.float64)]
    )
    centre_xy = world[:, :2].mean(axis=0)
    target = int(np.argmin(np.linalg.norm(world[:, :2] - centre_xy, axis=1)))
    target_world = world[target]

    rig = factory()
    probe = rig.add_probe(
        velocity_z=PROBE_VELOCITY,
        radius=PROBE_RADIUS,
        xy=(float(target_world[0]), float(target_world[1])),
        start_z=float(target_world[2]) + PROBE_RADIUS + PROBE_CLEARANCE,
        height=0.02,
    )
    rig.finalize()
    z0 = rig.body_position(probe)[2]
    rows = []
    for frame in range(NUM_FRAMES):
        rig.step(1)
        z = rig.body_position(probe)[2]
        rows.append(
            {
                "frame": frame,
                "probe_z_m": float(z),
                "displacement_m": float(z0 - z),
                "total_force_N": rig.total_force(),
                "active_taxels": rig.active_count(),
            }
        )
    return rows


def test_curve_written(indentation_curve, artifact_csv):
    path = artifact_csv("indentation", indentation_curve)
    assert path.exists()


def test_starts_unloaded(indentation_curve):
    assert indentation_curve[0]["total_force_N"] == pytest.approx(0.0, abs=1.0e-3)


def test_force_rises_with_indentation(indentation_curve):
    """Force at the end of travel should exceed the (near-zero) start."""
    start = indentation_curve[0]["total_force_N"]
    end = max(r["total_force_N"] for r in indentation_curve)
    assert end > start + 1.0e-3, f"no force rise: start={start}, peak={end}"


def test_force_monotone_trend(indentation_curve):
    """Coarsely, later-half mean force should exceed first-half mean force."""
    n = len(indentation_curve)
    first = [r["total_force_N"] for r in indentation_curve[: n // 2]]
    second = [r["total_force_N"] for r in indentation_curve[n // 2 :]]
    assert sum(second) / len(second) >= sum(first) / len(first)
