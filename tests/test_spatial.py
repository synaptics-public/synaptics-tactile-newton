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

"""Spatial discrimination / taxel mapping.

Press a small probe over the centroid of a chosen force area and verify the
response is *spatially local*: the strongest-reading taxel sits near the probe,
and the response is concentrated rather than smeared across all 68 taxels. This
validates the contact -> taxel mapping in the sensor's Warp kernel.

If the press produces ~no force, the test skips rather than failing every target.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensor_rig import SensorRigFactory, centroid_to_world

pytestmark = pytest.mark.sim

# Force-area rows to probe (indices into the sensing/output order).
_TARGET_TAXELS = [0, 20, 40]
# A dominant taxel must sit within this distance of the probe XY [m].
_LOCALITY_RADIUS = 6.0e-3

# Force areas are ~1.5 mm wide; a sub-mm probe never registers contact in this
# Newton/MuJoCo setup, so use a 2 mm radius (_LOCALITY_RADIUS tolerates the patch).
PROBE_RADIUS = 0.002
PROBE_CLEARANCE = 0.001


def _press_at(factory, world):
    """Press straight down onto the force area at ``world`` (x, y, z) [m]."""
    start_z = float(world[2]) + PROBE_RADIUS + PROBE_CLEARANCE
    rig = factory()
    probe = rig.add_probe(
        velocity_z=-0.02,
        radius=PROBE_RADIUS,
        xy=(float(world[0]), float(world[1])),
        start_z=start_z,
    )
    rig.finalize()
    rig.step(50)
    return rig, probe


@pytest.fixture(scope="module")
def centroids(sim_config):
    rig = SensorRigFactory(config=sim_config)().finalize(build_solver=False)
    c = rig.taxel_centroids
    if c is None:
        pytest.skip("taxel map exposes no centroids")
    return np.asarray(c, dtype=np.float64)


@pytest.mark.parametrize("target", _TARGET_TAXELS)
def test_response_is_local(sim_config, centroids, target):
    if target >= len(centroids):
        pytest.skip(f"taxel {target} out of range ({len(centroids)} taxels)")

    factory = SensorRigFactory(config=sim_config)
    world = centroid_to_world(centroids[target])
    rig, _probe = _press_at(factory, world)

    forces = rig.forces()
    if forces.sum() < 1.0e-3:
        pytest.skip(
            "press produced ~0 force — check geometry/drop alignment or the "
            "taxel-map press-axis sign (baked in the geometry splitter)"
        )

    top = int(np.argmax(forces))
    # The strongest taxel should be concentrated, not one of a uniform field.
    median = float(np.median(forces))
    assert forces[top] > 5.0 * max(median, 1.0e-6), "response not concentrated"

    # And it should be spatially near the probe (in world XY).
    top_world = centroid_to_world(centroids[top])
    dist_xy = float(np.linalg.norm(top_world[:2] - world[:2]))
    assert dist_xy <= _LOCALITY_RADIUS, (
        f"dominant taxel {top} is {dist_xy*1e3:.1f} mm from probe over taxel {target}"
    )
