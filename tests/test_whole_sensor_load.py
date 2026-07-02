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

"""Whole-sensor load: total reported force vs. the applied weight.

Rest a flat plate of known mass across the *entire* sensor, let it settle, and
compare the summed taxel force to the plate's weight (m*g). Unlike the per-area
checks this loads all force areas at once, so the sum should approach the full
weight if every area contributes correctly.

By request this test does **not** assert a tolerance — it prints the comparison
and emits a warning when the discrepancy is large, so regressions are visible
without turning the suite red.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sensor_rig import GRAVITY, SENSOR_Z, SensorRigFactory, local_mm_to_world

pytestmark = pytest.mark.sim

PLATE_MASS = 0.05            # kg
PLATE_HALF_THICKNESS = 0.001  # m (2 mm plate)
PLATE_MARGIN = 0.002        # m overhang beyond the sensor footprint
DROP_GAP = 0.005            # m start clearance above the force-area tips
WARN_FRACTION = 0.25        # warn if |reported - mg| / mg exceeds this


@pytest.fixture(scope="module")
def whole_sensor_load(sim_config):
    # Footprint of the force areas in world XY (and their top in Z).
    base = SensorRigFactory(config=sim_config)().finalize(build_solver=False)
    centroids = base.taxel_centroids
    if centroids is None:
        pytest.skip("taxel map exposes no centroids")
    world = np.array([local_mm_to_world(c) for c in centroids], dtype=np.float64)
    cx, cy = world[:, 0].mean(), world[:, 1].mean()
    hx = (world[:, 0].max() - world[:, 0].min()) / 2.0 + PLATE_MARGIN
    hy = (world[:, 1].max() - world[:, 1].min()) / 2.0 + PLATE_MARGIN
    tip_z = world[:, 2].max()

    rig = SensorRigFactory(config=sim_config)()
    height = tip_z + PLATE_HALF_THICKNESS + DROP_GAP - SENSOR_Z
    rig.add_box(
        mass=PLATE_MASS,
        half_extents=(hx, hy, PLATE_HALF_THICKNESS),
        xy=(float(cx), float(cy)),
        height=float(height),
        label="plate",
    )
    rig.finalize()
    rig.settle()
    return {
        "reported_N": rig.total_force(),
        "active": rig.active_count(),
        "num_taxels": rig.num_taxels,
    }


def test_whole_sensor_force_vs_weight(whole_sensor_load):
    reported = whole_sensor_load["reported_N"]
    expected = PLATE_MASS * GRAVITY
    diff = reported - expected
    frac = abs(diff) / expected

    print(
        "\nWhole-sensor load:"
        f"\n  applied weight  m*g = {expected:.4f} N (m={PLATE_MASS} kg)"
        f"\n  reported total force = {reported:.4f} N"
        f"\n  difference          = {diff:+.4f} N ({frac * 100:.1f}% of weight)"
        f"\n  active taxels       = {whole_sensor_load['active']}"
        f"/{whole_sensor_load['num_taxels']}"
    )

    if frac > WARN_FRACTION:
        warnings.warn(
            f"whole-sensor force off by {frac * 100:.1f}% "
            f"(reported {reported:.4f} N vs weight {expected:.4f} N)",
            stacklevel=2,
        )
    # Intentionally no assertion on the magnitude (see module docstring).
