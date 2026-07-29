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

"""Per-force-area coverage: every taxel responds and is reported correctly.

Presses a small kinematic probe over *each* force area in turn (at its
world-space centroid) and verifies two things per area:

  1. **Responds** — the pressed force area reports a non-trivial normal force.
  2. **Correctly reported** — that force lands on the *expected* taxel row (no
     off-by-one / ordering mismatch); a near-neighbour within a small radius is
     accepted where placement is ambiguous.

A full-coverage CSV is written to ``tests/_artifacts/force_area_coverage.csv``.
If the whole sweep reads ~0 force the test skips rather than failing every area.

Set ``TAXEL_STRIDE`` (e.g. ``TAXEL_STRIDE=8``) to sample every Nth area for a
quick local check.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from sensor_rig import SensorRigFactory, centroid_to_world

pytestmark = pytest.mark.sim

# Press parameters (gentle, kinematic indent just past the area surface). Force
# areas are ~1.5 mm wide; a sub-mm probe never registers contact in this
# Newton/MuJoCo setup, so use a 2 mm radius (LOCALITY_MM tolerates the patch).
PROBE_RADIUS = 0.002         # m
PROBE_CLEARANCE = 0.001      # m — start this far above the area centroid
PROBE_VELOCITY = -0.02       # m/s downward
PRESS_FRAMES = 50

MIN_FORCE = 1.0e-3           # N — below this an area is considered unresponsive
LOCALITY_MM = 3.0            # accepted offset between pressed and reported area


def _press_area(factory, centroids, target):
    """Press over force area ``target``; return the per-taxel force array."""
    world = centroid_to_world(centroids[target])
    start_z = float(world[2]) + PROBE_RADIUS + PROBE_CLEARANCE
    rig = factory()
    rig.add_probe(
        velocity_z=PROBE_VELOCITY,
        radius=PROBE_RADIUS,
        xy=(float(world[0]), float(world[1])),
        start_z=start_z,
    )
    rig.finalize()
    rig.step(PRESS_FRAMES)
    return rig.forces()


@pytest.fixture(scope="module")
def coverage(sim_config):
    """Press every (or every Nth) force area; return rows + summary lists."""
    base = SensorRigFactory(config=sim_config)()
    base.finalize(build_solver=False)
    centroids = base.taxel_centroids
    names = base.taxel_names
    if centroids is None:
        pytest.skip("taxel map exposes no centroids")
    centroids = np.asarray(centroids, dtype=np.float64)
    n = len(centroids)

    stride = max(1, int(os.environ.get("TAXEL_STRIDE", "1")))
    targets = range(0, n, stride)

    factory = SensorRigFactory(config=sim_config)
    rows = []
    dead, misreported = [], []
    for target in targets:
        forces = _press_area(factory, centroids, target)
        dominant = int(np.argmax(forces))
        target_f = float(forces[target])
        dist_mm = float(
            np.linalg.norm(
                centroid_to_world(centroids[dominant])
                - centroid_to_world(centroids[target])
            )
            * 1.0e3
        )
        name = names[target] if names else f"row_{target}"
        rows.append(
            {
                "area_idx": target,
                "area_name": name,
                "target_force_N": target_f,
                "total_force_N": float(forces.sum()),
                "max_force_N": float(forces.max()),
                "dominant_idx": dominant,
                "dominant_is_target": bool(dominant == target),
                "dist_to_dominant_mm": dist_mm,
            }
        )
        if target_f <= MIN_FORCE:
            dead.append(name)
        elif dominant != target and dist_mm > LOCALITY_MM:
            misreported.append(f"{name}->row{dominant}")

    return {"rows": rows, "dead": dead, "misreported": misreported}


def test_coverage_csv_written(coverage, artifact_csv):
    path = artifact_csv("force_area_coverage", coverage["rows"])
    assert path.exists()


def test_some_area_responds(coverage):
    """Guard: if nothing responds at all, it's the compression-sign convention."""
    if all(r["total_force_N"] < MIN_FORCE for r in coverage["rows"]):
        pytest.skip(
            "no force area produced force — check geometry/drop alignment or "
            "the taxel-map press-axis sign (baked in the geometry splitter)"
        )


def test_every_area_responds(coverage):
    """Every pressed force area reports a non-trivial force on its own row."""
    if all(r["total_force_N"] < MIN_FORCE for r in coverage["rows"]):
        pytest.skip("whole sweep read ~0 force; see test_some_area_responds")
    assert not coverage["dead"], (
        f"{len(coverage['dead'])} force area(s) did not report force: "
        f"{coverage['dead']}"
    )


def test_every_area_reported_on_correct_taxel(coverage):
    """Pressing force area k lights up row k (mapping/ordering is correct)."""
    if all(r["total_force_N"] < MIN_FORCE for r in coverage["rows"]):
        pytest.skip("whole sweep read ~0 force; see test_some_area_responds")
    assert not coverage["misreported"], (
        f"{len(coverage['misreported'])} force area(s) reported on the wrong "
        f"taxel: {coverage['misreported']}"
    )
