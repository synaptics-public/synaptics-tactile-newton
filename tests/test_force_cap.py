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

"""Per-taxel force capping (saturation).

Each taxel's reading is clamped to ``[0, force_max]`` in the readout kernel.
``force_max`` affects only that clamp, never the contact dynamics, so the same
press yields identical raw axial forces regardless of the cap. This test exploits
that: press a probe into a central force area once with an effectively infinite
cap to read the raw peak ``R``, then repeat the identical press with the cap
below ``R`` and confirm the taxel saturates at exactly that cap.

A driven indenter (not a resting cube) is used because it concentrates the load
on a single force area, so one taxel reliably exceeds the per-taxel cap.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sensor_rig import SensorRigFactory, local_mm_to_world

pytestmark = pytest.mark.sim

PROBE_VELOCITY = -0.012   # 12 mm/s downward (matches the indentation test)
PROBE_RADIUS = 0.002
PROBE_CLEARANCE = 0.0005  # start the probe 0.5 mm above the force-area surface
NUM_FRAMES = 120
UNCAPPED = 1.0e9          # effectively no cap: taxel reports its raw axial force
CAP_FRACTION = 0.5        # set the binding cap to half the observed raw peak
# Defensive floor: if the press produces essentially no force the cap has nothing
# to bind against, so we skip rather than fail. (Not expected to trigger.)
FORCE_FLOOR = 1.0e-5


def _press_peak_taxel(config) -> float:
    """Press a probe straight down into the central force area; return the peak
    single-taxel force [N] over the press for the given ``config`` (its
    ``force_max`` sets the readout cap)."""
    factory = SensorRigFactory(config=config)

    # Read-only survey to locate the central force area to indent into.
    survey = factory().finalize(build_solver=False)
    centroids = survey.taxel_centroids
    if centroids is None:
        pytest.skip("taxel map exposes no centroids")
    world = np.array(
        [local_mm_to_world(c) for c in np.asarray(centroids, dtype=np.float64)]
    )
    centre_xy = world[:, :2].mean(axis=0)
    target = int(np.argmin(np.linalg.norm(world[:, :2] - centre_xy, axis=1)))
    tw = world[target]

    rig = factory()
    rig.add_probe(
        velocity_z=PROBE_VELOCITY,
        radius=PROBE_RADIUS,
        xy=(float(tw[0]), float(tw[1])),
        start_z=float(tw[2]) + PROBE_RADIUS + PROBE_CLEARANCE,
    )
    rig.finalize()
    peak = 0.0
    for _ in range(NUM_FRAMES):
        rig.step(1)
        peak = max(peak, float(rig.forces().max()))
    return peak


@pytest.fixture(scope="module")
def cap_run(sim_config):
    """Press once uncapped (raw peak), then identically with a binding cap."""
    raw_peak = _press_peak_taxel(replace(sim_config, force_max=UNCAPPED))
    cap = raw_peak * CAP_FRACTION
    capped_peak = _press_peak_taxel(replace(sim_config, force_max=cap))
    return {"raw_peak": raw_peak, "cap": cap, "capped_peak": capped_peak}


def test_taxel_saturates_at_force_max(cap_run, artifact_csv):
    raw_peak = cap_run["raw_peak"]
    cap = cap_run["cap"]
    capped_peak = cap_run["capped_peak"]

    artifact_csv(
        "force_cap",
        [
            {"run": "uncapped", "force_max_N": UNCAPPED, "peak_taxel_N": raw_peak},
            {"run": "capped", "force_max_N": cap, "peak_taxel_N": capped_peak},
        ],
    )

    print(
        "\nForce cap (per-taxel saturation):"
        f"\n  uncapped raw peak taxel = {raw_peak:.6f} N"
        f"\n  cap (force_max)         = {cap:.6f} N"
        f"\n  capped peak taxel       = {capped_peak:.6f} N"
    )

    if raw_peak < FORCE_FLOOR:
        pytest.skip(
            f"press produced essentially no force (raw peak={raw_peak:.3g}); "
            "nothing for the cap to bind against"
        )

    # The cap must actually bind: the uncapped reading exceeds it...
    assert raw_peak > cap, (raw_peak, cap)
    # ...and the capped reading pins to force_max exactly (the clamp clips it),
    # never exceeding it.
    assert capped_peak == pytest.approx(cap, rel=1.0e-4, abs=1.0e-9)
    assert capped_peak <= cap + 1.0e-9
