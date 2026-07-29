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

"""Per-taxel world position (``CTSOutput.positions_w``).

Two independent checks on the world-frame taxel positions the sensor reports:

1. **Geometry cross-check** (``test_positions_w_matches_geometry``): the sensor
   derives ``positions_w`` from the Newton *shape transforms*; the harness's
   ``centroid_to_world`` derives the same points from the *taxel-map centroids*
   (a separate data path: the map's declared units -> metres, then mount
   rotation + lift). The two must agree, which catches a frame/units regression
   in either path — a wrong unit scale applied to the centroids, or a skipped
   placement transform in the world-position kernel.

2. **Motion tracking** (``test_positions_w_tracks_body_motion``): the rig's CTS
   is world-static (body ``-1``), so it never exercises the moving-body branch
   of the kernel. This drives the kernel directly with a known body pose to
   confirm a taxel on a moving body is transformed by that pose while a static
   taxel passes through unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensor_rig import DEFAULT_SENSOR_LABEL, SensorRigFactory, centroid_to_world

pytestmark = pytest.mark.sim

# positions_w (from shape transforms) must match the geometry-derived centroid
# (from the taxel map) to well under the ~1.5 mm force-area pitch.
_POS_TOL = 1.0e-4  # 0.1 mm


def _rig_with_positions(sim_config):
    """A finalized, stepped rig whose sensor has fresh ``positions_w``.

    A high, non-contacting weight is added only so the model has a joint (the
    MuJoCo solver requires one); it does not touch the sensor.
    """
    rig = SensorRigFactory(config=sim_config)()
    rig.add_weight(mass=0.01, height=0.5)
    rig.finalize()
    rig.step(3)
    return rig


def test_positions_w_shape(sim_config):
    rig = _rig_with_positions(sim_config)
    sensor = rig.sensors[DEFAULT_SENSOR_LABEL]
    pw = sensor.data.positions_w.numpy()
    assert pw.shape == (sensor.num_taxels, 3)
    assert np.isfinite(pw).all()


def test_positions_w_matches_geometry(sim_config):
    rig = _rig_with_positions(sim_config)
    sensor = rig.sensors[DEFAULT_SENSOR_LABEL]
    centroids = sensor.taxel_centroids
    if centroids is None:
        pytest.skip("taxel map exposes no centroids to cross-check against")

    pw = sensor.data.positions_w.numpy()
    gt = np.array([centroid_to_world(c) for c in np.asarray(centroids)])
    err = np.linalg.norm(pw - gt, axis=1)
    assert err.max() < _POS_TOL, (
        f"positions_w disagrees with the taxel-map geometry by up to "
        f"{err.max() * 1e3:.3f} mm (tol {_POS_TOL * 1e3:.3f} mm) — check the "
        f"world-position kernel's frame/units (mm vs m, missing placement xform)"
    )


def test_positions_w_tracks_body_motion():
    """The moving-body branch (body >= 0) must apply the body's world pose.

    Standalone kernel check (the rig's sensor is world-static, so it only covers
    the passthrough branch). Two taxels on a body translated + rotated 90 deg
    about Z, one taxel world-static.
    """
    wp = pytest.importorskip("warp")
    from synaptics_tactile_newton.kernels import _taxel_world_positions

    dev = "cpu"
    # Taxels 0,1 on body 0; taxel 2 is world-static (body -1).
    local_pos = wp.array(
        np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.05, 0.06, 0.07]], np.float32),
        dtype=wp.vec3,
        device=dev,
    )
    bodies = wp.array(np.array([0, 0, -1], np.int32), dtype=wp.int32, device=dev)
    # Body 0: translate (0.1, 0.2, 0.3), rotate +90 deg about Z (x,y,z,w).
    s = np.sin(np.pi / 4.0)
    c = np.cos(np.pi / 4.0)
    body_q = wp.array(
        np.array([[0.1, 0.2, 0.3, 0.0, 0.0, s, c]], np.float32),
        dtype=wp.transform,
        device=dev,
    )
    out = wp.zeros(3, dtype=wp.vec3, device=dev)
    wp.launch(
        _taxel_world_positions,
        dim=3,
        inputs=[local_pos, bodies, body_q, out],
        device=dev,
    )
    got = out.numpy()

    # Rz(90) maps (x, y, 0) -> (-y, x, 0), then add the translation.
    expected = np.array(
        [
            [0.1 + 0.0, 0.2 + 0.01, 0.3],  # (0.01,0,0) -> (0,0.01,0)
            [0.1 - 0.02, 0.2 + 0.0, 0.3],  # (0,0.02,0) -> (-0.02,0,0)
            [0.05, 0.06, 0.07],            # static: unchanged
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)
