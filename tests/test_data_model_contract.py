# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Enforce the ``CTSSensor`` data-model contract.

``data`` is the only device-resident (GPU) surface; every ``taxel_*`` property
is static host metadata, resolved once and re-read without a device copy. These
checks fail if a property regresses to a per-access GPU->CPU transfer or returns
the wrong side (host vs device).
"""

from __future__ import annotations

import numpy as np
import pytest

from sensor_rig import DEFAULT_SENSOR_LABEL, SensorRigFactory

pytestmark = pytest.mark.sim


def _read_only_sensor(sim_config):
    """A finalized, unstepped rig exposing only its sensor (no solver needed)."""
    rig = SensorRigFactory(config=sim_config)()
    rig.finalize(build_solver=False)
    return rig.sensors[DEFAULT_SENSOR_LABEL]


def test_data_is_device_resident(sim_config):
    import warp as wp

    data = _read_only_sensor(sim_config).data
    assert isinstance(data.force, wp.array)
    assert isinstance(data.total_force, wp.array)
    assert isinstance(data.positions_w, wp.array)


def test_taxel_metadata_is_host(sim_config):
    sensor = _read_only_sensor(sim_config)
    assert isinstance(sensor.taxel_bodies, np.ndarray)
    assert isinstance(sensor.taxel_centroids, np.ndarray)
    assert isinstance(sensor.taxel_names, list)


def test_taxel_bodies_is_cached_not_recopied(sim_config):
    """Repeated reads return the same host object (no per-access device copy)."""
    sensor = _read_only_sensor(sim_config)
    assert sensor.taxel_bodies is sensor.taxel_bodies
