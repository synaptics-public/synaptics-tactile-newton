# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Synaptics Capacitive Tactile Sensor (CTS) extension for Isaac Sim.

Kit discovers :class:`SynapticsTactileSensorExtension` here. The import is
guarded so that ``python -m synaptics.sensors.tactile.diagnostics`` still works
from a python that has never booted Kit — the diagnostics CLI is the first
thing anyone runs when the extension will not load.
"""

try:
    from .extension import SynapticsTactileSensorExtension
except ModuleNotFoundError as error:
    if not str(error.name or "").startswith(("omni", "carb", "pxr")):
        raise
    SynapticsTactileSensorExtension = None

__all__ = ["SynapticsTactileSensorExtension"]
