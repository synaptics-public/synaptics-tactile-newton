# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Configuration for :class:`CTSSensorIsaacLab`.

A thin :class:`~isaaclab.sensors.SensorBaseCfg` subclass that carries the same
construction parameters as the core :class:`CTSSensor`, so the wrapper
can build one bound to the Isaac Lab scene's Newton model.
"""

from __future__ import annotations

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils.configclass import configclass

from .sensor import CTSSensorIsaacLab


@configclass
class CTSSensorCfg(SensorBaseCfg):
    """Config for the Isaac Lab Synaptics CTS tactile sensor wrapper.

    Inherits the standard Isaac Lab sensor fields (``prim_path``,
    ``update_period``, ``history_length``, ``debug_vis``) from
    :class:`SensorBaseCfg`, and adds the Synaptics-specific parameters that are
    forwarded verbatim to :class:`CTSSensor`.
    """

    class_type: type = CTSSensorIsaacLab
    """The sensor class instantiated by the Isaac Lab scene from this config."""

    sensing_shape_pattern: str = "*/forceArea_*"
    """Glob over the USD hierarchy selecting the force-area collider shapes."""

    taxel_map: str | None = None
    """Path to the geometry splitter's ``<out>_taxel_map.json`` (press axis +
    per-taxel names/centroids). When set it OVERRIDES ``sensing_axis``."""

    force_max: float = 100.0
    """Per-taxel saturation force [N]."""

    sensing_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Body-local press axis; ignored when ``taxel_map`` is provided."""

    mount_rotation: tuple[float, float, float, float] | None = None
    """Optional (x, y, z, w) quaternion baked into the press axis at build time,
    for sensors statically fixed to the world. Leave ``None`` for a sensor on a
    moving body."""
