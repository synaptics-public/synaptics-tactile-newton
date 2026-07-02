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

"""Configuration for :class:`CTSSensorIsaacLab`.

A thin :class:`~isaaclab.sensors.SensorBaseCfg` subclass that carries the same
construction parameters as the core :class:`CTSSensor`, so the wrapper
can build one bound to the Isaac Lab scene's Newton model.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils.configclass import configclass

from .sensor import CTSSensorIsaacLab


def _taxel_marker(color: tuple[float, float, float]) -> sim_utils.SphereCfg:
    return sim_utils.SphereCfg(
        radius=0.0005,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


CTS_SENSOR_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/CTSSensor",
    markers={
        # Force ramp: cold (unloaded) -> hot (saturated). ``_debug_vis_callback``
        # selects a prototype per taxel by its force fraction, so more colors ==
        # a finer readout. Order matters (index 0 == lowest force).
        "force_0": _taxel_marker((0.0, 0.0, 1.0)),  # blue
        "force_1": _taxel_marker((0.0, 1.0, 1.0)),  # cyan
        "force_2": _taxel_marker((0.0, 1.0, 0.0)),  # green
        "force_3": _taxel_marker((1.0, 1.0, 0.0)),  # yellow
        "force_4": _taxel_marker((1.0, 0.5, 0.0)),  # orange
        "force_5": _taxel_marker((1.0, 0.0, 0.0)),  # red
    },
)
"""Default per-taxel debug markers: a blue->red force ramp of spheres."""


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

    force_max: float = 10.0
    """Per-taxel saturation force [N]."""

    sensing_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Body-local press axis; ignored when ``taxel_map`` is provided."""

    mount_rotation: tuple[float, float, float, float] | None = None
    """Optional (x, y, z, w) quaternion baked into the press axis at build time,
    for sensors statically fixed to the world. Leave ``None`` for a sensor on a
    moving body."""

    visualizer_cfg: VisualizationMarkersCfg = CTS_SENSOR_MARKER_CFG
    """Markers drawn per taxel when ``debug_vis`` is enabled: each taxel placed
    at its world position, colored by force fraction along the marker ramp.
    Requires a ``taxel_map`` (which supplies the positions)."""
