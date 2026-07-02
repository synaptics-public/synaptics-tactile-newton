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

"""Isaac Lab ``SensorBase`` wrapper exposing the Synaptics CTS.

A thin wrapper: all contact-to-signal physics stays in the core
:class:`CTSSensor`. This class only plugs into Isaac Lab's lazy sensor
lifecycle, feeds the scene's live Newton ``state``/``contacts`` into
:meth:`CTSSensor.update`, and exposes the :class:`CTSOutput` tensors batched
across ``num_envs``.

Newton backend only (no PhysX path). The ``_resolve_newton_*`` hooks reach the
scene's model/state/contacts via ``isaaclab_newton``'s ``NewtonManager``
singleton, so the sim must be created with ``SimulationCfg(physics=NewtonCfg())``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import SensorBase

from ..output import CTSOutput
from ..sensor import CTSSensor

if TYPE_CHECKING:
    from .sensor_cfg import CTSSensorCfg


class CTSSensorIsaacLab(SensorBase):
    """Isaac Lab ``SensorBase`` wrapper around :class:`CTSSensor`."""

    cfg: "CTSSensorCfg"

    def __init__(self, cfg: "CTSSensorCfg"):
        # The visualizer must exist before SensorBase.__init__, which calls
        # set_debug_vis() while wiring up debug vis.
        self._tactile_visualizer: VisualizationMarkers | None = None
        super().__init__(cfg)
        # Heavy construction is deferred to _initialize_impl (the SensorBase
        # contract): at __init__ time the sim/model do not yet exist.
        self._sensor: CTSSensor | None = None
        self._data: CTSOutput | None = None
        self._taxels_per_env: int = 0

    # -- SensorBase lifecycle --------------------------------------------- #

    def _initialize_impl(self) -> None:
        super()._initialize_impl()

        # Tell the Newton backend to compute contact forces each step (the
        # manager only populates ``contacts.force`` when this flag is set).
        self._newton_manager()._report_contacts = True

        # Build ONE core sensor bound to the scene's finalized Newton model. This
        # runs on PHYSICS_READY, BEFORE the backend allocates its ``Contacts``
        # buffer, so the core sensor's ``SensorContact`` registers its ``"force"``
        # request in time to be included in that allocation.
        model = self._resolve_newton_model()
        self._sensor = CTSSensor(
            model,
            sensing_shape_pattern=self.cfg.sensing_shape_pattern,
            taxel_map=self.cfg.taxel_map,
            sensing_axis=self.cfg.sensing_axis,
            force_max=self.cfg.force_max,
            mount_rotation=self.cfg.mount_rotation,
        )

        # Allocate batched output buffers: (num_envs, taxels_per_env) and
        # (num_envs, 3). The single core ``SensorContact`` matches the force
        # areas in EVERY environment at once, so its flat output has one row per
        # matched shape across all envs, ordered env-major. We split that back
        # into per-env rows here, which requires every env to expose the same
        # number of taxels.
        n_env = self.num_instances
        core_taxels = self._sensor.num_taxels
        if n_env <= 0 or core_taxels % n_env != 0:
            raise RuntimeError(
                f"Sensing pattern '{self.cfg.sensing_shape_pattern}' matched "
                f"{core_taxels} shapes, which is not a whole multiple of "
                f"num_envs={n_env}. The pattern must match the same force areas "
                f"in every environment (and only collision shapes \u2014 e.g. exclude "
                f"'*_visual' twins)."
            )
        self._taxels_per_env = core_taxels // n_env
        device = model.device
        self._data = CTSOutput(
            force=wp.zeros((n_env, self._taxels_per_env), dtype=wp.float32, device=device),
            total_force=wp.zeros((n_env, 3), dtype=wp.float32, device=device),
            positions_w=wp.zeros(
                (n_env, self._taxels_per_env, 3), dtype=wp.float32, device=device
            ),
        )

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        # Run the core contact-to-signal step (all envs in one launch), then copy
        # the per-env results into the batched buffers. ``env_mask`` is a
        # (num_envs,) warp bool array: True == update this env.
        state, contacts = self._resolve_newton_state_contacts()
        self._sensor.update(state, contacts)
        self._copy_core_into_batched_buffers(env_mask, self._sensor.data)
        # Net per-env force vector: the core's taxels are laid out env-major, so
        # a group == one env's taxels.
        self._sensor.reduce_total_force(
            state, self._taxels_per_env, self._data.total_force
        )

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        super().reset(env_ids, env_mask)
        if self._data is None:
            return
        # Zero the selected env rows. Mirror SensorBase's index/mask resolution.
        mask = self._resolve_indices_and_mask(env_ids, env_mask).numpy()
        rows = mask.nonzero()[0]
        if rows.size == 0:
            return
        force_np = self._data.force.numpy()
        total_np = self._data.total_force.numpy()
        force_np[rows] = 0.0
        total_np[rows] = 0.0
        self._data.force.assign(force_np)
        self._data.total_force.assign(total_np)

    # -- Sensor output ----------------------------------------------------- #

    @property
    def data(self) -> CTSOutput:
        """Batched Synaptics tactile output (GPU tensors).

        Triggers the lazy SensorBase update so callers always read a fresh
        signal at the configured ``update_period``.
        """
        self._update_outdated_buffers()
        return self._data

    # -- Debug visualization ---------------------------------------------- #

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Create/show or hide the per-taxel markers (SensorBase debug-vis hook)."""
        if debug_vis:
            if self._tactile_visualizer is None:
                self._tactile_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self._tactile_visualizer.set_visibility(True)
        elif self._tactile_visualizer is not None:
            self._tactile_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Draw one marker per taxel at its world position, colored by force.

        Called each render frame while debug vis is on; no-ops without taxel
        positions to place.
        """
        if self._tactile_visualizer is None or self._sensor is None or self._data is None:
            return
        state, _ = self._resolve_newton_state_contacts()
        positions = self._sensor.taxel_positions_world(state)
        if positions is None or positions.shape[0] == 0:
            return

        # Force per taxel, same env-major order as ``positions``.
        forces = self._data.force.numpy().reshape(-1)
        frac = np.clip(forces / max(float(self.cfg.force_max), 1e-9), 0.0, 1.0)

        # Color prototype per taxel by force fraction; grow the marker with force.
        n_colors = len(self.cfg.visualizer_cfg.markers)
        marker_indices = np.minimum((frac * n_colors).astype(np.int64), n_colors - 1)
        scales = np.ones((positions.shape[0], 3), dtype=np.float32) * (1.0 + 2.0 * frac[:, None])

        self._tactile_visualizer.visualize(
            translations=positions.astype(np.float32),
            marker_indices=marker_indices,
            scales=scales,
        )

    # -- Batching --------------------------------------------------------- #

    def _copy_core_into_batched_buffers(self, env_mask, core_data) -> None:
        """Split the core's flat env-major output into per-env rows.

        ``core_data.force`` is a flat ``(num_envs * taxels_per_env,)`` array;
        reshaping recovers one row per env, copied into the batched buffer for
        the envs flagged in ``env_mask``.
        """
        rows = env_mask.numpy().nonzero()[0]
        if rows.size == 0:
            return
        n_env = self.num_instances
        src_force = core_data.force.numpy().reshape(n_env, self._taxels_per_env)
        force_np = self._data.force.numpy()
        force_np[rows] = src_force[rows]
        self._data.force.assign(force_np)

        # World-frame taxel positions, split the same env-major way as force.
        if self._data.positions_w is not None and core_data.positions_w is not None:
            src_pos = core_data.positions_w.numpy().reshape(
                n_env, self._taxels_per_env, 3
            )
            pos_np = self._data.positions_w.numpy()
            pos_np[rows] = src_pos[rows]
            self._data.positions_w.assign(pos_np)

    # -- Newton backend access (Isaac Lab 3.x NewtonManager) -------------- #

    @staticmethod
    def _newton_manager():
        """Return the Isaac Lab Newton physics manager singleton.

        Errors if the sim is not on the Newton backend (this wrapper has no
        PhysX path); select Newton via ``SimulationCfg(physics=NewtonCfg())``.
        """
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError as exc:  # pragma: no cover - depends on Isaac Lab extras
            raise RuntimeError(
                "The Synaptics tactile sensor requires the Isaac Lab Newton "
                "backend (isaaclab_newton), which is not importable."
            ) from exc
        if NewtonManager.get_model() is None:
            raise RuntimeError(
                "Newton backend is not active. Construct the simulation with "
                "SimulationCfg(physics=NewtonCfg()) so the Newton model exists."
            )
        return NewtonManager

    def _resolve_newton_model(self):
        """Return the scene's finalized Newton ``Model``."""
        return self._newton_manager().get_model()

    def _resolve_newton_state_contacts(self):
        """Return the scene's current Newton ``(state, contacts)``.

        ``state`` is the live post-step state; ``contacts`` is the backend's
        shared ``Contacts`` buffer, whose forces are refreshed each step because
        ``_initialize_impl`` set ``_report_contacts = True``.
        """
        mgr = self._newton_manager()
        return mgr.get_state_0(), mgr._contacts
