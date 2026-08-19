# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

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
from isaaclab.sensors import SensorBase

from ..output import CTSOutput
from ..sensor import CTSSensor
from ..kernels import _scatter_env_force, _scatter_env_positions

if TYPE_CHECKING:
    from .sensor_cfg import CTSSensorCfg


class CTSSensorIsaacLab(SensorBase):
    """Isaac Lab ``SensorBase`` wrapper around :class:`CTSSensor`."""

    cfg: "CTSSensorCfg"

    def __init__(self, cfg: "CTSSensorCfg"):
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

        # This wrapper batches ONE physical sensor across the environments, so
        # the pattern must resolve to a single sensing body per env (the sensor
        # reads real per-body contact forces, and one env's taxels become one
        # output row). Reshaping the matched bodies into (num_envs, taxels_per_env)
        # and requiring each row to be a single body — with exactly one distinct
        # body per env — makes that a checked fact. It turns the otherwise-silent
        # case of a pattern spanning several bodies per env (e.g. two fingers
        # matched at once, whose taxels would be concatenated into each env row)
        # into a clear error: give each physical sensor its own config.
        taxel_bodies = self._sensor.taxel_bodies.reshape(n_env, self._taxels_per_env)
        single_body_per_env = bool(np.all(taxel_bodies == taxel_bodies[:, :1]))
        distinct_bodies = int(np.unique(taxel_bodies[taxel_bodies >= 0]).size)
        if not single_body_per_env or distinct_bodies not in (0, n_env):
            raise RuntimeError(
                f"Sensing pattern '{self.cfg.sensing_shape_pattern}' does not map "
                f"to exactly one sensing body per environment (matched "
                f"{distinct_bodies} sensing bodies across num_envs={n_env}). Each "
                f"CTSSensorCfg must scope a single physical sensor; give every "
                f"sensor its own config with a pattern that selects only its force "
                f"areas (e.g. '*/LeftFinger/forceArea_*' vs '*/RightFinger/forceArea_*')."
            )
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

    # -- Batching --------------------------------------------------------- #

    def _copy_core_into_batched_buffers(self, env_mask, core_data) -> None:
        """Copy each env's taxels from the core's flat output into its own row.

        The core produces every env's taxels in one flat array, laid out one env
        after another ("env-major")::

            [ env0 taxels | env1 taxels | env2 taxels | ... ]   len = num_envs * taxels_per_env

        This splits that into the batched buffers, one row per env. The copy runs
        in a GPU kernel so we never move data to the host — a host round-trip here
        would stall the sim every step. ``env_mask`` picks which rows to refresh.
        """
        n_env = self.num_instances
        wp.launch(
            _scatter_env_force,
            dim=(n_env, self._taxels_per_env),
            inputs=[
                core_data.force,
                env_mask,
                self._taxels_per_env,
                self._data.force,
            ],
        )

        # World-frame taxel positions, split the same env-major way as force.
        if self._data.positions_w is not None and core_data.positions_w is not None:
            wp.launch(
                _scatter_env_positions,
                dim=(n_env, self._taxels_per_env),
                inputs=[
                    core_data.positions_w,
                    env_mask,
                    self._taxels_per_env,
                    self._data.positions_w,
                ],
            )

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
