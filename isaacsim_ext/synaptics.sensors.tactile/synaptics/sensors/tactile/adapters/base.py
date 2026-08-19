# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Adapter boundary for the Isaac Sim Newton backend.

Everything version-specific about ``isaacsim.physics.newton`` lives behind this
contract. The rest of the extension consumes only:

* :meth:`NewtonBackendAdapter.is_ready` — is there a finalized model to bind to
* :meth:`NewtonBackendAdapter.model` / :meth:`state` / :meth:`contacts`
* :meth:`NewtonBackendAdapter.subscribe_physics_step`
* :meth:`NewtonBackendAdapter.contact_forces_available`

``isaacsim.physics.newton`` is pre-1.0 (0.6.0 in Isaac Sim 6.0.0-rc.22, 0.8.1 in
6.0.1) and its contact handling already changed once between those two builds.
This file is where that churn is absorbed.
"""

from importlib import import_module


class NewtonBackendAdapter:
    """Contract for one Isaac Sim version's Newton backend."""

    isaac_version = "unknown"
    extension_dependency = "isaacsim.physics.newton"
    backend_module_candidates: tuple[str, ...] = ()

    def __init__(self):
        self._backend = None

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """True when this Isaac Sim's Newton backend module can be imported."""
        try:
            self.backend()
            return True
        except ImportError:
            return False

    def backend(self):
        """Return the resolved backend module, or raise ``ImportError``."""
        if self._backend is not None:
            return self._backend

        import_errors = []
        for module_name in self.backend_module_candidates:
            try:
                self._backend = import_module(module_name)
                return self._backend
            except ImportError as error:
                import_errors.append(f"{module_name}: {error}")

        raise ImportError(
            f"Could not import the Newton physics backend for Isaac Sim "
            f"{self.isaac_version}. Tried: {'; '.join(import_errors)}"
        )

    # ------------------------------------------------------------------ #
    # Simulation surface
    # ------------------------------------------------------------------ #

    def newton_stage(self):
        """The backend's ``NewtonStage``, or None before it is constructed."""
        return self.backend().acquire_stage()

    def physics_interface(self):
        """The backend's physics interface, or None before it is constructed."""
        return self.backend().acquire_physics_interface()

    def active_physics_engine(self) -> str:
        """Name of the engine actually driving the stage ("newton", "physx", ...)."""
        return self.backend().get_active_physics_engine()

    def is_ready(self) -> bool:
        """True when a finalized Newton ``Model`` exists and can be bound to.

        Newton is initialized lazily inside the first ``step_sim()`` after Play,
        so this is False until the timeline is playing — and False again after
        Stop, which nulls the model.
        """
        stage = self.newton_stage()
        return bool(
            stage is not None
            and getattr(stage, "initialized", False)
            and not getattr(stage, "_init_failed", False)
            and getattr(stage, "model", None) is not None
        )

    def init_failed(self) -> bool:
        """True when USD parsing threw and the backend silently disabled physics.

        6.0.1 wraps ``initialize_newton()`` in try/except and latches a flag
        rather than raising, so a bad stage looks like "no signal" unless this
        is surfaced.
        """
        stage = self.newton_stage()
        return bool(stage is not None and getattr(stage, "_init_failed", False))

    def model(self):
        """The finalized ``newton.Model``, or None."""
        stage = self.newton_stage()
        return getattr(stage, "model", None) if stage is not None else None

    def state(self):
        """The current ``newton.State``.

        ``state_0``/``state_1`` are swapped every step — resolve this per step,
        never cache the returned object.
        """
        stage = self.newton_stage()
        return getattr(stage, "state_0", None) if stage is not None else None

    def contacts(self):
        """The persistent ``newton.Contacts`` (re-collided in place each step)."""
        stage = self.newton_stage()
        return getattr(stage, "contacts", None) if stage is not None else None

    def solver(self):
        stage = self.newton_stage()
        return getattr(stage, "solver", None) if stage is not None else None

    def solver_type(self) -> str:
        """Configured solver: "mujoco", "xpbd", or "unknown"."""
        stage = self.newton_stage()
        if stage is None:
            return "unknown"
        cfg = getattr(stage, "cfg", None)
        solver_cfg = getattr(cfg, "solver_cfg", None)
        solver_type = getattr(solver_cfg, "solver_type", None)
        if solver_type:
            return str(solver_type)
        solver = getattr(stage, "solver", None)
        if solver is None:
            return "unknown"
        return type(solver).__name__.removeprefix("Solver").lower()

    def physics_dt(self) -> float | None:
        stage = self.newton_stage()
        return getattr(stage, "sim_dt", None) if stage is not None else None

    def device(self) -> str | None:
        stage = self.newton_stage()
        return getattr(stage, "device_str", None) if stage is not None else None

    def contact_forces_available(self) -> tuple[bool, str]:
        """Can this configuration produce per-contact forces at all?

        ``contacts.force`` is only filled where the backend calls
        ``solver.update_contacts()``, and it does that on the **MuJoCo GPU path
        only**. XPBD implements ``update_contacts`` but the backend never calls
        it; MuJoCo-on-CPU is skipped by the same guard. In both cases the sensor
        would read a buffer of zeros, so we refuse to arm instead.

        Returns:
            ``(ok, reason)`` — ``reason`` is empty when ``ok`` is True.
        """
        solver_type = self.solver_type()
        if solver_type == "unknown":
            return False, "No Newton solver is active yet (press Play)."
        if solver_type != "mujoco":
            return False, (
                f"Solver is '{solver_type}'. Isaac Sim only calls "
                f"solver.update_contacts() on the MuJoCo path, so contact "
                f"forces would read as zero. Switch the Newton solver to MuJoCo."
            )
        if getattr(self.solver(), "use_mujoco_cpu", False):
            return False, (
                "Solver is MuJoCo on CPU (use_mujoco_cpu=True). Isaac Sim skips "
                "solver.update_contacts() on that path, so contact forces would "
                "read as zero. Use the MuJoCo GPU path."
            )
        return True, ""

    def contact_force_attribute_allocated(self) -> bool | None:
        """True when ``contacts.force`` exists on the live contact buffer.

        Isaac Sim 6.0.1 requests the ``force`` extended contact attribute itself
        (``model.request_contact_attributes("force")`` before the first
        ``collide()``), which is why this extension needs no allocation
        workaround. Returns None when there is nothing to inspect yet.
        """
        contacts = self.contacts()
        if contacts is None:
            return None
        return getattr(contacts, "force", None) is not None

    # ------------------------------------------------------------------ #
    # Step subscription
    # ------------------------------------------------------------------ #

    def subscribe_physics_step(self, callback) -> bool:
        """Register ``callback(dt)``, fired after each physics step.

        Callbacks run outside the CUDA graph, so Warp launches are safe there.
        """
        interface = self.physics_interface()
        if interface is None:
            return False
        interface.subscribe_physics_step_events(callback)
        return True

    def unsubscribe_physics_step(self, callback) -> bool:
        interface = self.physics_interface()
        if interface is None:
            return False
        try:
            interface.unsubscribe_physics_step_events(callback)
        except ValueError:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def describe(self) -> dict:
        """Flat record of the live backend state, for diagnostics and the panel."""
        record = {
            "adapter": type(self).__name__,
            "isaac_version": self.isaac_version,
            "extension_dependency": self.extension_dependency,
            "stage_constructed": self.newton_stage() is not None,
            "interface_constructed": self.physics_interface() is not None,
            "ready": self.is_ready(),
            "init_failed": self.init_failed(),
            "solver_type": self.solver_type(),
            "physics_dt": self.physics_dt(),
            "device": self.device(),
            "contact_force_attribute": self.contact_force_attribute_allocated(),
        }
        try:
            record["active_engine"] = self.active_physics_engine()
        except Exception as error:  # noqa: BLE001 - reporting must not raise.
            record["active_engine"] = f"error: {error}"
        forces_ok, reason = self.contact_forces_available()
        record["contact_forces_available"] = forces_ok
        record["contact_forces_reason"] = reason
        return record
