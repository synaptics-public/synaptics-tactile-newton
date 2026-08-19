# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""The adapter: binds ``CTSSensor`` to Isaac Sim's live Newton simulation.

Owns *when* to build and tear down a sensor and *where* the state comes from —
no signal maths.

Lifecycle, forced on us by the backend:

* Newton builds its model **lazily, inside the first step after Play**, so a
  sensor cannot be constructed on Play — only once a model exists.
* ``state_0``/``state_1`` are **swapped every step**, so the state is resolved
  per step and never cached.
* **Stop destroys the model**, so every sensor must be rebuilt on the next Play.
"""

import fnmatch

import carb
import numpy as np
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom

from .adapters import get_newton_adapter
from .config import DEFAULT_FORCE_MAX_N, load_model_config, resolve_model_assets
from .spawn import find_sensor_prims, read_metadata

_LOG_PREFIX = "[Synaptics Tactile]"

#: The runtime owned by the running extension, if any.
_active_runtime = None


def set_active_runtime(runtime) -> None:
    """Publish (or clear) the extension's runtime."""
    global _active_runtime
    _active_runtime = runtime


def get_active_runtime():
    """The extension's runtime, or None outside a running extension.

    Test harnesses should prefer this over constructing their own: two runtimes
    both bind to the same model and each builds its own ``CTSSensor``, doubling
    the per-step GPU work for no benefit.
    """
    return _active_runtime


class SensorBinding:
    """One spawned sensor prim, bound to the live Newton model."""

    def __init__(self, prim_path: str, sensor, config: dict):
        self.prim_path = prim_path
        self.sensor = sensor
        self.config = config
        self.latest_forces = None       # (num_taxels,) host copy, newtons
        self.latest_total_force = 0.0   # magnitude of the net force vector [N]

    @property
    def num_taxels(self) -> int:
        return int(self.sensor.num_taxels)

    def read_back(self) -> None:
        """Copy this step's output to the host.

        A device->host copy per step is fine for a GUI readout of 52 values and
        keeps the panel simple. Anything batched (Isaac Lab) should read
        ``sensor.data`` on the device instead.
        """
        data = self.sensor.data
        self.latest_forces = data.force.numpy()
        self.latest_total_force = float(np.linalg.norm(data.total_force.numpy()))


class TactileRuntime:
    """Owns every bound sensor for the current Play session."""

    def __init__(self, adapter=None):
        self._adapter = adapter
        self._bindings: list[SensorBinding] = []
        self._timeline_sub = None
        self._step_subscribed = False
        self._bind_attempted = False
        self._bound_model = None
        self._last_error = ""
        self._steps = 0

    # ------------------------------------------------------------------ #
    # Extension lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Begin listening for Play/Stop."""
        timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = (
            timeline.get_timeline_event_stream().create_subscription_to_pop(
                self._on_timeline_event, name="synaptics.tactile.timeline"
            )
        )

    def stop(self) -> None:
        """Drop every subscription and binding. Safe to call twice."""
        self._unbind()
        self._unsubscribe_step()
        self._timeline_sub = None

    # ------------------------------------------------------------------ #
    # State for the panel
    # ------------------------------------------------------------------ #

    @property
    def bindings(self) -> list:
        return list(self._bindings)

    @property
    def is_bound(self) -> bool:
        return bool(self._bindings)

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def step_count(self) -> int:
        return self._steps

    # ------------------------------------------------------------------ #
    # Timeline
    # ------------------------------------------------------------------ #

    def _on_timeline_event(self, event) -> None:
        event_type = event.type
        if event_type == int(omni.timeline.TimelineEventType.PLAY):
            self._on_play()
        elif event_type == int(omni.timeline.TimelineEventType.STOP):
            self._on_stop()

    def _on_play(self) -> None:
        self._bind_attempted = False
        self._last_error = ""
        # Only a fresh Play restarts the count. Stepping the timeline frame by
        # frame (``forward_one_frame``) emits a PLAY event per frame, so
        # resetting unconditionally would peg the counter at one frame's worth
        # of steps and hide exactly the detail someone stepping is looking for.
        if not self._bindings:
            self._steps = 0
        # The model does not exist yet — Newton builds it inside the first
        # step. Subscribe now and bind on the first step that has one.
        self._subscribe_step()

    def _on_stop(self) -> None:
        self._unbind()
        self._unsubscribe_step()

    # ------------------------------------------------------------------ #
    # Physics step
    # ------------------------------------------------------------------ #

    def _subscribe_step(self) -> None:
        if self._step_subscribed:
            return
        if self._adapter is None:
            try:
                self._adapter = get_newton_adapter()
            except ImportError as error:
                self._last_error = str(error)
                return
        if self._adapter.subscribe_physics_step(self._on_physics_step):
            self._step_subscribed = True

    def _unsubscribe_step(self) -> None:
        if self._step_subscribed and self._adapter is not None:
            self._adapter.unsubscribe_physics_step(self._on_physics_step)
        self._step_subscribed = False

    def _on_physics_step(self, dt: float) -> None:
        """Fires after each physics step, outside the CUDA graph."""
        adapter = self._adapter
        if adapter is None or not adapter.is_ready():
            return

        # Newton rebuilds its model whenever the stage changes, not only on
        # Play, and the old Model/Contacts are destroyed with it. Binding once
        # per Play would leave us holding a sensor built against freed buffers —
        # still reporting numbers, silently stale. Rebind when the model
        # identity changes.
        model = adapter.model()
        if self._bindings and model is not self._bound_model:
            carb.log_info(f"{_LOG_PREFIX} Newton model changed; rebinding sensors.")
            self._unbind()

        if not self._bindings and not self._bind_attempted:
            # False means "try again next step": the solver is often not
            # populated on the very first one, and latching then would leave the
            # sensor dead for the whole Play session.
            self._bind_attempted = self._bind()

        if not self._bindings:
            return

        # state_0/state_1 swap every step: resolve, never cache.
        state = adapter.state()
        contacts = adapter.contacts()
        if state is None or contacts is None:
            return

        for binding in self._bindings:
            binding.sensor.update(state, contacts)
        self._steps += 1

    # ------------------------------------------------------------------ #
    # Binding
    # ------------------------------------------------------------------ #

    def _bind(self) -> bool:
        """Build one ``CTSSensor`` per tagged prim against the live model.

        Returns False when the failure is transient and worth retrying.
        """
        adapter = self._adapter
        ok, reason = adapter.contact_forces_available()
        if not ok:
            self._last_error = reason
            carb.log_error(f"{_LOG_PREFIX} refusing to arm: {reason}")
            return False

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False

        try:
            from synaptics_tactile_newton import CTSSensor
        except ImportError as error:
            self._last_error = (
                f"synaptics_tactile_newton is not importable inside Kit ({error}). "
                "Install it into Isaac Sim's python, or launch with "
                "--/app/python/extraPaths/0=<repo>."
            )
            carb.log_error(f"{_LOG_PREFIX} {self._last_error}")
            return False

        model = adapter.model()
        prims = find_sensor_prims(stage)
        if not prims:
            self._last_error = "No Synaptics sensor prims on this stage."
            return False

        bound = []
        for prim in prims:
            prim_path = str(prim.GetPath())
            config = read_metadata(prim) or {}
            pattern = self._scoped_pattern(prim_path, config)

            matched = self._pattern_matches(model, pattern)
            if not matched:
                # Silent here would look exactly like "the sensor reads zero",
                # so say which glob missed and what the labels actually are.
                sample = [
                    str(label)
                    for label in model.shape_label
                    if prim_path in str(label)
                ][:3]
                self._last_error = (
                    f"No Newton shapes matched '{pattern}' for {prim_path}. "
                    f"Shape labels under that prim look like: {sample or '(none)'}"
                )
                carb.log_error(f"{_LOG_PREFIX} {self._last_error}")
                continue

            try:
                sensor = CTSSensor(
                    model,
                    sensing_shape_pattern=pattern,
                    taxel_map=self._taxel_map_path(config),
                    mount_rotation=self._mount_rotation(prim),
                    force_max=float(config.get("forceMaxN", DEFAULT_FORCE_MAX_N)),
                )
            except Exception as error:  # noqa: BLE001 - one bad prim must not kill the rest.
                carb.log_error(f"{_LOG_PREFIX} could not bind {prim_path}: {error}")
                self._last_error = f"{prim_path}: {error}"
                continue

            bound.append(SensorBinding(prim_path, sensor, dict(config)))
            carb.log_info(
                f"{_LOG_PREFIX} bound {prim_path}: {sensor.num_taxels} taxels"
            )

        self._bindings = bound
        if bound:
            self._bound_model = model
            self._last_error = ""
        elif not self._last_error:
            self._last_error = (
                f"Found {len(prims)} sensor prim(s) but bound none — see the console."
            )
        return True

    @staticmethod
    def _taxel_map_path(config: dict) -> str:
        """Where this sensor's taxel map lives *on this machine*.

        Spawning records an absolute path, which is wrong the moment the stage
        is opened anywhere else, so resolve from the model name first and only
        fall back to the recorded path.
        """
        model_name = config.get("model")
        if model_name:
            try:
                _, taxel_map = resolve_model_assets(load_model_config(None, str(model_name)))
                return str(taxel_map)
            except (FileNotFoundError, ValueError):
                pass
        return str(config["taxelMap"])

    @staticmethod
    def _scoped_pattern(prim_path: str, config: dict) -> str:
        """Scope the model's shape glob to one sensor prim.

        Newton labels shapes with their full USD prim path, so prefixing with
        the sensor's own path is what stops two sensors on one stage from each
        claiming the other's taxels.

        The profile's pattern is a *leaf* glob (``*/forceArea_NNN``); the asset
        nests its force areas under intermediate scopes
        (``<sensor>/force_areas/forceArea_000``), so the prefix is joined with
        ``/*`` — and since fnmatch's ``*`` also matches ``/``, that spans any
        nesting depth.
        """
        pattern = str(config.get("sensingShapePattern", "*/forceArea_[0-9][0-9][0-9]"))
        leaf = pattern.rsplit("/", 1)[-1]
        return f"{prim_path}/*{leaf}"

    @staticmethod
    def _pattern_matches(model, pattern: str) -> bool:
        return any(fnmatch.fnmatch(str(label), pattern) for label in model.shape_label)

    @staticmethod
    def _mount_rotation(prim) -> tuple:
        """The sensor prim's world rotation as ``(x, y, z, w)``.

        The force areas are static colliders (body ``-1``), so the kernel has no
        body pose to rotate the press axis by — the mount rotation has to be
        baked in at construction. This is the Isaac Sim equivalent of the
        hardcoded ``MOUNT_ROTATION`` in the standalone example, except derived
        from the prim's actual placement.
        """
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        quat = transform.RemoveScaleShear().ExtractRotationQuat()
        imag = quat.GetImaginary()
        return (float(imag[0]), float(imag[1]), float(imag[2]), float(quat.GetReal()))

    def _unbind(self) -> None:
        self._bindings = []
        self._bind_attempted = False
        self._bound_model = None
        self._steps = 0

    # ------------------------------------------------------------------ #
    # Readout
    # ------------------------------------------------------------------ #

    def read_back(self) -> None:
        """Refresh host-side copies for every bound sensor."""
        for binding in self._bindings:
            binding.read_back()

    def readout(self) -> dict:
        """Everything the panel needs for one refresh.

        Returns per-sensor forces plus the static metadata the heatmap lays out
        from, so the panel never reaches into ``CTSSensor`` itself.
        """
        self.read_back()
        sensors = []
        for binding in self._bindings:
            sensor = binding.sensor
            sensors.append(
                {
                    "prim_path": binding.prim_path,
                    "forces": binding.latest_forces,
                    "total_force": binding.latest_total_force,
                    "centroids": sensor.taxel_centroids,
                    "names": sensor.taxel_names,
                    "num_taxels": binding.num_taxels,
                    "force_max": float(binding.config.get("forceMaxN", DEFAULT_FORCE_MAX_N)),
                }
            )
        return {
            "sensors": sensors,
            "steps": self._steps,
            "error": self._last_error,
        }

    def summary(self) -> str:
        """One-screen text status. Formats the last read-back; does not re-read."""
        if self._last_error:
            return f"Not reading: {self._last_error}"
        if not self._bindings:
            return "No sensor bound. Load a scenario and press Play."

        lines = [f"Bound sensors: {len(self._bindings)}  |  steps: {self._steps}"]
        for binding in self._bindings:
            forces = binding.latest_forces
            if forces is None:
                continue
            active = int((forces > 1e-4).sum())
            lines.append(
                f"{binding.prim_path}\n"
                f"  total |F| : {binding.latest_total_force:8.4f} N\n"
                f"  Sum taxels: {float(forces.sum()):8.4f} N over {binding.num_taxels} taxels\n"
                f"  peak taxel: {float(forces.max()):8.4f} N   active: {active}"
            )
        return "\n".join(lines)
