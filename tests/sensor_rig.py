# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Reusable Newton scene for tactile-sensor validation tests.

:class:`SensorRig` loads the physics-enabled CTS USD, attaches a
:class:`synaptics_tactile_newton.CTSSensor`, adds test bodies (dead-weights,
kinematic probes), steps the solver, and reads per-taxel forces. The CTS USD is
in METERS, Z-up, with the molded force areas pointing +Y, so the rig rotates it
+90 deg about X to make them face +Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import warp as wp
import newton

from synaptics_tactile_newton import CTSSensor

# --------------------------------------------------------------------------- #
# Geometry / asset defaults. These constants are the single source of truth for
# the default asset path (conftest.py imports them).
# --------------------------------------------------------------------------- #
_ASSETS = Path(__file__).resolve().parent.parent / "synaptics_tactile_newton" / "assets"
DEFAULT_SENSOR_USD = _ASSETS / "cts0.0.usd"
DEFAULT_TAXEL_MAP = _ASSETS / "cts0.0_taxel_map.json"
DEFAULT_SENSOR_LABEL = "default"
_FORCE_AREA_PATTERN = "forceArea_*"
_CONTACT_GAP = 1.0e-3

# +90 deg about X so the +Y-pointing force areas face +Z (up). quat = (x,y,z,w).
_HALF = math.pi / 4.0
MOUNT_ROTATION = wp.quat(math.sin(_HALF), 0.0, 0.0, math.cos(_HALF))
MOUNT_QUAT_XYZW = np.array(
    [math.sin(_HALF), 0.0, 0.0, math.cos(_HALF)], dtype=np.float64
)

# Lift the sensor so it sits just above the ground plane.
SENSOR_Z = 0.01

# --------------------------------------------------------------------------- #
# Global test camera
# --------------------------------------------------------------------------- #
# Sensor-framed camera preset shared by both viewers. They default to a
# metre-scale world, useless for this ~cm sensor, so we override eye/target/up.
# ``CAMERA_UP`` is only used by the Rerun blueprint (ViewerGL derives its up).
CAMERA_POS = (0.039113, -0.074668, 0.039714)
CAMERA_TARGET = (0.0, 0.0, 0.011259)
CAMERA_UP = (0.018591, 0.120664, 0.992519)

GRAVITY = 9.81


def quat_mul_xyzw(q1, q2) -> np.ndarray:
    """Hamilton product of two (x, y, z, w) quaternions (applies ``q1`` after ``q2``)."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def local_to_world(centroid, quat_xyzw=MOUNT_QUAT_XYZW) -> np.ndarray:
    """Map a taxel centroid (body-local, metres) to a world position [m].

    Applies the same mount rotation + lift used when the USD is loaded, so a
    probe placed here sits above that force area. ``quat_xyzw`` defaults to the
    upright mount; pass a composed tilt to target a tilted sensor.

    Takes ``CTSSensor.taxel_centroids`` directly — those are already metres, in
    the same frame as the USD geometry, so there is no unit conversion here.
    """
    v = np.asarray(centroid, dtype=np.float64).reshape(3)
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    qxyz = q[:3]
    w = q[3]
    t = np.cross(qxyz, v) + w * v
    world = v + 2.0 * np.cross(qxyz, t)
    world[2] += SENSOR_Z
    return world


def centroid_to_world(centroid) -> np.ndarray:
    """Map a centroid to world [m] under the default upright mount rotation."""
    return local_to_world(centroid, MOUNT_QUAT_XYZW)


@dataclass
class SimConfig:
    """Tunable simulation parameters shared by all tests."""

    dt: float = 1.0 / 480.0
    substeps: int = 4
    njmax: int = 1024            # max MuJoCo constraint rows/world (avoid nefc overflow)
    nconmax: int = 512           # max MuJoCo contact points/world (avoid overflow)
    force_max: float = 10.0
    settle_vel: float = 1.0e-3      # max body speed [m/s] considered "at rest"
    settle_patience: int = 5        # consecutive calm frames required
    max_steps: int = 1500           # hard cap on a settle loop
    device: Optional[str] = None
    sensor_usd: str | Path = DEFAULT_SENSOR_USD
    taxel_map: str | Path = DEFAULT_TAXEL_MAP

    # --- optional Newton viewer (headless by default) ---------------------- #
    viewer: bool = False            # stream the run to a Newton viewer
    gui: bool = False               # use the native OpenGL ViewerGL window
                                    # instead of the Rerun web viewer
    viewer_port: int = 9090         # web viewer port (http://localhost:<port>)
    viewer_rrd: Optional[str] = None  # also record the session to this .rrd file


@dataclass
class SensorRigFactory:
    """Callable that produces a fresh :class:`SensorRig` per invocation.

    Tests that need a clean scene each run (e.g. repeatability) call the factory
    repeatedly; each call rebuilds the model from scratch.
    """

    config: SimConfig = field(default_factory=SimConfig)

    def __call__(self) -> "SensorRig":
        return SensorRig(self.config)


class SensorRig:
    """A buildable Newton scene: sensor + optional test bodies.

    Lifecycle: construct -> ``add_weight``/``add_probe`` (any number, before
    finalize) -> ``finalize`` -> ``step``/``settle`` -> read ``forces``. The
    sensor is created after finalize but before contacts, matching the order
    ``CTSSensor`` requires.
    """

    # Process-wide Newton viewer (one web server / port reused across rigs).
    _VIEWER = None
    # Latest (sim_time, state, contacts) logged to the viewer (for GL hold loop).
    _LAST = None

    def __init__(self, config: SimConfig, mount_quat_xyzw=MOUNT_QUAT_XYZW,
                 add_default_sensor: bool = True):
        self.config = config
        # Mount rotation applied to the loaded USD and baked into the sensor's
        # static press axis. It is also used by ``local_to_world`` for probe
        # placement.
        self.mount_quat_xyzw = np.asarray(mount_quat_xyzw, dtype=np.float64)
        if config.device is not None:
            wp.set_device(config.device)
        self.device = wp.get_device().alias

        usd_path = Path(config.sensor_usd).resolve()
        if not usd_path.exists():
            raise FileNotFoundError(
                f"CTS sensor USD not found at {usd_path}. The asset ships "
                "inside the package under synaptics_tactile_newton/assets/ — "
                "reinstall the package if it is missing."
            )
        self._usd_path = usd_path

        self.builder = newton.ModelBuilder()
        self.builder.rigid_gap = _CONTACT_GAP
        self.builder.add_ground_plane()

        # Sensors are static world shapes. Their CTSSensor instances can only
        # be created after the model is finalized.
        self._sensor_rotations: dict[str, object] = {}
        self.sensors: dict[str, CTSSensor] = {}
        if add_default_sensor:
            self.add_sensor(
                (0.0, 0.0, SENSOR_Z),
                self.mount_quat_xyzw,
                label=DEFAULT_SENSOR_LABEL,
            )

        # body idx -> spatial velocity command (linear xyz, angular xyz). The
        # probe is DISPLACEMENT-controlled: each substep we prescribe the body's
        # free-joint translation (joint_q), which SolverMuJoCo reads back into
        # qpos every step, rather than setting body_qd (which it ignores).
        # Resolved in finalize() once the model/joints exist.
        self._kinematic: dict[int, tuple] = {}
        self._kinematic_qd_start: dict[int, int] = {}
        self._kinematic_q_start: dict[int, int] = {}
        self._kinematic_pos: dict[int, object] = {}
        # body indices that are free dynamic test bodies (for settle detection)
        self._free_bodies: list[int] = []
        self.model = None
        self.solver = None
        self.contacts = None
        self.state_0 = None
        self.state_1 = None
        self.viewer = None
        self._sim_time = 0.0

    # --- scene construction (before finalize) ------------------------------ #
    def add_box(
        self,
        mass: float,
        half_extents: tuple = (0.005, 0.005, 0.005),
        xy: tuple = (0.0, 0.0),
        height: float = 0.03,
        label: str = "weight",
    ) -> int:
        """Add a free box of known mass resting/falling above the sensor.

        ``half_extents`` is ``(hx, hy, hz)`` [m]; a wide thin box acts as a plate
        that loads the whole sensor at once.
        """
        hx, hy, hz = half_extents
        hx, hy, hz = float(hx), float(hy), float(hz)
        body = self.builder.add_body(
            xform=wp.transform(
                (xy[0], xy[1], SENSOR_Z + height), wp.quat_identity()
            ),
            label=label,
        )
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = float(mass) / (8.0 * hx * hy * hz)
        self.builder.add_shape_box(
            body=body,
            hx=hx, hy=hy, hz=hz,
            cfg=cfg,
            color=(1.0, 0.2, 0.2),
            label=f"{label}_shape",
        )
        self._free_bodies.append(body)
        return body

    def add_weight(
        self,
        mass: float,
        half_extent: float = 0.005,
        xy: tuple = (0.0, 0.0),
        height: float = 0.03,
        label: str = "weight",
    ) -> int:
        """Add a free cube of known mass resting/falling above the sensor."""
        return self.add_box(
            mass=mass,
            half_extents=(half_extent, half_extent, half_extent),
            xy=xy,
            height=height,
            label=label,
        )

    def add_probe(
        self,
        velocity_z: float = -0.001,
        radius: float = 0.002,
        xy: tuple = (0.0, 0.0),
        height: float = 0.02,
        start_z: Optional[float] = None,
        mu: Optional[float] = None,
        label: str = "probe",
    ) -> int:
        """Add a velocity-driven (kinematic) spherical probe.

        The probe's spatial velocity is re-commanded each frame, approximating a
        linear motor: contacts perturb it within a frame but the command is
        re-asserted next frame. ``velocity_z`` is the downward (negative) world-Z
        speed [m/s].

        ``start_z`` sets the probe centre's absolute world Z [m] (default
        ``SENSOR_Z + height``); pass a value just above a force area's centroid
        to place the probe directly over that taxel. ``mu`` sets the probe's
        friction; use a large value (e.g. for a tilted sensor) so it grips the
        surface instead of sliding off the dome under a non-axial press.
        """
        z0 = start_z if start_z is not None else SENSOR_Z + height
        # A 2 mm-radius sphere's true inertia (~5e-11 kg*m^2) is below Newton's
        # inertia floor, so finalize() would clamp it and warn. The probe is
        # velocity-driven (kinematic), so its inertia never enters the dynamics;
        # give it an explicit floor-clearing inertia and lock it so the sphere
        # shape can't recompute a sub-floor tensor.
        density = 1000.0
        mass = density * (4.0 / 3.0) * math.pi * radius**3
        probe_I = 1.0e-6  # kg*m^2, at/above Newton's inertia floor
        body = self.builder.add_body(
            xform=wp.transform(
                (xy[0], xy[1], z0), wp.quat_identity()
            ),
            mass=mass,
            inertia=wp.mat33(probe_I, 0.0, 0.0, 0.0, probe_I, 0.0, 0.0, 0.0, probe_I),
            lock_inertia=True,
            label=label,
        )
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = density
        if mu is not None:
            cfg.mu = float(mu)
        self.builder.add_shape_sphere(
            body=body,
            radius=radius,
            cfg=cfg,
            color=(0.2, 0.4, 1.0),
            label=f"{label}_shape",
        )
        # Newton's body_qd spatial vector is (linear, angular), so world-Z speed
        # goes in index 2.
        self._kinematic[body] = (0.0, 0.0, float(velocity_z), 0.0, 0.0, 0.0)
        return body

    def add_sensor(
        self,
        position,
        orientation_xyzw,
        label: str,
    ) -> None:
        """Add a STATIC sensor (``body == -1``) at a fixed world pose.

        Loads the CTS USD into a scratch builder and merges it with a
        ``label_prefix`` so a per-sensor glob scopes one ``CTSSensor`` to this
        sensor's pads. ``position`` [m] and ``orientation_xyzw`` (x, y, z, w)
        place the sensor; the orientation is also baked into the press axis.
        After ``finalize()``, the resulting ``CTSSensor`` is available as
        ``rig.sensors[label]``.

        Multiple sensors share one scene, so a single ball can press several and
        viewer/step/settle work unchanged. Loading as static colliders
        (``body == -1``) keeps a single rigid body from owning many colliders,
        which the USD parser cannot handle.
        """
        if label in self._sensor_rotations:
            raise ValueError(f"sensor label already exists: {label!r}")
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        q = np.asarray(orientation_xyzw, dtype=np.float64).reshape(4)
        rot = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        # Merge via a scratch builder so a label prefix distinguishes repeated
        # loads of the same USD (add_usd cannot prefix labels on its own).
        sub = newton.ModelBuilder()
        sub.rigid_gap = _CONTACT_GAP
        sub.add_usd(str(self._usd_path))
        self.builder.add_builder(
            sub,
            xform=wp.transform(
                (float(pos[0]), float(pos[1]), float(pos[2])), rot
            ),
            label_prefix=f"{label}/",
        )
        self._sensor_rotations[label] = rot

    def add_ball(
        self,
        mass: float,
        radius: float,
        xy: tuple = (0.0, 0.0),
        height: float = 0.03,
        start_z: Optional[float] = None,
        mu: float = 1.0,
        label: str = "ball",
    ) -> int:
        """Add a free dynamic ball of known mass above the sensors.

        ``start_z`` sets the centre's absolute world Z [m] (default
        ``SENSOR_Z + height``). The ball also supplies the free joint the MuJoCo
        solver requires, so a scene of only static sensors + one ball is valid.
        """
        z0 = start_z if start_z is not None else SENSOR_Z + height
        body = self.builder.add_body(
            xform=wp.transform((xy[0], xy[1], z0), wp.quat_identity()),
            label=label,
        )
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = float(mass) / ((4.0 / 3.0) * math.pi * radius**3)
        cfg.mu = float(mu)
        self.builder.add_shape_sphere(
            body=body,
            radius=radius,
            cfg=cfg,
            color=(1.0, 0.2, 0.2),
            label=f"{label}_shape",
        )
        self._free_bodies.append(body)
        return body

    # --- finalize ---------------------------------------------------------- #
    def finalize(self, build_solver: bool = True) -> "SensorRig":
        """Build the model, sensor and contacts.

        ``build_solver=False`` skips the MuJoCo solver. Use it for read-only
        rigs that only inspect the sensor (e.g. taxel centroids/names) without
        stepping — a body-less scene has no joints, which the MuJoCo solver
        rejects.
        """
        cfg = self.config
        static_shapes = [
            shape for shape, body in enumerate(self.builder.shape_body) if body == -1
        ]
        for offset, shape_a in enumerate(static_shapes):
            for shape_b in static_shapes[offset + 1:]:
                self.builder.add_shape_collision_filter_pair(shape_a, shape_b)

        self.model = self.builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )

        # Resolve each kinematic body's free-joint DOFs and record its start
        # pose. add_body() auto-creates a free joint; q layout is [pos(3),
        # quat(4)], qd layout is [omega(3), v(3)].
        self._kinematic_qd_start = {}
        self._kinematic_q_start = {}
        self._kinematic_pos = {}
        if self._kinematic:
            joint_child = self.model.joint_child.numpy()
            joint_qd_start = self.model.joint_qd_start.numpy()
            joint_q_start = self.model.joint_q_start.numpy()
            jq = self.state_0.joint_q.numpy()
            for body in self._kinematic:
                matches = np.where(joint_child == body)[0]
                if len(matches) == 0:
                    raise RuntimeError(
                        f"kinematic body {body} has no free joint to drive"
                    )
                j = int(matches[0])
                qs = int(joint_q_start[j])
                self._kinematic_q_start[body] = qs
                self._kinematic_qd_start[body] = int(joint_qd_start[j])
                self._kinematic_pos[body] = jq[qs:qs + 3].astype(np.float64).copy()

        if build_solver:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, njmax=cfg.njmax, nconmax=cfg.nconmax
            )

        # Sensors must be created BEFORE model.contacts() to request the force buffer.
        for label, rotation in self._sensor_rotations.items():
            self.sensors[label] = CTSSensor(
                self.model,
                sensing_shape_pattern=f"{label}/*/{_FORCE_AREA_PATTERN}",
                taxel_map=cfg.taxel_map,
                mount_rotation=rotation,
                force_max=cfg.force_max,
            )
        self.contacts = self.model.contacts()
        if cfg.viewer:
            self._ensure_viewer()
        return self

    # --- optional Newton viewer ------------------------------------------- #
    def _ensure_viewer(self) -> None:
        """Attach a Newton viewer (process-wide singleton). Headless-safe.

        ``gui`` selects the native OpenGL window (``ViewerGL``); otherwise the
        Rerun web viewer (``ViewerRerun``). Stays headless if unavailable.
        """
        if self.config.gui:
            self._ensure_gl_viewer()
        else:
            self._ensure_rerun_viewer()

    def _ensure_rerun_viewer(self) -> None:
        """Attach a Newton ``ViewerRerun`` (reused process-wide). Headless-safe."""
        cfg = self.config
        try:
            from newton.viewer import ViewerRerun
        except ImportError:
            print("Newton viewer unavailable (rerun not installed); headless.")
            return
        viewer = getattr(SensorRig, "_VIEWER", None)
        if viewer is None:
            viewer = ViewerRerun(
                serve_web_viewer=True,
                web_port=cfg.viewer_port,
                keep_historical_data=True,
                record_to_rrd=cfg.viewer_rrd,
            )
            SensorRig._VIEWER = viewer
            print(f"Newton viewer at http://localhost:{cfg.viewer_port}")
        self.viewer = viewer
        self.viewer.set_model(self.model)
        self._apply_camera()
        self._recolor_shapes()
        self._sim_time = 0.0

    def _ensure_gl_viewer(self) -> None:
        """Attach Newton's native OpenGL ``ViewerGL`` window (process-wide).

        Needs a display (X11 / RDP / local); stays headless if unavailable.
        """
        try:
            from newton.viewer import ViewerGL
        except Exception as exc:  # noqa: BLE001
            print(f"Newton GL viewer unavailable ({exc}); headless.")
            return
        viewer = getattr(SensorRig, "_VIEWER", None)
        if viewer is None:
            viewer = ViewerGL()
            SensorRig._VIEWER = viewer
            print("Newton GL viewer window opened (close it or Ctrl+C to exit).")
        self.viewer = viewer
        self.viewer.set_model(self.model)
        self._apply_camera()
        self._recolor_shapes()
        self._sim_time = 0.0

    def _apply_camera(self) -> None:
        """Aim the active viewer at the global test camera (``CAMERA_*``).

        Drives a ViewerGL camera directly or a ViewerRerun via a blueprint.
        Framing is non-essential, so any failure leaves the default in place.
        """
        if self.viewer is None:
            return
        if self.config.gui:
            try:
                from pyglet.math import Vec3 as _PyVec3

                self.viewer.camera.pos = _PyVec3(*CAMERA_POS)
                self.viewer.camera.look_at(CAMERA_TARGET)
            except Exception:  # noqa: BLE001 - camera framing is non-essential
                pass
            return
        try:
            import rerun as rr
            import rerun.blueprint as rrb

            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.Spatial3DView(
                        origin="/",
                        eye_controls=rrb.EyeControls3D(
                            position=CAMERA_POS,
                            look_target=CAMERA_TARGET,
                            eye_up=CAMERA_UP,
                        ),
                    ),
                    auto_layout=True,
                )
            )
        except Exception:  # noqa: BLE001 - camera framing is non-essential
            pass

    def _recolor_shapes(self) -> None:
        """Cosmetic: restore intended group colors (Newton ignores displayColor)."""
        try:
            colors = self.model.shape_color.numpy().copy()
            for i, label in enumerate(self.model.shape_label):
                name = str(label)
                if "forceArea_" in name or "rubber_base" in name:
                    colors[i] = (0.85, 0.10, 0.10)
                elif "holder" in name:
                    colors[i] = (0.50, 0.50, 0.50)
            self.model.shape_color.assign(colors)
        except Exception:  # noqa: BLE001 - colouring is purely cosmetic
            pass

    # --- stepping ---------------------------------------------------------- #
    def _apply_kinematic(self, sub_dt: float) -> None:
        if not self._kinematic:
            return
        # Advance the prescribed translation by v*sub_dt and write it into the
        # free-joint coords (joint_q) plus matching velocity (joint_qd); the
        # solver reads both back each step. command is (vx, vy, vz, wx, wy, wz).
        q = self.state_0.joint_q.numpy()
        qd = self.state_0.joint_qd.numpy()
        for body in self._kinematic:
            vx, vy, vz, wx, wy, wz = self._kinematic[body]
            qs = self._kinematic_q_start[body]
            ds = self._kinematic_qd_start[body]
            pos = self._kinematic_pos[body]
            pos[0] += vx * sub_dt
            pos[1] += vy * sub_dt
            pos[2] += vz * sub_dt
            q[qs + 0] = pos[0]
            q[qs + 1] = pos[1]
            q[qs + 2] = pos[2]
            qd[ds + 0] = wx
            qd[ds + 1] = wy
            qd[ds + 2] = wz
            qd[ds + 3] = vx
            qd[ds + 4] = vy
            qd[ds + 5] = vz
        self.state_0.joint_q.assign(q)
        self.state_0.joint_qd.assign(qd)

    def step(self, n: int = 1) -> None:
        if self.solver is None:
            raise RuntimeError("step() requires a solver; finalize(build_solver=True)")
        cfg = self.config
        sub_dt = cfg.dt / cfg.substeps
        for _ in range(n):
            for _ in range(cfg.substeps):
                self._apply_kinematic(sub_dt)
                self.state_0.clear_forces()
                self.model.collide(self.state_0, self.contacts)
                self.solver.step(
                    self.state_0, self.state_1, None, self.contacts, dt=sub_dt
                )
                self.solver.update_contacts(self.contacts, self.state_1)
                self.state_0, self.state_1 = self.state_1, self.state_0
            for sensor in self.sensors.values():
                sensor.update(self.state_0, self.contacts)
            if self.viewer is not None:
                self.viewer.begin_frame(self._sim_time)
                self.viewer.log_state(self.state_0)
                self.viewer.log_contacts(self.contacts, self.state_0)
                self.viewer.end_frame()
                self._sim_time += cfg.dt
                # Remember the latest frame so a GL hold loop can keep rendering
                # it after the test finishes (the web viewer keeps its own state).
                SensorRig._LAST = (self._sim_time, self.state_0, self.contacts)

    def max_free_speed(self) -> float:
        """Largest linear speed among free dynamic test bodies [m/s]."""
        if not self._free_bodies:
            return 0.0
        qd = self.state_0.body_qd.numpy()
        # body_qd is (linear, angular), so the linear speed is components [0:3].
        lin = qd[self._free_bodies, 0:3]
        return float(np.linalg.norm(lin, axis=1).max())

    def settle(self) -> int:
        """Step until free bodies are at rest (or ``max_steps``). Returns steps."""
        cfg = self.config
        calm = 0
        steps = 0
        while steps < cfg.max_steps:
            self.step(1)
            steps += 1
            if self.max_free_speed() <= cfg.settle_vel:
                calm += 1
                if calm >= cfg.settle_patience:
                    break
            else:
                calm = 0
        return steps

    # --- readouts ---------------------------------------------------------- #
    def forces(self) -> np.ndarray:
        """Per-taxel normal force [N], shape (num_taxels,)."""
        return self.sensors[DEFAULT_SENSOR_LABEL].data.force.numpy()

    def total_force_vector(self) -> np.ndarray:
        """Net normal-force vector [N] on the sensing surface, shape (3,)."""
        return self.sensors[DEFAULT_SENSOR_LABEL].data.total_force.numpy()

    def total_force(self) -> float:
        """Scalar net normal force [N] = magnitude of the total-force vector."""
        return float(np.linalg.norm(self.total_force_vector()))

    def active_count(self, threshold: float = 1.0e-4) -> int:
        return int((self.forces() > threshold).sum())

    def body_position(self, body: int) -> np.ndarray:
        q = self.state_0.body_q.numpy()
        return q[body, 0:3].copy()

    @property
    def num_taxels(self) -> int:
        return self.sensors[DEFAULT_SENSOR_LABEL].num_taxels

    @property
    def taxel_centroids(self) -> Optional[np.ndarray]:
        return self.sensors[DEFAULT_SENSOR_LABEL].taxel_centroids

    @property
    def taxel_names(self):
        """Per-taxel force-area names in output row order, or None."""
        return self.sensors[DEFAULT_SENSOR_LABEL].taxel_names
