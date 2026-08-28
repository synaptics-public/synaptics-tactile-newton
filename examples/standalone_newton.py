# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Standalone Newton example — CTS tactile sensor drop test.

Loads the physics-enabled CTS sensor USD, drops a known-mass cube straight down
onto the force areas, and prints per-taxel and total normal forces. After the
cube settles the total should approach m*g.

The force areas point +Y in the USD's local frame and import as STATIC colliders
(body == -1), so the sensor is loaded rotated +90 deg about X (force areas face
+Z), and that same rotation is passed as ``mount_rotation`` to bake the static
press axes into world frame.

Pass ``--help`` for the flag list and copy-pasteable example commands (the
script prints its own path, so the examples run as-is).
"""

import argparse
import math
import os

import warp as wp
import newton

from synaptics_tactile_newton import CTSSensor, format_forces

_HERE = os.path.dirname(__file__)
_ASSETS = os.path.join(_HERE, "..", "synaptics_tactile_newton", "assets")

# Physics-enabled CTS USD (grouped force areas + housing colliders) and its taxel map.
SENSOR_USD = os.path.join(_ASSETS, "cts0.0.usd")
TAXEL_MAP = os.path.join(_ASSETS, "cts0.0_taxel_map.json")

# Sensing shapes: one collider per force area, named forceArea_000..forceArea_067.
SENSING_PATTERN = "*/forceArea_*"

# Reorient the sensor so the +Y-pointing force areas face +Z (up): +90 deg about X.
# Quaternion (x, y, z, w) = (sin(t/2), 0, 0, cos(t/2)) with t = +pi/2.
_HALF = math.pi / 4.0
MOUNT_ROTATION = wp.quat(math.sin(_HALF), 0.0, 0.0, math.cos(_HALF))

# Lift the sensor so it sits just above the ground plane.
SENSOR_Z = 0.01

# Viewer camera (Rerun blueprint).
CAMERA_POS = (0.039113, -0.074668, 0.039714)
CAMERA_TARGET = (0.0, 0.0, 0.011259)
CAMERA_UP = (0.018591, 0.120664, 0.992519)

# Slower interactive navigation for the Newton GL viewer: its defaults (4.0 m/s
# fly, 0.15 scroll zoom) move far too fast at this cm-scale scene.
VIEWER_MOVE_SPEED = 0.1          # WASD fly speed [m/s] (ViewerGL default 4.0)
VIEWER_ZOOM_SENSITIVITY = 0.015  # scroll-wheel zoom (ViewerGL default 0.15)

# Path of THIS script relative to the current directory, so the example commands
# stay correct even if the file is renamed, without the noise of a long abs path.
_SCRIPT = os.path.relpath(__file__)
_EXAMPLES = f"""\
examples:
  # default: Rerun web viewer, cube drop, prints per-taxel + total force
  .venv-newton/bin/python {_SCRIPT}

  # headless: no viewer, run 400 steps
  .venv-newton/bin/python {_SCRIPT} --steps 400 --no-viewer

  # native OpenGL viewer window (needs a display)
  .venv-newton/bin/python {_SCRIPT} --gui
"""


def main():
    parser = argparse.ArgumentParser(
        description="CTS tactile sensor drop test",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=400, help="Simulation frames")
    parser.add_argument("--dt", type=float, default=1.0 / 480.0, help="Frame dt")
    parser.add_argument("--substeps", type=int, default=4, help="Substeps per frame")
    parser.add_argument("--mass", type=float, default=0.05, help="Cube mass [kg]")
    parser.add_argument(
        "--half-extent", type=float, default=0.005, help="Cube half-size [m]"
    )
    parser.add_argument(
        "--drop-height", type=float, default=0.05, help="Cube start height [m]"
    )
    parser.add_argument("--force-max", type=float, default=100.0, help="Taxel sat. [N]")
    parser.add_argument(
        "--no-mujoco-contacts",
        dest="use_mujoco_contacts",
        action="store_false",
        help="Pass use_mujoco_contacts=False to SolverMuJoCo so Newton's own "
             "collision pipeline (SDF/mesh contacts) generates contacts instead "
             "of MuJoCo's convex-hull path. Lets concave 'none' mesh colliders "
             "(housing/holder) collide as true triangle meshes and gives better "
             "spatial force distribution for the taxels. Default: MuJoCo contacts.",
    )
    parser.add_argument(
        "--nconmax",
        type=int,
        default=256,
        help="MJWarp contact-buffer size. SolverMuJoCo auto-sizes this from the "
             "INITIAL state (cube still airborne -> ~0 contacts -> ~48), which "
             "is too low for this scene: at rest the cube straddles ~9 convex "
             "taxel cells for ~88 legitimate contacts. 256 leaves headroom for "
             "the deeper-overlap impact frames. Only consumed with "
             "--no-mujoco-contacts (MuJoCo's own narrow-phase budgets itself).",
    )
    parser.add_argument(
        "--rrd",
        default=None,
        help="Also record the session to this .rrd file (open in the native "
             "Rerun desktop app; avoids the browser/JSPI entirely).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use Newton's native OpenGL viewer window (ViewerGL) instead of the "
             "Rerun web viewer. Interactive (WASD/mouse camera, pause/step, "
             "contact/joint toggles). Requires a display (X11 / RDP / local).",
    )
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    wp.init()

    usd_path = os.path.abspath(SENSOR_USD)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(
            f"CTS sensor USD not found at {usd_path}. The asset ships inside "
            "the package under synaptics_tactile_newton/assets/ — reinstall the "
            "package if it is missing."
        )

    # --- Build the scene --------------------------------------------------- #
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    builder.add_usd(
        usd_path,
        xform=wp.transform((0.0, 0.0, SENSOR_Z), MOUNT_ROTATION),
    )

    # Known-mass cube, centered over the sensor, dropped from above.
    he = args.half_extent
    cube_body = builder.add_body(
        xform=wp.transform(
            (0.0, 0.0, SENSOR_Z + args.drop_height), wp.quat_identity()
        ),
        label="test_cube",
    )
    cube_cfg = newton.ModelBuilder.ShapeConfig()
    cube_cfg.density = args.mass / (2.0 * he) ** 3
    builder.add_shape_box(
        body=cube_body,
        hx=he, hy=he, hz=he,
        cfg=cube_cfg,
        color=(1.0, 0.2, 0.2),
        label="test_cube_shape",
    )

    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(
        model,
        njmax=512,
        nconmax=args.nconmax,
        use_mujoco_contacts=args.use_mujoco_contacts,
    )
    if not args.use_mujoco_contacts:
        print(
            f"SolverMuJoCo: use_mujoco_contacts=False (Newton collision pipeline), "
            f"nconmax={args.nconmax}"
        )
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # --- Sensor ------------------------------------------------------------ #
    # Create the sensor BEFORE model.contacts() so it can request the contact
    # ``force`` buffer be allocated.
    sensor = CTSSensor(
        model,
        sensing_shape_pattern=SENSING_PATTERN,
        taxel_map=TAXEL_MAP,
        mount_rotation=MOUNT_ROTATION,
        force_max=args.force_max,
    )

    contacts = model.contacts()

    # --- Optional viewer --------------------------------------------------- #
    viewer = None
    if not args.no_viewer and args.gui:
        # Native OpenGL window. Needs a display (X11 / RDP / local session).
        try:
            from newton.viewer import ViewerGL
            from pyglet.math import Vec3 as _PyVec3

            viewer = ViewerGL()
            viewer.set_model(model)
            # Place the eye and aim it at the sensor.
            viewer.camera.pos = _PyVec3(*CAMERA_POS)
            viewer.camera.look_at(CAMERA_TARGET)
            # Slow the GL viewer's navigation: its defaults (4.0 m/s fly, 0.15
            # scroll zoom) move far too fast at this cm-scale scene. These live
            # on the viewer instance; guarded so a renamed attribute is a no-op.
            if hasattr(viewer, "_cam_speed"):
                viewer._cam_speed = VIEWER_MOVE_SPEED
            if hasattr(viewer, "_camera_dolly_scroll_sensitivity"):
                viewer._camera_dolly_scroll_sensitivity = VIEWER_ZOOM_SENSITIVITY
            if hasattr(viewer, "_camera_dolly_drag_sensitivity"):
                viewer._camera_dolly_drag_sensitivity = VIEWER_ZOOM_SENSITIVITY * 0.1
            print("Newton GL viewer window opened (close it or Ctrl+C to exit).")
        except Exception as exc:  # noqa: BLE001
            print(f"GL viewer unavailable ({exc}); running without viewer")
            viewer = None
    elif not args.no_viewer:
        try:
            from newton.viewer import ViewerRerun
            import rerun as rr
            import rerun.blueprint as rrb

            # keep_historical_data=True retains every frame so the Rerun
            # timeline can be scrubbed back and forth.
            viewer = ViewerRerun(
                serve_web_viewer=True,
                web_port=9090,
                keep_historical_data=True,
                record_to_rrd=args.rrd,
            )
            viewer.set_model(model)
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
            print("Rerun viewer at http://localhost:9090")
            print(
                "  (remote/SSH: forward ports 9090 AND 9876 \u2014 the page is on "
                "9090, scene data streams over 9876)"
            )
            if args.rrd:
                print(f"Recording to: {os.path.abspath(args.rrd)}")
        except ImportError:
            print("Rerun not available, running without viewer")

    expected = args.mass * 9.81
    print(
        f"Cube mass {args.mass} kg -> expected total force after settling "
        f"~{expected:.3f} N"
    )
    print(f"{'step':>5} | total[N]  active  max-taxel[N]")
    print("-" * 50)

    # --- Simulation loop --------------------------------------------------- #
    for step in range(args.steps):
        for _ in range(args.substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, None, contacts, dt=args.dt / args.substeps)
            solver.update_contacts(contacts, state_1)
            state_0, state_1 = state_1, state_0

        sensor.update(state_0, contacts)

        if step % 20 == 0 or step == args.steps - 1:
            f = sensor.data.force.numpy()
            print(
                f"{step:5d} | {f.sum():8.4f}  {int((f > 1e-4).sum()):5d}   "
                f"{f.max():8.4f}"
            )

        if viewer:
            viewer.begin_frame(step * args.dt)
            viewer.log_state(state_0)
            viewer.log_contacts(contacts, state_0)
            viewer.end_frame()

    f = sensor.data.force.numpy()
    print("-" * 50)
    print(
        f"Final total: {f.sum():.4f} N (expected ~{expected:.3f} N), "
        f"active taxels: {int((f > 1e-4).sum())}/{sensor.num_taxels}"
    )
    print(format_forces(sensor.data))
    if f.sum() < 1e-4:
        print(
            "NOTE: total force ~0 — the cube may have missed/bounced off the "
            "force areas (check geometry/drop alignment), or the taxel-map "
            "press-axis sign is wrong (fix it in the geometry splitter)."
        )

    if viewer:
        if args.gui:
            # Keep pumping frames so the native window stays interactive.
            print("GL viewer open — close the window (or Ctrl+C) to exit.")
            try:
                hold_t = args.steps * args.dt
                while viewer.is_running():
                    viewer.begin_frame(hold_t)
                    viewer.log_state(state_0)
                    viewer.log_contacts(contacts, state_0)
                    viewer.end_frame()
            except KeyboardInterrupt:
                pass
        else:
            print("Viewer still running at http://localhost:9090 — Ctrl+C to exit.")
            try:
                import time

                while viewer.is_running():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        viewer.close()


if __name__ == "__main__":
    main()

