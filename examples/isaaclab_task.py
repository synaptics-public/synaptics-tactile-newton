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

"""Isaac Lab demo — Synaptics CTS tactile sensor (rich CLI + artifact saving).

A runnable demo of the CTS tactile sensor in an Isaac Lab scene: it drops an
object onto the sensor pads and reads the per-taxel signal through the standard
``sensor.data`` property, wrapped in a CLI with on-disk artifact capture.

* **CLI flags** — pick the indenter shape/offset, override force saturation,
  and dump artifacts to disk.
* **Deliberate interaction** — an object is dropped onto the pads; the natural
  impact → settle motion exercises the sensor's rise → saturate (``force_max``)
  → rest response. A lateral offset presses off-center to show the spatial
  distribution across force areas.
* **Artifact saving** — per-step per-taxel force (``force.npy``), net force
  vector (``total_force.npy``), world-frame taxel positions (``positions_w.npy``),
  a per-step/per-env ``readout.csv``, and an optional force-field heatmap PNG.

The signal is the CTS's real per-taxel normal force from the Newton solver; there
are no optical RGB/depth or shear channels.

Why the box-built body
----------------------
The sensor body is assembled directly through Newton's ``ModelBuilder`` (a
per-world cloner hook) and never handed to USD's physics parser: the
multi-collider CTS body triggers an OpenUSD parallel-parse race that crashes on
this stack (isaac-sim/IsaacSim#692).

Run with the Isaac Lab launcher (``./isaaclab.sh -p``); pass ``--help`` for the
prerequisites, the full flag list, and copy-pasteable example commands (the
script prints its own path, so the examples run as-is).
"""

import os
import argparse

from isaaclab.app import AppLauncher

# --- CLI + app launch (Isaac Lab boilerplate; the app must start first) ----- #
# Path of THIS script relative to the current directory (whatever it is named in
# this repo), so the example commands stay correct after the release rename
# without the noise of a long absolute path.
_SCRIPT = os.path.relpath(__file__)
_EXAMPLES = f"""\
first activate the venv and cd to your Isaac Lab checkout (it boots Kit, so the
Isaac Sim app env must be active):
  source .venv-newton/bin/activate
  cd /path/to/IsaacLab            # must contain the _isaac_sim symlink

then, for example:
  # live view: watch the press
  ./isaaclab.sh -p {_SCRIPT} \\
      --num_envs 4 --object sphere --offset_x 0.004 \\
      --steps 600 --viz newton

  # artifacts: no viewer, dump per-step force/positions/CSV + a heatmap PNG
  ./isaaclab.sh -p {_SCRIPT} \\
      --num_envs 4 --object sphere --offset_x 0.004 \\
      --steps 600 --save_dir ./cts_demo_artifacts --heatmap --viz none
"""
parser = argparse.ArgumentParser(
    description="Synaptics CTS tactile sensor demo (box-built body).",
    epilog=_EXAMPLES,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument(
    "--object",
    choices=("cube", "sphere", "cylinder"),
    default="cube",
    help="Indenter shape dropped onto the pads.",
)
parser.add_argument("--offset_x", type=float, default=0.0, help="Lateral press offset X [m].")
parser.add_argument("--offset_y", type=float, default=0.0, help="Lateral press offset Y [m].")
parser.add_argument("--force_max", type=float, default=100.0, help="Per-taxel force saturation [N].")
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Stop after N sim steps (0 = run until the window is closed).",
)
parser.add_argument(
    "--reset_interval",
    type=int,
    default=500,
    help="Re-drop the object every N steps (0 = never reset).",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="If set, dump per-step force/positions/CSV artifacts to this directory.",
)
parser.add_argument(
    "--heatmap",
    action="store_true",
    help="Also save a force-field heatmap PNG (implies --save_dir; needs matplotlib).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --heatmap on its own still needs somewhere to write; default a local folder.
if args_cli.heatmap and not args_cli.save_dir:
    args_cli.save_dir = "cts_demo_artifacts"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator is live."""

import csv
import fnmatch
import math

import numpy as np
import warp as wp
import newton

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonCfg, NewtonManager
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg

from synaptics_tactile_newton.isaaclab import CTSSensorCfg

# The sensor body is built by hand below (not loaded from USD like the cube)
# because the multi-collider CTS body crashes OpenUSD's parallel physics parse
# (isaac-sim/IsaacSim#692), which runs deep inside Isaac Sim/Kit's cloner. Kit
# forces the thread limit before USD initializes, so the single-thread workaround
# can't be applied. Instead we read only the geometry (UsdGeom, which is safe)
# and rebuild the physics directly in Newton.

# --- CTS geometry (same assets the standalone Newton example uses) ---------- #
_HERE = os.path.dirname(__file__)
_ASSETS = os.path.join(_HERE, "..", "synaptics_tactile_newton", "assets")
SENSOR_USD = os.path.abspath(os.path.join(_ASSETS, "cts0.0.usd"))
TAXEL_MAP = os.path.abspath(os.path.join(_ASSETS, "cts0.0_taxel_map.json"))

# Force-area collider prims in the USD; one collider per match. The trailing
# [0-9][0-9][0-9] keeps the match to the collision force areas only (Newton's USD
# importer also emits visual-only ``forceArea_NNN_visual`` twins).
FORCE_AREA_PATTERN = "forceArea_[0-9][0-9][0-9]"
# The sensor wrapper matches Newton SHAPE LABELS with this glob. Boxes are
# labelled ``Sensor/Module/forceArea_NNN`` in every world, so ``*/`` matches the
# force areas across all environments; the label basename keys the taxel map.
SENSING_PATTERN = "*/forceArea_[0-9][0-9][0-9]"

# The molded force areas point +Y in the sensor's local frame. A +90 deg rotation
# about X turns that into +Z so an object dropped from above lands on the pads.
# The sensor body carries this as its pose (the sensor reads the live body pose,
# so no ``mount_rotation`` is passed to the wrapper). Quaternion is (x, y, z, w).
_HALF = math.pi / 4.0
MOUNT_ROTATION = wp.quat(math.sin(_HALF), 0.0, 0.0, math.cos(_HALF))

# Sensor body origin height above each environment's ground plane [m].
SENSOR_Z = 0.012

# Height the indenter is dropped from [m]: above the pad tops so it does not spawn
# inside them, low enough that the impact energy stays small.
DROP_Z = 0.05

# Initial framing for the interactive Newton GL / Rerun viewers. Those viewers
# read their starting camera ONLY from the visualizer cfg's eye/lookat
# (``sim.set_camera_view`` moves the Kit viewport, not these), so the pose is
# attached to the visualizer cfg in ``_viewer_cfgs``. ``lookat`` is a point along
# the view direction; the 65 deg FOV is the default focal length, so it is not
# set here.
VIEWER_EYE = (0.40, -0.57, 0.21)
VIEWER_LOOKAT = (-0.087, -0.060, 0.033)

# Slower interactive navigation for the Newton GL viewer (its defaults feel too
# fast at this scene scale). The GL viewer exposes these only as instance
# attributes, so they are applied to the live viewer in
# ``_slow_viewer_navigation``. The Rerun viewer navigates browser-side and is
# left untouched.
VIEWER_MOVE_SPEED = 0.1          # WASD fly speed [m/s] (ViewerGL default 4.0)
VIEWER_ZOOM_SENSITIVITY = 0.015  # scroll-wheel zoom (ViewerGL default 0.15)

# Rerun's 3D view has no FOV control (its EyeControls3D ignores the cfg focal
# length) and uses a narrower default than the Newton viewer, so VIEWER_EYE frames
# more tightly in Rerun. The Rerun eye is pulled back along the same view
# direction by this factor to match the Newton viewer; raise to zoom out, lower
# to zoom in.
RERUN_PULLBACK = 1.5

# Contact stiffness/damping the assets were tuned with (authored on the force
# areas as ``newton:contact_ke`` / ``newton:contact_kd``); used as fallbacks if
# the prims do not carry them.
DEFAULT_CONTACT_KE = 250000.0
DEFAULT_CONTACT_KD = 1000.0

# Populated once in ``main()`` before the scene is cloned; read by the per-world
# builder hook. Each entry is a per-shape dict from ``load_force_area_shapes``
# (box / convex hull / mesh, in METERS in the sensor body-local frame).
_SENSOR_SHAPES: list = []

# The non-force-area housing meshes (board base, connectors, holder). Populated
# in ``main()`` alongside ``_SENSOR_SHAPES`` and added to the same body as real
# colliding triangle meshes, so the whole sensor collides, not just the pads.
_SENSOR_STRUCTURE: list = []


def load_force_area_shapes(usd_path):
    """Read each ``forceArea_NNN`` collider from the USD, preserving its shape.

    Uses ``UsdGeom`` only (never ``UsdPhysics``), so opening the asset never
    invokes USD's physics parser. Each force area is returned in the sensor
    body-local frame, in METERS, faithful to how it was authored: a
    ``UsdGeom.Mesh`` pad as its triangle geometry (a convex hull, or the raw mesh
    when ``physics:approximation`` is ``none``), any other collider (e.g. a
    ``UsdGeom.Cube``) as an axis-aligned box from its bounds. Returns per-shape
    dicts (``name``, ``kind``, ``ke``, ``kd`` + geometry fields), sorted by name.
    """
    from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"could not open USD stage: {usd_path}")
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0

    prims = [p for p in stage.Traverse() if fnmatch.fnmatch(p.GetName(), FORCE_AREA_PATTERN)]
    prims.sort(key=lambda p: p.GetName())
    if not prims:
        raise RuntimeError(f"no force-area prims matching '{FORCE_AREA_PATTERN}' in {usd_path}")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    shapes = []
    for p in prims:
        ke_attr = p.GetAttribute("newton:contact_ke")
        kd_attr = p.GetAttribute("newton:contact_kd")
        ke = float(ke_attr.Get()) if ke_attr and ke_attr.HasValue() else DEFAULT_CONTACT_KE
        kd = float(kd_attr.Get()) if kd_attr and kd_attr.HasValue() else DEFAULT_CONTACT_KD

        bbox = cache.ComputeLocalBound(p)
        points = UsdGeom.Mesh(p).GetPointsAttr().Get() if p.IsA(UsdGeom.Mesh) else None

        if points:
            # Bake the prim transform into the vertices, convert to metres, and
            # fan-triangulate polygonal faces.
            mesh = UsdGeom.Mesh(p)
            matrix = bbox.GetMatrix()
            verts = (
                np.array(
                    [matrix.Transform(Gf.Vec3d(v[0], v[1], v[2])) for v in points],
                    dtype=np.float64,
                )
                * mpu
            )
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            face_idx = mesh.GetFaceVertexIndicesAttr().Get() or []
            tris = []
            offset = 0
            for c in counts:
                for k in range(1, c - 1):
                    tris += [face_idx[offset], face_idx[offset + k], face_idx[offset + k + 1]]
                offset += c
            approx_attr = p.GetAttribute("physics:approximation")
            approx = approx_attr.Get() if approx_attr and approx_attr.HasValue() else "convexHull"
            shapes.append(
                {
                    "name": p.GetName(),
                    "kind": "mesh" if approx == "none" else "hull",
                    "ke": ke,
                    "kd": kd,
                    "vertices": verts.astype(np.float32),
                    "indices": np.array(tris, dtype=np.int32),
                }
            )
        else:
            rng = bbox.ComputeAlignedRange()
            mn = np.array(rng.GetMin(), dtype=np.float64) * mpu
            mx = np.array(rng.GetMax(), dtype=np.float64) * mpu
            shapes.append(
                {
                    "name": p.GetName(),
                    "kind": "box",
                    "ke": ke,
                    "kd": kd,
                    "center": 0.5 * (mn + mx),
                    "half": np.maximum(0.5 * (mx - mn), 1e-6),
                }
            )
    return shapes


def load_structure_meshes(usd_path):
    """Read the sensor's non-force-area housing meshes as colliding triangle geometry.

    Everything around the force areas (board base, connectors, holder) is a
    ``UsdGeom.Mesh``. Like :func:`load_force_area_shapes` this uses ``UsdGeom``
    only (never the physics parser), returning each mesh as raw triangles in the
    sensor body-local frame, in METERS, so the housing collides with the real
    geometry it was authored as. Returns per-shape dicts (``name``, ``ke``,
    ``kd``, ``vertices``, ``indices``), sorted by name.
    """
    from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"could not open USD stage: {usd_path}")
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0

    prims = [
        p
        for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh) and not fnmatch.fnmatch(p.GetName(), FORCE_AREA_PATTERN)
    ]
    prims.sort(key=lambda p: p.GetName())

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    shapes = []
    for p in prims:
        mesh = UsdGeom.Mesh(p)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        # Bake the prim transform into the vertices, convert to metres, and
        # fan-triangulate polygonal faces (same handling as the mesh force areas).
        # /Root and the housing's parent Xforms are identity, so this matrix lands
        # the geometry in the same body-local frame as the pads.
        matrix = cache.ComputeLocalBound(p).GetMatrix()
        verts = (
            np.array(
                [matrix.Transform(Gf.Vec3d(v[0], v[1], v[2])) for v in points],
                dtype=np.float64,
            )
            * mpu
        )
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        face_idx = mesh.GetFaceVertexIndicesAttr().Get() or []
        tris = []
        offset = 0
        for c in counts:
            for k in range(1, c - 1):
                tris += [face_idx[offset], face_idx[offset + k], face_idx[offset + k + 1]]
            offset += c
        shapes.append(
            {
                "name": p.GetName(),
                "ke": DEFAULT_CONTACT_KE,
                "kd": DEFAULT_CONTACT_KD,
                "vertices": verts.astype(np.float32),
                "indices": np.array(tris, dtype=np.int32),
            }
        )
    return shapes


def _add_structure_shape(builder, body, shape, label):
    """Add one housing mesh to ``body`` as a colliding triangle mesh."""
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = shape["ke"]
    cfg.kd = shape["kd"]
    mesh = newton.Mesh(shape["vertices"], shape["indices"], compute_inertia=False)
    builder.add_shape_mesh(
        body=body, mesh=mesh, cfg=cfg, color=(0.55, 0.55, 0.6), label=label
    )


def _add_force_area_shape(builder, body, shape, label):
    """Add one force-area collider (box / convex hull / triangle mesh) to ``body``.

    Dispatches on the ``kind`` recorded by :func:`load_force_area_shapes`, so the
    Newton shape matches how the force area was authored in USD.
    """
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = shape["ke"]
    cfg.kd = shape["kd"]
    color = (0.2, 0.6, 1.0)
    if shape["kind"] == "box":
        center = shape["center"]
        half = shape["half"]
        builder.add_shape_box(
            body=body,
            xform=wp.transform(
                (float(center[0]), float(center[1]), float(center[2])), wp.quat_identity()
            ),
            hx=float(half[0]),
            hy=float(half[1]),
            hz=float(half[2]),
            cfg=cfg,
            color=color,
            label=label,
        )
        return
    mesh = newton.Mesh(shape["vertices"], shape["indices"], compute_inertia=False)
    add_shape = (
        builder.add_shape_mesh if shape["kind"] == "mesh" else builder.add_shape_convex_hull
    )
    add_shape(body=body, mesh=mesh, cfg=cfg, color=color, label=label)


def _add_sensor_to_world(builder, world_idx, env_position, env_rotation):
    """Per-world builder hook: add the sensor body for one environment.

    The cloner calls this once per environment, so the force-area shapes are laid
    out env-major (world 0's taxels, then world 1's, ...), the order the batched
    sensor wrapper expects.
    """
    if not _SENSOR_SHAPES:
        return

    env_pos = wp.vec3(float(env_position[0]), float(env_position[1]), float(env_position[2]))
    env_rot = wp.quat(
        float(env_rotation[0]), float(env_rotation[1]), float(env_rotation[2]), float(env_rotation[3])
    )
    # Place the sensor at the environment origin plus the local height offset,
    # carrying the reorienting mount rotation as its pose.
    body_pos = env_pos + wp.quat_rotate(env_rot, wp.vec3(0.0, 0.0, SENSOR_Z))
    body_rot = env_rot * MOUNT_ROTATION

    body = builder.add_body(
        xform=wp.transform(body_pos, body_rot), label="Sensor/Module", is_kinematic=True
    )
    for shape in _SENSOR_SHAPES:
        _add_force_area_shape(
            builder, body, shape, label=f"Sensor/Module/{shape['name']}"
        )
    # The rest of the sensor (board base, connectors, holder) as colliding meshes.
    for shape in _SENSOR_STRUCTURE:
        _add_structure_shape(
            builder, body, shape, label=f"Sensor/Module/{shape['name']}"
        )


def _install_sensor_hook():
    """Register the per-world sensor hook on the Newton cloner (idempotent)."""
    if not hasattr(NewtonManager, "_per_world_builder_hooks"):
        NewtonManager._per_world_builder_hooks = []
    if _add_sensor_to_world not in NewtonManager._per_world_builder_hooks:
        NewtonManager._per_world_builder_hooks.append(_add_sensor_to_world)


@clone
def spawn_sensor_frame(prim_path, cfg, translation=None, orientation=None):
    """Spawn a bare ``Xform`` at the per-env sensor path (no physics).

    The sensor's physics is built in Newton by the per-world hook, so this frame
    carries no colliders; it exists only for the wrapper to attach to.
    """
    from pxr import UsdGeom  # noqa: PLC0415

    stage = get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        UsdGeom.Xform.Define(stage, prim_path)
    return stage.GetPrimAtPath(prim_path)


@configclass
class SensorFrameCfg(SpawnerCfg):
    """Spawner cfg for :func:`spawn_sensor_frame` (a bare per-env Xform)."""

    func = spawn_sensor_frame


def _indenter_spawn():
    """Build the drop-object spawn cfg for the ``--object`` choice.

    Same mass and collision across shapes so the readouts are comparable; only
    the geometry (and thus the contact footprint on the pads) changes.
    """
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
    )
    if args_cli.object == "sphere":
        return sim_utils.SphereCfg(radius=0.006, **common)
    if args_cli.object == "cylinder":
        return sim_utils.CylinderCfg(radius=0.006, height=0.01, **common)
    return sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01), **common)


@configclass
class TactileSensorSceneCfg(InteractiveSceneCfg):
    """Scene: ground, light, a per-env sensor frame, a drop object, and the sensor."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Bare per-env frame; the sensor body itself is built in Newton by the
    # per-world hook (see ``_add_sensor_to_world``).
    sensor_frame = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Sensor",
        spawn=SensorFrameCfg(),
    )

    # Known-mass indenter dropped onto the force areas (which face +Z). A single
    # collider on its own body, so it imports through the normal USD path. Shape
    # and lateral offset come from the CLI (``--object`` / ``--offset_*``).
    indenter = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Indenter",
        spawn=_indenter_spawn(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(args_cli.offset_x, args_cli.offset_y, DROP_Z)
        ),
    )

    # The Synaptics tactile sensor wrapper. It binds to the boxes the per-world
    # hook added (by Newton shape label, not USD prims). No ``mount_rotation``:
    # the sensor body moves, so the kernel rotates each press axis by its live pose.
    tactile = CTSSensorCfg(
        prim_path="{ENV_REGEX_NS}/Sensor",
        update_period=0.0,  # update every step
        sensing_shape_pattern=SENSING_PATTERN,
        taxel_map=TAXEL_MAP,
        force_max=args_cli.force_max,
    )


def _save_heatmap(save_dir, force_arr, positions):
    """Save a force-field heatmap PNG for env 0 at its peak-force frame.

    A scatter of each taxel's world position (recentred) colored by its real
    normal force.
    """
    if not positions:
        print("[WARN] no positions_w (taxel_map required); skipping heatmap.")
        return
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("[WARN] matplotlib not installed; skipping heatmap (pip install matplotlib).")
        return

    pos_arr = np.stack(positions)  # (T, num_envs, taxels, 3)
    totals = force_arr[:, 0].sum(axis=1)  # env-0 total force per frame
    t = int(totals.argmax())
    xy = pos_arr[t, 0, :, :2]
    xy = (xy - xy.mean(axis=0)) * 1000.0  # recentre -> mm
    forces = force_arr[t, 0]

    fig, ax = plt.subplots(figsize=(5, 5))
    vmax = max(float(args_cli.force_max), float(forces.max()) or 1.0)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=forces, s=120, cmap="turbo", vmin=0.0, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"CTS force field (env0, peak @ step index {t})")
    fig.colorbar(sc, ax=ax, label="normal force [N]")
    out = os.path.join(save_dir, "heatmap.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]: Saved {out}")


def _save_artifacts(save_dir, steps, forces, totals, positions):
    """Dump the captured tactile signal (npy + csv, optional heatmap) to disk."""
    os.makedirs(save_dir, exist_ok=True)
    force_arr = np.stack(forces)  # (T, num_envs, taxels)
    total_arr = np.stack(totals)  # (T, num_envs, 3) net force vector [N]
    np.save(os.path.join(save_dir, "force.npy"), force_arr)
    np.save(os.path.join(save_dir, "total_force.npy"), total_arr)
    if positions:
        np.save(os.path.join(save_dir, "positions_w.npy"), np.stack(positions))

    with open(os.path.join(save_dir, "readout.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["step", "env", "total_force_N", "net_force_N", "active_taxels", "peak_taxel_N"]
        )
        for i, step in enumerate(steps):
            for e in range(force_arr.shape[1]):
                fe = force_arr[i, e]
                net = float(np.linalg.norm(total_arr[i, e]))
                writer.writerow(
                    [step, e, f"{fe.sum():.5f}", f"{net:.5f}",
                     int((fe > 1e-4).sum()), f"{fe.max():.5f}"]
                )

    print(
        f"[INFO]: Saved force.npy {force_arr.shape}, total_force.npy "
        f"{total_arr.shape} + readout.csv ({len(steps)} frames) to "
        f"{os.path.abspath(save_dir)}"
    )
    if args_cli.heatmap:
        _save_heatmap(save_dir, force_arr, positions)


def _viewer_open(sim):
    """True while at least one interactive viewer window is still open.

    Closing the Newton/Rerun window does NOT stop the Kit app
    (``simulation_app.is_running()`` stays True), so the keep-alive loop watches
    the viewer itself: the manager drops a closed viewer from ``sim._visualizers``,
    and a still-live viewer reports ``is_running()``.
    """
    for viz in getattr(sim, "_visualizers", []):
        viewer = getattr(viz, "_viewer", None)
        if viewer is None:
            continue
        is_running = getattr(viewer, "is_running", None)
        if is_running is None or is_running():
            return True
    return False


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Step the sim, print a per-env summary, and (optionally) capture artifacts."""
    sim_dt = sim.get_physics_dt()
    count = 0          # steps since the last reset (drives re-drops)
    total_steps = 0    # monotonic step counter (drives --steps + artifacts)
    save = bool(args_cli.save_dir)
    frames_force: list = []
    frames_total: list = []
    frames_pos: list = []
    frames_step: list = []

    while simulation_app.is_running():
        if args_cli.reset_interval > 0 and count % args_cli.reset_interval == 0:
            count = 0
            scene.reset()
            print("[INFO]: Reset scene (object re-dropped).")

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1
        total_steps += 1

        # Read the tactile signal exactly like any other Isaac Lab sensor.
        data = scene["tactile"].data
        f = data.force.numpy()  # (num_envs, num_taxels), per-taxel normal force [N]
        per_env = " | ".join(
            f"env{e}: {f[e].sum():6.2f}N act{int((f[e] > 1e-4).sum()):2d} pk{f[e].max():5.2f}N"
            for e in range(f.shape[0])
        )
        print(f"step {total_steps:6d} | {per_env}")

        if save:
            frames_force.append(f.copy())
            frames_total.append(data.total_force.numpy().copy())
            frames_step.append(total_steps)
            if data.positions_w is not None:
                frames_pos.append(data.positions_w.numpy().copy())

        if args_cli.steps > 0 and total_steps >= args_cli.steps:
            break

    if save and frames_force:
        _save_artifacts(
            args_cli.save_dir, frames_step, frames_force, frames_total, frames_pos
        )

    # With an interactive viewer up, don't tear the window down when the measured
    # run ends: keep stepping so the resting press stays live and inspectable
    # until the viewer window is closed.
    requested = getattr(args_cli, "visualizer", None) or []
    if requested and "none" not in requested:
        print(
            "[INFO]: Measured run complete; leaving the viewer open "
            "(close the window to exit)."
        )
        # Closing the window does NOT stop the Kit app, so watch the viewer
        # itself (see ``_viewer_open``) or this loop never ends.
        while simulation_app.is_running() and _viewer_open(sim):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)


def _viewer_cfgs():
    """Visualizer cfgs that start the selected interactive viewer at VIEWER_EYE.

    The Newton GL and Rerun viewers take their initial camera only from the
    visualizer cfg's ``eye``/``lookat`` (``sim.set_camera_view`` is a no-op for
    them), so the framing is attached to the cfg for whichever viewer ``--viz``
    requested. Nothing is added when no interactive viewer was requested.
    """
    requested = getattr(args_cli, "visualizer", None) or []
    cfgs = []
    if "newton" in requested:
        from isaaclab_visualizers.newton import NewtonVisualizerCfg  # noqa: PLC0415

        cfgs.append(NewtonVisualizerCfg(eye=VIEWER_EYE, lookat=VIEWER_LOOKAT))
    if "rerun" in requested:
        from isaaclab_visualizers.rerun import RerunVisualizerCfg  # noqa: PLC0415

        # Rerun ignores the cfg FOV, so VIEWER_EYE frames the scene more tightly
        # there. Keep the same view direction but pull the eye back along it by
        # RERUN_PULLBACK to match the Newton viewer.
        rerun_eye = tuple(
            look + RERUN_PULLBACK * (eye - look)
            for eye, look in zip(VIEWER_EYE, VIEWER_LOOKAT)
        )
        cfgs.append(RerunVisualizerCfg(eye=rerun_eye, lookat=VIEWER_LOOKAT))
    return cfgs


def _slow_viewer_navigation(sim):
    """Lower the Newton GL viewer's move/zoom speed after it is created.

    The GL viewer exposes navigation speed only as instance attributes, so they
    are set on the live viewer once the visualizers exist (after ``sim.reset``).
    Guarded with ``getattr``/``hasattr`` so a missing viewer or renamed attribute
    is a no-op, and so the Rerun viewer (which lacks these) is left untouched.
    """
    for viz in getattr(sim, "_visualizers", []):
        viewer = getattr(viz, "_viewer", None)
        if viewer is None:
            continue
        if hasattr(viewer, "_cam_speed"):
            viewer._cam_speed = VIEWER_MOVE_SPEED
        if hasattr(viewer, "_camera_dolly_scroll_sensitivity"):
            viewer._camera_dolly_scroll_sensitivity = VIEWER_ZOOM_SENSITIVITY
        if hasattr(viewer, "_camera_dolly_drag_sensitivity"):
            viewer._camera_dolly_drag_sensitivity = VIEWER_ZOOM_SENSITIVITY * 0.1


def main():
    # Read the force-area geometry once (UsdGeom only) and install the per-world
    # builder hook BEFORE the scene is cloned, so the cloner adds the sensor body
    # per environment.
    global _SENSOR_SHAPES, _SENSOR_STRUCTURE
    _SENSOR_SHAPES = load_force_area_shapes(SENSOR_USD)
    _SENSOR_STRUCTURE = load_structure_meshes(SENSOR_USD)
    print(
        f"Loaded {len(_SENSOR_SHAPES)} force areas + {len(_SENSOR_STRUCTURE)} "
        f"housing meshes from {os.path.basename(SENSOR_USD)}"
    )
    _install_sensor_hook()

    # physics=NewtonCfg() selects the Newton backend (the sensor is Newton-only).
    # njmax/nconmax give the MuJoCo-Warp buffers headroom: the defaults overflow
    # when the object presses many taxel boxes at once across envs. The solver
    # step (dt/num_substeps) must stay below ~sqrt(m/ke) for the stiff contact
    # against the rigid kinematic sensor to be stable; 480 Hz with 4 substeps works.
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 480.0,
        device=args_cli.device,
        physics=NewtonCfg(num_substeps=4, solver_cfg=MJWarpSolverCfg(njmax=512, nconmax=256)),
        visualizer_cfgs=_viewer_cfgs(),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(VIEWER_EYE), target=list(VIEWER_LOOKAT))

    scene_cfg = TactileSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    _slow_viewer_navigation(sim)
    print("[INFO]: Setup complete. Reading Synaptics tactile sensor data...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
