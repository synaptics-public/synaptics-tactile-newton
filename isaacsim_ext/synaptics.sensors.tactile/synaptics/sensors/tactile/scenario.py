# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""One-click demo scene: sensor, indenter, ground, light, camera.

Hand-assembling this scene is fiddly in ways that are easy to get silently
wrong — the sensor spawns on edge, the default cube is 30x the whole sensor,
and Isaac Sim's camera cannot focus on something 4 mm thick. Worse, Newton
returns **no model at all** from a fixture-only stage (it bails when
``builder.body_count == 0``), so a scene without a dynamic body looks exactly
like a broken sensor.

So the scene is built in code, once, and used by both the panel's *Load
Scenario* button and the headless test harness. The numbers match
``examples/standalone_newton.py`` and ``tests/sensor_rig.py`` — a 50 g, 10 mm
cube dropped on the pads — so a reading here is comparable to the pytest suite.
"""

import math

from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from .config import load_model_config
from .spawn import spawn_sensor

#: Root of everything this module creates, so a rebuild can wipe it cleanly.
SCENARIO_ROOT = "/World/SynapticsTactileDemo"

#: The physics scene lives **outside** SCENARIO_ROOT deliberately. Isaac Sim's
#: SimulationManager caches a reference to it, so deleting it on every rebuild
#: produced "Removing stale physics scene reference (prim is no longer valid)".
#: It is also stage-wide state rather than something this demo owns.
PHYSICS_SCENE_PATH = "/World/PhysicsScene"

#: Sensor height above the ground plane [m]. Matches ``tests/sensor_rig.py``.
SENSOR_Z = 0.01

#: Physics rate [Hz], pinned rather than inherited. Newton's USD importer
#: defaults to 1000 when the scene does not say, which is easy to mistake for
#: Isaac's own 600 Hz config default — and the rate changes the forces, so it
#: belongs in the scene where it can be read.
PHYSICS_RATE_HZ = 1000.0

#: Timeline rate [fps]. Under fixed time stepping each frame advances
#: 1/this of sim time, so it is both the playback speed and the granularity of
#: a single Step. 60 is real-time; the Step buttons retune it deliberately.
DEFAULT_TIME_CODES_PER_SECOND = 60.0

#: Indenter: 10 mm cube, 50 g — the dead-weight test's mass, so Sum(taxel
#: force) should settle at m*g = 0.4905 N.
INDENTER_HALF_EXTENT = 0.005
INDENTER_MASS = 0.05
INDENTER_DROP_HEIGHT = 0.03

GRAVITY = 9.81

#: Ground: a wide, thin static box. A UsdGeomPlane has no thickness for the
#: collider to work with, and we only need something for a missed drop to land
#: on.
GROUND_HALF_EXTENT = 0.25
GROUND_THICKNESS = 0.01

#: Viewport camera preset for this sensor, from examples/standalone_newton.py.
CAMERA_EYE = (0.039113, -0.074668, 0.039714)

#: Isaac Sim's default near clip is 1 cm — wider than the sensor is thick, so
#: framing it clips it away entirely.
CAMERA_CLIPPING = (0.0001, 50.0)


def _mount_rotation() -> Gf.Quatf:
    """Rotation putting the sensor's +Y sensing face along the stage's +Z.

    The CTS asset is authored with its force areas on the +Y face (the plate is
    thin in Y), so referenced into a Z-up stage it stands on edge. +90 deg about
    X lays it flat, pads up — the same ``MOUNT_ROTATION`` the standalone example
    bakes.
    """
    half = math.pi / 4.0  # half of +90 deg
    return Gf.Quatf(math.cos(half), Gf.Vec3f(math.sin(half), 0.0, 0.0))


def mount_rotation_xyzw() -> tuple:
    """The mount rotation as ``(x, y, z, w)``, the order ``CTSSensor`` wants."""
    q = _mount_rotation()
    imag = q.GetImaginary()
    return (float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal()))


def clear_scenario(stage: Usd.Stage) -> bool:
    """Remove a previously built scenario. True when something was removed."""
    if not stage.GetPrimAtPath(SCENARIO_ROOT).IsValid():
        return False
    stage.RemovePrim(SCENARIO_ROOT)
    return True


def build_scenario(
    stage: Usd.Stage,
    model_name: str = "CTS0.0",
    extension_root=None,
    frame_camera: bool = True,
    indenter_extents=None,
) -> dict:
    """Build the demo scene under :data:`SCENARIO_ROOT`, replacing any previous one.

    Args:
        indenter_extents: optional ``(x, y, z)`` full size [m] for the indenter.
            Defaults to a 10 mm cube. A footprint wider than the pad presses
            every taxel at once, which is what exercises the solver's contact
            buffer.

    Returns a dict describing what was built — prim paths plus the expected
    settled force — so callers can report and assert without re-deriving it.
    """
    clear_scenario(stage)
    stage.SetTimeCodesPerSecond(DEFAULT_TIME_CODES_PER_SECOND)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, SCENARIO_ROOT)

    _ensure_physics_scene(stage)
    _add_ground(stage)
    _add_light(stage)

    sensor_path, sensor_config = _add_sensor(stage, model_name, extension_root)
    indenter_path = _add_indenter(stage, indenter_extents)

    if frame_camera:
        frame_sensor_camera(stage)

    return {
        "root": SCENARIO_ROOT,
        "sensor_path": sensor_path,
        "indenter_path": indenter_path,
        "model": model_name,
        "num_taxels": int(sensor_config.get("numTaxels", 0)),
        "indenter_mass_kg": INDENTER_MASS,
        "expected_total_force_n": INDENTER_MASS * GRAVITY,
        "mount_rotation_xyzw": mount_rotation_xyzw(),
    }


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #


def _ensure_physics_scene(stage: Usd.Stage) -> str:
    """Author a physics scene if the stage has none.

    Newton reads the timestep from here when present and falls back to its own
    600 Hz config otherwise; gravity has to be authored either way.
    """
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return str(prim.GetPath())

    path = PHYSICS_SCENE_PATH
    scene = UsdPhysics.Scene.Define(stage, path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(GRAVITY)
    # PhysxSchema ships with Kit, not core USD, so import it here: this module
    # stays importable (and its geometry testable) outside Isaac Sim.
    try:
        from pxr import PhysxSchema
    except ImportError:
        return path
    PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr().Set(
        PHYSICS_RATE_HZ
    )
    return path


def _add_ground(stage: Usd.Stage) -> str:
    path = f"{SCENARIO_ROOT}/GroundPlane"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -GROUND_THICKNESS * 0.5))
    cube.AddScaleOp().Set(
        Gf.Vec3f(GROUND_HALF_EXTENT * 2.0, GROUND_HALF_EXTENT * 2.0, GROUND_THICKNESS)
    )
    # Collider only, no RigidBodyAPI: static, and therefore not subject to the
    # many-collider parse crash.
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return path


def _add_light(stage: Usd.Stage) -> str:
    path = f"{SCENARIO_ROOT}/DistantLight"
    light = UsdLux.DistantLight.Define(stage, path)
    light.CreateIntensityAttr().Set(3000.0)
    light.CreateAngleAttr().Set(1.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))
    return path


def _add_sensor(stage: Usd.Stage, model_name: str, extension_root):
    """Spawn the sensor and lay it flat, pads up, clear of the ground."""
    sensor_path, sensor_config = spawn_sensor(
        stage, model_name, parent_path=SCENARIO_ROOT, extension_root=extension_root
    )
    prim = stage.GetPrimAtPath(sensor_path)
    xform = UsdGeom.Xformable(prim)

    # spawn_sensor may already have authored a scale op for a non-metre stage;
    # keep it and prepend placement so the order reads translate-rotate-scale.
    existing = xform.GetOrderedXformOps()
    translate = xform.AddTranslateOp(opSuffix="place")
    translate.Set(Gf.Vec3d(0.0, 0.0, SENSOR_Z))
    orient = xform.AddOrientOp(opSuffix="mount")
    orient.Set(_mount_rotation())
    xform.SetXformOpOrder([translate, orient] + list(existing))

    return sensor_path, sensor_config


def _add_indenter(stage: Usd.Stage, extents=None) -> str:
    """A known-mass box above the pads — the dynamic body Newton requires."""
    path = f"{SCENARIO_ROOT}/Indenter"
    cube = UsdGeom.Cube.Define(stage, path)
    # A UsdGeomCube of size s spans -s/2..s/2, so size = 2 * half extent; a
    # non-uniform footprint comes from scaling that unit cube.
    cube.GetSizeAttr().Set(INDENTER_HALF_EXTENT * 2.0)
    cube.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, SENSOR_Z + INDENTER_DROP_HEIGHT)
    )
    if extents is not None:
        side = INDENTER_HALF_EXTENT * 2.0
        cube.AddScaleOp().Set(Gf.Vec3f(*(float(e) / side for e in extents)))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.2, 0.2)])

    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(INDENTER_MASS)
    return path


def frame_sensor_camera(stage: Usd.Stage, camera_path: str = "/OmniverseKit_Persp") -> bool:
    """Point the viewport camera at the sensor and fix its clipping range.

    Authoring happens in the **session layer**. Kit defines the viewport camera
    there, and the session layer outranks the root layer — write to the default
    edit target instead and the opinion is composed away silently, leaving the
    camera exactly where it was.

    The default near clip is 1 cm, which is wider than this sensor is thick, so
    the clip fix matters as much as the placement: without it, framing the
    sensor puts it inside the near plane and it disappears.

    Returns False when the camera prim does not exist.
    """
    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        return False

    eye = Gf.Vec3d(*CAMERA_EYE)
    target = Gf.Vec3d(0.0, 0.0, SENSOR_Z)
    transform = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(*CAMERA_CLIPPING))
        xform = UsdGeom.Xformable(prim)
        # Kit's camera carries translate/rotateXYZ/scale ops; replacing the op
        # order with a single transform is simpler than decomposing to Euler,
        # and the manipulator re-authors its own ops on the next orbit anyway.
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(transform)
    return True


def reset_indenter(stage: Usd.Stage) -> bool:
    """Put the indenter back at its drop height. True when it was found.

    Note this only rewrites the **authored** pose. Newton publishes simulated
    poses to Fabric and never writes them back to the prim's xform ops, so
    during playback the authored translate still reads as the drop height and
    this call changes nothing on its own — see :func:`reset_simulation`, which
    is what a "Reset" button actually needs.
    """
    prim = stage.GetPrimAtPath(f"{SCENARIO_ROOT}/Indenter")
    if not prim.IsValid():
        return False
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(0.0, 0.0, SENSOR_Z + INDENTER_DROP_HEIGHT))
            return True
    return False


def reset_simulation(stage: Usd.Stage) -> bool:
    """Return the scene to its authored initial state.

    **Stopping the timeline is the reset.** Newton destroys its model on Stop
    and rebuilds it from USD on the next Play, so that — not rewriting prim
    transforms — is what puts the indenter back. Rewriting the authored pose
    afterwards is belt-and-braces for the case where something moved it.

    Returns True when a scenario was present to reset.
    """
    import omni.timeline

    omni.timeline.get_timeline_interface().stop()
    # Stepping retunes the timeline rate; Reset puts playback back to real time.
    stage.SetTimeCodesPerSecond(DEFAULT_TIME_CODES_PER_SECOND)
    return reset_indenter(stage)
