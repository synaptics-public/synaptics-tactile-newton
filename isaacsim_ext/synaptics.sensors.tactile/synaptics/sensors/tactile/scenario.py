# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""One-click demo scenes: sensor, indenters, ground, light, camera.

Hand-assembling this scene is fiddly in ways that are easy to get silently
wrong — the sensor spawns on edge, the default cube is 30x the whole sensor,
and Isaac Sim's camera cannot focus on something 4 mm thick. Worse, Newton
returns **no model at all** from a fixture-only stage (it bails when
``builder.body_count == 0``), so a scene without a dynamic body looks exactly
like a broken sensor.

So the scene is built in code, once, and used by both the panel's *Load
Scenario* button and the headless test harness. The default scene's numbers
match ``examples/standalone_newton.py`` and ``tests/sensor_rig.py`` — a 50 g,
10 mm cube dropped on the pads — so a reading there is comparable to the pytest
suite.

Scenes beyond that one are presentation: several objects, staggered arrivals,
slow-motion playback. They differ from the default only in data — a list of
:class:`Indenter` specs plus a playback rate and a camera — so a new one is an
entry in :data:`SCENES` rather than new code.
"""

import math
from dataclasses import dataclass, replace

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from .config import load_model_config
from .spawn import METADATA_NAMESPACE, spawn_sensor

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

#: Top of the force areas [m]. The pads sit 1.28 mm above the sensor origin, so
#: this — not :data:`SENSOR_Z` — is what a fall distance is measured to.
PAD_SURFACE_Z = SENSOR_Z + 0.00128

#: Half-extent of the taxel array in the mounted frame [m].
#:
#: Overhanging it is not a soft edge: ``force_base`` is coplanar with the pads
#: and reaches further out, so an indenter hanging over the side rests on the
#: base instead. Measured at 3/52 taxels and 3 % of the weight.
TAXEL_ARRAY_HALF_X = 0.01475
TAXEL_ARRAY_HALF_Y = 0.00645

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

#: Kit refinement level for implicit surfaces (spheres, cylinders). Measured,
#: Kit's tessellation converges here: levels 3 through 8 render to within 0.2 %
#: of the same pixels, while level 0 is visibly a ~20-gon. So this is not a
#: quality dial with headroom above it — it is the ceiling, and anything
#: rounder needs real mesh geometry rather than a bigger number.
IMPLICIT_REFINEMENT_LEVEL = 2


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Indenter:
    """One object dropped onto the sensor.

    ``position`` is ``(x, y)`` in the mounted frame, which is also the panel's
    surface ``(u, v)``: a taxel drawn at the left of the heatmap is at -x. Both
    heights are to the object's **centre**, above :data:`SENSOR_Z`.
    """

    name: str
    position: tuple = (0.0, 0.0)
    drop_height: float = INDENTER_DROP_HEIGHT
    mass: float = INDENTER_MASS
    #: Full size [m] of a box indenter. Ignored when ``radius`` is set.
    extents: tuple = (
        INDENTER_HALF_EXTENT * 2.0,
        INDENTER_HALF_EXTENT * 2.0,
        INDENTER_HALF_EXTENT * 2.0,
    )
    #: Set to build a sphere instead of a box.
    radius: float | None = None
    #: Set alongside ``radius`` to build a cylinder of this length [m] instead
    #: of a sphere. Where a sphere bears on a point, a cylinder bears on a
    #: line, which is the whole reason to have one: it loads a row of taxels at
    #: once and its orientation is legible in the heatmap.
    length: float | None = None
    #: Heading [deg] this object travels along, matching its ramp's. A cylinder
    #: only rolls about an axis square to its direction of travel, so this is
    #: what squares it up; a sphere and a box do not care.
    heading_deg: float = 0.0
    #: Tilt [deg] of a cylinder's axis out of the horizontal. Zero lies it
    #: flat; positive dips the end that points along ``heading + 90``, so that
    #: end strikes first and the rest of the line slaps down after it.
    #: Cylinders only — a sphere has no axis and a box stays axis-aligned.
    pitch_deg: float = 0.0
    color: tuple = (0.9, 0.2, 0.2)
    #: Contact damping, authored as ``newton:contact_kd``. Newton turns ke/kd
    #: into MuJoCo's solref, where ``timeconst = 2/(kd*width)`` sets how quickly
    #: a contact is resolved — so kd is what decides whether a fast body is
    #: caught or thrown back off. Needed once an indenter gets small: these
    #: cubes hold 50 g in 6 mm, and at the default softness the higher one
    #: sinks in on impact and leaves the sensor entirely. None keeps the
    #: importer's default.
    contact_kd: float | None = None
    #: Contact stiffness, authored as ``newton:contact_ke``. Newton's default is
    #: 2.5e3 — 100x softer than every collider in ``cts0.0.usd``, which carries
    #: the 2.5e5 that docs/contact-stiffness.md settles on. MuJoCo mixes the
    #: pair's solref, so a default-soft indenter halves the stiffness of every
    #: pad contact it makes and sinks ~0.13 mm into a dimple of its own making.
    #: A resting object does not care — the contact still balances its weight —
    #: but a rolling one has to climb out of that dimple continuously, and the
    #: damping in it bleeds off the energy as drag. Measured on the sphere
    #: scene: 0.24 m/s^2 of it, enough to stop the ball mid-array. None keeps
    #: the importer's default.
    contact_ke: float | None = None

    def footprint_half(self) -> tuple:
        """Half-width in x and y of the contact patch this object rests on.

        A sphere touches at a point under its centre, so its radius does not
        widen where it bears — only a flat face does. A cylinder bears on a
        line: it is a point across its rolling direction and half its length
        along its axis, which sits square to the heading.
        """
        if self.radius is not None and self.length is not None:
            axis = math.radians(self.heading_deg + 90.0)
            half = self.length * 0.5
            return (abs(half * math.cos(axis)), abs(half * math.sin(axis)))
        if self.radius is not None:
            return (0.0, 0.0)
        return (self.extents[0] * 0.5, self.extents[1] * 0.5)

    def overhangs_taxels(self) -> bool:
        """True when this object would land partly on the coplanar base."""
        half_x, half_y = self.footprint_half()
        return (
            abs(self.position[0]) + half_x > TAXEL_ARRAY_HALF_X
            or abs(self.position[1]) + half_y > TAXEL_ARRAY_HALF_Y
        )


@dataclass(frozen=True)
class Ramp:
    """A static incline that gets an object moving sideways.

    Newton's USD importer reads position, rotation and the kinematic flag off a
    rigid body and ignores ``physics:velocity``, so an authored initial velocity
    is silently dropped — measured, a sphere given 0.06 m/s did not move. Height
    is the only input gravity will accept, so lateral motion has to be rolled
    down a slope.

    The ramp descends along ``heading_deg`` in the XY plane, meeting the pad
    surface at ``(low_x, low_y)``. Heading 0 runs straight down the array's long
    axis; a heading off zero sweeps the short axis as well, which is the only
    way a single pass says anything about the sensor's second dimension.
    """

    name: str
    #: Descent angle [deg]. Shallow enough and rolling resistance wins; the
    #: sphere scene is tuned to arrive at the pads at roughly 0.15 m/s.
    angle_deg: float = 5.0
    #: Horizontal length of the descent [m].
    run: float = 0.022
    #: Where the low end meets the pad surface.
    low_x: float = -0.016
    low_y: float = 0.0
    #: Compass heading of the descent [deg]: 0 is +x, 90 is +y.
    heading_deg: float = 0.0
    width: float = 0.014
    thickness: float = 0.004
    color: tuple = (0.35, 0.35, 0.38)

    def direction(self) -> tuple:
        """Unit ``(dx, dy)`` the object travels as it comes down."""
        heading = math.radians(self.heading_deg)
        return (math.cos(heading), math.sin(heading))

    def point_at(self, back: float) -> tuple:
        """``(x, y, z)`` on the top surface, ``back`` [m] horizontally uphill.

        ``back`` is measured along the ground, not along the slope, so the rise
        is ``back * tan(angle)`` — the same convention the flat-ramp version
        used when it subtracted two x coordinates.
        """
        dx, dy = self.direction()
        return (
            self.low_x - back * dx,
            self.low_y - back * dy,
            PAD_SURFACE_Z + back * math.tan(math.radians(self.angle_deg)),
        )

    def rest_height(self, radius: float) -> float:
        """Vertical gap from the surface up to a resting roller's centre.

        A roller sits a perpendicular ``radius`` off the slope, which is
        ``radius / cos(angle)`` measured straight up.
        """
        return radius / math.cos(math.radians(self.angle_deg))


@dataclass(frozen=True)
class Scene:
    """A named arrangement of indenters, plus how to play and frame it."""

    label: str
    description: str
    indenters: tuple
    ramps: tuple = ()
    #: Playback rate. Each rendered frame advances 1/this of sim time, so
    #: 480 against the default 60 is 8x slow motion.
    time_codes_per_second: float = DEFAULT_TIME_CODES_PER_SECOND
    #: MuJoCo contacts per world to ask Isaac Sim's solver for. Its default of
    #: 200 is not enough once a scene has more than one object touching the
    #: 52 pads, and the overflow is reported as a log line rather than an
    #: error while the dropped contacts read as missing force. None keeps
    #: whatever Isaac Sim is configured with.
    nconmax: int | None = None
    camera_eye: tuple = CAMERA_EYE
    camera_target: tuple = (0.0, 0.0, SENSOR_Z)
    #: Pin the heatmap's colour scale [N per taxel] instead of auto-ranging to
    #: each frame's peak. Auto-ranging is right for exploring one contact, wrong
    #: for filming two: the second object's impact peak rescales the grid and
    #: the first object — still pressing, unchanged — renders black. None keeps
    #: auto-ranging.
    force_scale_n: float | None = None
    #: Whether the panel's scene list offers this scene. The headless
    #: harnesses can still build a hidden scene by key.
    panel: bool = True


def fall_time(drop_height: float, half_height: float) -> float:
    """Free-fall time [s] from a centre height to first contact with the pads."""
    distance = SENSOR_Z + drop_height - half_height - PAD_SURFACE_Z
    return math.sqrt(2.0 * distance / GRAVITY) if distance > 0.0 else 0.0


#: Kit setting that makes one rendered frame advance exactly
#: ``1 / timeCodesPerSecond`` of sim time.
FIXED_TIME_STEPPING_SETTING = "/app/player/useFixedTimeStepping"


def set_playback_rate(stage: Usd.Stage, time_codes_per_second: float) -> None:
    """Set the playback rate, and make the rate actually govern stepping.

    Two things have to be true, and neither is guaranteed:

    * The rate must reach the **timeline**, not just the stage. The timeline
      picks it up an app update later, so reading it back immediately still
      reports the old value.
    * **Fixed time stepping must be on.** Without it the timeline advances by
      wall-clock instead, and ``timeCodesPerSecond`` changes nothing. The GUI
      Newton app enables it; ``isaacsim.exp.base.kit``, which the headless
      scripts run under, disables it — measured, a scene authored at 480 still
      stepped 1/60 s per frame there.

    Setting it here also makes the panel's Step buttons exact, since they retune
    the same rate to mean "one frame = N physics steps".

    Both halves are skipped outside Kit, where this module still has to import
    for the geometry to be testable.
    """
    stage.SetTimeCodesPerSecond(time_codes_per_second)
    try:
        import carb.settings
        import omni.timeline
    except ImportError:
        return
    carb.settings.get_settings().set(FIXED_TIME_STEPPING_SETTING, True)
    omni.timeline.get_timeline_interface().set_time_codes_per_second(time_codes_per_second)


def set_contact_buffer(nconmax: int) -> None:
    """Ask Isaac Sim's solver for ``nconmax`` contacts per world.

    A no-op outside Kit, where this module still has to import for the geometry
    to be testable.
    """
    try:
        from .adapters import get_newton_adapter

        get_newton_adapter().set_contact_buffer_size(nconmax)
    except ImportError:
        return


#: Camera for the drop scenes. Barely wider than the default preset — it only
#: has to reach from the ground to the high cube's 3 cm release, and framing any
#: looser shrinks the sensor for nothing.
_TALL_CAMERA_EYE = (0.038, -0.072, 0.049)
_TALL_CAMERA_TARGET = (0.0, 0.0, 0.020)

#: Presentation cubes are 6 mm, versus the 10 mm measurement cube. Length, not
#: volume — a 6 mm cube holding the same 50 g is ~230,000 kg/m^3.
_STAGGER_CUBE = 0.006

#: Contact damping for those cubes. A body this dense stops being caught by a
#: soft contact past roughly 0.9 m/s: it sinks into the pads on impact and is
#: thrown clear, and the scene settles at one cube's weight instead of two.
#: Measured on a 60 mm drop, which is twice what the scene now uses — so this is
#: headroom rather than a floor, and raising the drop heights stays safe.
#: Newton turns ke/kd into MuJoCo's solref as timeconst = 2/(kd*width), so kd is
#: the term that sets how fast a contact resolves.
_STAGGER_CONTACT_KD = 600.0

#: Two cubes at x = -+9 mm cover three columns each with a wide cold gap
#: between, and still clear the taxel array edge by 2 mm.
_STAGGER_X = 0.009

#: Contacts per world for the multi-object scenes. Isaac Sim's default of 200
#: overflows here — observed peaks around 330 — because 52 pads plus the housing
#: give a single landing many contacts at once. Overflow is only logged, and the
#: contacts it drops read as missing force.
#:
#: Do not raise this much further: at 1024 every contact force reads zero while
#: the physics stays correct and nothing reports an error. 768 still works.
_DEMO_NCONMAX = 512

#: Fixed heatmap scale [N per taxel] for the presentation scenes. A settled
#: cube peaks around 0.10-0.14 N per taxel, so this keeps a resting object
#: mid-ramp and lets an impact clamp to the top of the scale.
_DEMO_FORCE_SCALE_N = 0.18

_ROLLER_RADIUS = 0.004

#: The sphere crosses on a diagonal rather than straight down the long axis.
#: The array is 29.5 x 12.9 mm, so a straight run only ever excites the middle
#: row and demonstrates nothing about the short axis — half the sensor goes
#: unused in the one shot meant to show a contact moving over it.
#:
#: 18 deg sweeps +-4.8 mm of v across the run, about three quarters of the
#: 6.45 mm half-width. Corner to corner would be 23.6 deg, but that puts the
#: ramp's foot at |y| = 7 mm, off the side of the board with nothing under it.
#:
#: Negative heading: the ramp's foot sits at the north-west corner (-x, +y)
#: and the ball descends toward the south-east, instead of the original
#: south-west to north-east run.
_ROLLER_HEADING_DEG = -18.0

#: How far back along the heading the ramp's foot sits from the array centre.
#: Half the long axis is 14.75 mm, so this clears the pads by ~2 mm.
_ROLLER_FOOT = 0.017

_ROLLER_RAMP = Ramp(
    name="Ramp",
    heading_deg=_ROLLER_HEADING_DEG,
    low_x=-_ROLLER_FOOT * math.cos(math.radians(_ROLLER_HEADING_DEG)),
    low_y=-_ROLLER_FOOT * math.sin(math.radians(_ROLLER_HEADING_DEG)),
)

#: Released 18 mm up the ramp, which is a 1.6 mm descent — enough to reach the
#: pads at ~0.15 m/s and cross the array in about 0.2 s.
_ROLLER_START_BACK = 0.018
_ROLLER_START = _ROLLER_RAMP.point_at(_ROLLER_START_BACK)

#: The cylinder runs straight down the long axis instead. Its contact is a line
#: across the array's width, and keeping that line square to the run is what
#: makes it legible — a diagonal cylinder would hang its ends off both long
#: sides. Orientation is a parameter now, so a diagonal one is a number change
#: if it is ever wanted.
_BAR_RADIUS = 0.004
_BAR_LENGTH = 0.012
_BAR_RAMP = Ramp(name="BarRamp", low_x=-0.016, width=0.016)
_BAR_START = _BAR_RAMP.point_at(_ROLLER_START_BACK)

#: Contact pair for the roller, matching what every collider in ``cts0.0.usd``
#: carries — see the tuning table in docs/contact-stiffness.md.
#:
#: This is not presentation tuning; it is the difference between a ball and a
#: dead weight. At the importer's default ``ke`` the sphere left the ramp at
#: 113 mm/s and stopped at x=+7 mm — 8 mm short of the array edge — then rocked
#: in place for the rest of the take. Matching the pads removes 0.24 m/s^2 of
#: contact drag: it now crosses the whole array and rolls off the far end, and
#: the ride is 7x smoother (centre height wobbles 0.011 mm rather than 0.080).
_ROLLER_CONTACT_KE = 2.5e5
_ROLLER_CONTACT_KD = 1e3

#: The wire: the bar stretched thin — 2 mm across by 20 mm — and dropped
#: rather than rolled, its axis yawed 18 deg off the long axis and pitched
#: 18 deg out of the horizontal. One end strikes first and the contact grows
#: from a point into a diagonal line as the rest slaps down.
#:
#: Landed, the tips reach +-9.0 mm of u and +-2.9 mm of v — on taxel rows,
#: not just inside the array outline. That distinction is measured: at
#: 25 deg by 24 mm the tips landed at v = +-4.8 mm, in the chamfered
#: corners where the coplanar force_base carries the load instead of pads,
#: and the settled reading was 0.247 N of the wire's 0.491 N weight.
_WIRE_RADIUS = 0.001
_WIRE_LENGTH = 0.020
#: Compass direction [deg] of the wire's axis. ``Indenter.heading_deg`` is a
#: direction of travel with the axis square to it, so the scene authors
#: heading = axis - 90.
_WIRE_AXIS_DEG = 18.0
_WIRE_PITCH_DEG = 18.0
#: Centre release height above SENSOR_Z, placing the dipped tip 3 mm above
#: the pads. It arrives at ~0.24 m/s — well under the 0.9 m/s ceiling the
#: stagger cubes measured for a contact this damped to catch.
_WIRE_DROP = (
    (PAD_SURFACE_Z - SENSOR_Z)
    + 0.5 * _WIRE_LENGTH * math.sin(math.radians(_WIRE_PITCH_DEG))
    + _WIRE_RADIUS * math.cos(math.radians(_WIRE_PITCH_DEG))
    + 0.003
)

SCENES = {
    # Not offered in the panel: as a demo it is a cube sitting still. It stays
    # defined because it is the measurement scene — kit_dead_weight.py,
    # kit_scenario_test.py and kit_robustness.py all build it as the default
    # and compare its settled reading against m*g.
    "single_drop": Scene(
        label="Single drop (50 g cube)",
        description=(
            "One 10 mm, 50 g cube dropped centred. The only scene whose reading "
            "is verifiable: Sum(taxel force) settles at m*g = 0.4905 N."
        ),
        indenters=(Indenter(name="Indenter"),),
        panel=False,
    ),
    "two_cube_stagger": Scene(
        label="Two cubes, staggered",
        description=(
            "Two 50 g, 6 mm cubes released together onto opposite ends of the "
            "array. The blue one falls from twice the height, so it lands ~26 ms "
            "later; at 8x slow motion that reads as ~0.2 s apart."
        ),
        indenters=(
            Indenter(
                name="Indenter_Left",
                position=(-_STAGGER_X, 0.0),
                drop_height=0.015,
                extents=(_STAGGER_CUBE,) * 3,
                contact_kd=_STAGGER_CONTACT_KD,
                color=(0.90, 0.25, 0.20),
            ),
            Indenter(
                name="Indenter_Right",
                position=(_STAGGER_X, 0.0),
                drop_height=0.03,
                extents=(_STAGGER_CUBE,) * 3,
                contact_kd=_STAGGER_CONTACT_KD,
                color=(0.20, 0.45, 0.90),
            ),
        ),
        time_codes_per_second=480.0,
        nconmax=_DEMO_NCONMAX,
        camera_eye=_TALL_CAMERA_EYE,
        camera_target=_TALL_CAMERA_TARGET,
        force_scale_n=_DEMO_FORCE_SCALE_N,
    ),
    "two_larger_cube_stagger": Scene(
        label="Two larger cubes, staggered",
        description=(
            "Two 100 g, 8 mm cubes released together onto opposite ends of the "
            "array. The blue one falls from twice the height, so it lands ~26 ms "
            "later; at 8x slow motion that reads as ~0.2 s apart."
        ),
        indenters=(
            Indenter(
                name="Indenter_Left",
                position=(-_STAGGER_X, 0.0),
                drop_height=0.015,
                extents=(1.35 * _STAGGER_CUBE,) * 3,
                contact_kd=_STAGGER_CONTACT_KD,
                color=(0.90, 0.25, 0.20),
            ),
            Indenter(
                name="Indenter_Right",
                position=(_STAGGER_X, 0.0),
                drop_height=0.03,
                extents=(1.35 * _STAGGER_CUBE,) * 3,
                contact_kd=_STAGGER_CONTACT_KD,
                color=(0.20, 0.45, 0.90),
            ),
        ),
        time_codes_per_second=480.0,
        nconmax=_DEMO_NCONMAX,
        camera_eye=_TALL_CAMERA_EYE,
        camera_target=_TALL_CAMERA_TARGET,
        force_scale_n=_DEMO_FORCE_SCALE_N,
    ),
    "rolling_sphere": Scene(
        label="Rolling sphere (diagonal)",
        description=(
            "A 50 g, 8 mm sphere rolls down a ramp set 18 deg off the long "
            "axis, so its contact patch crosses the array corner to corner "
            "rather than straight down the middle. A small round patch that "
            "moves in both u and v."
        ),
        indenters=(
            Indenter(
                name="Roller",
                position=(_ROLLER_START[0], _ROLLER_START[1]),
                drop_height=_ROLLER_START[2]
                - SENSOR_Z
                + _ROLLER_RAMP.rest_height(_ROLLER_RADIUS)
                + 0.0003,
                radius=_ROLLER_RADIUS,
                heading_deg=_ROLLER_HEADING_DEG,
                contact_ke=_ROLLER_CONTACT_KE,
                contact_kd=_ROLLER_CONTACT_KD,
                color=(0.95, 0.75, 0.15),
            ),
        ),
        ramps=(_ROLLER_RAMP,),
        time_codes_per_second=480.0,
        nconmax=_DEMO_NCONMAX,
        force_scale_n=_DEMO_FORCE_SCALE_N,
    ),
    "rolling_cylinder": Scene(
        label="Rolling cylinder",
        description=(
            "A 50 g, 8 mm diameter by 12 mm cylinder rolls the length of the "
            "array. Where the sphere presses a point, this presses a line "
            "across the full width — the shot that shows the sensor resolving "
            "the orientation of what is touching it, not just its position."
        ),
        indenters=(
            Indenter(
                name="Bar",
                position=(_BAR_START[0], _BAR_START[1]),
                drop_height=_BAR_START[2]
                - SENSOR_Z
                + _BAR_RAMP.rest_height(_BAR_RADIUS)
                + 0.0003,
                radius=_BAR_RADIUS,
                length=_BAR_LENGTH,
                heading_deg=_BAR_RAMP.heading_deg,
                contact_ke=_ROLLER_CONTACT_KE,
                contact_kd=_ROLLER_CONTACT_KD,
                color=(0.25, 0.70, 0.55),
            ),
        ),
        ramps=(_BAR_RAMP,),
        time_codes_per_second=480.0,
        nconmax=_DEMO_NCONMAX,
        # A line contact spreads the same 0.49 N over a row rather than one or
        # two pads, so the per-taxel peak is well under the sphere's and the
        # sphere's scale would render this scene nearly black.
        force_scale_n=0.10,
    ),
    "wire_drop": Scene(
        label="Wire drop (awkward angle)",
        description=(
            "A 2 mm thick, 20 mm rigid wire dropped tilted, its axis 18 deg "
            "off the long axis. One end strikes first and the rest slaps "
            "down: a point that grows into a diagonal line carrying most of "
            "the wire's weight."
        ),
        indenters=(
            Indenter(
                name="Wire",
                drop_height=_WIRE_DROP,
                radius=_WIRE_RADIUS,
                length=_WIRE_LENGTH,
                heading_deg=_WIRE_AXIS_DEG - 90.0,
                pitch_deg=_WIRE_PITCH_DEG,
                contact_kd=_STAGGER_CONTACT_KD,
                color=(0.80, 0.35, 0.75),
            ),
        ),
        time_codes_per_second=480.0,
        nconmax=_DEMO_NCONMAX,
        # Longer line than the bar's, so the same 0.49 N spreads thinner
        # still; the bar's scale keeps it readable.
        force_scale_n=0.10,
    ),
}

#: ``single_drop`` stays the default ``build_scenario`` target because it is
#: the one scene whose reading is verifiable — ``kit_dead_weight.py`` and the
#: other harnesses compare it against m*g — even though the panel no longer
#: lists it.
DEFAULT_SCENE = "single_drop"


def scene_keys() -> list:
    """Panel scene keys, in declaration order. Hidden scenes stay buildable."""
    return [key for key, scene in SCENES.items() if scene.panel]


def scene_labels() -> list:
    """Human-readable scene labels, matching :func:`scene_keys` order."""
    return [SCENES[key].label for key in scene_keys()]


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
    scene: str = DEFAULT_SCENE,
) -> dict:
    """Build a demo scene under :data:`SCENARIO_ROOT`, replacing any previous one.

    Args:
        scene: a key of :data:`SCENES`.
        indenter_extents: optional ``(x, y, z)`` full size [m] overriding the
            **first** indenter's box. A footprint wider than the pad presses
            every taxel at once, which is what exercises the solver's contact
            buffer.

    Returns a dict describing what was built — prim paths plus the expected
    settled force — so callers can report and assert without re-deriving it.
    """
    if scene not in SCENES:
        raise ValueError(f"Unknown scene {scene!r}. Known: {sorted(SCENES)}")
    definition = SCENES[scene]

    indenters = list(definition.indenters)
    if indenter_extents is not None and indenters:
        indenters[0] = replace(
            indenters[0], extents=tuple(float(e) for e in indenter_extents), radius=None
        )

    clear_scenario(stage)
    set_playback_rate(stage, definition.time_codes_per_second)
    if definition.nconmax:
        set_contact_buffer(definition.nconmax)
    UsdGeom.Xform.Define(stage, "/World")
    root = UsdGeom.Xform.Define(stage, SCENARIO_ROOT).GetPrim()

    # Remember the scene on the prim rather than in a module global: Reset has to
    # restore this scene's playback rate, and a saved stage reopens knowing what
    # it is. clear_scenario wipes the root, so it cannot go stale.
    root.SetCustomDataByKey(f"{METADATA_NAMESPACE}:scene", scene)
    root.SetCustomDataByKey(
        f"{METADATA_NAMESPACE}:timeCodesPerSecond", float(definition.time_codes_per_second)
    )

    _ensure_physics_scene(stage)
    _add_ground(stage)
    _add_light(stage)

    sensor_path, sensor_config = _add_sensor(stage, model_name, extension_root)
    for ramp in definition.ramps:
        _add_ramp(stage, ramp)
    indenter_paths = [_add_indenter(stage, indenter) for indenter in indenters]

    if frame_camera:
        frame_sensor_camera(
            stage, eye=definition.camera_eye, target=definition.camera_target
        )

    total_mass = sum(indenter.mass for indenter in indenters)
    return {
        "root": SCENARIO_ROOT,
        "sensor_path": sensor_path,
        # Singular for the one-indenter callers that predate multi-object scenes.
        "indenter_path": indenter_paths[0] if indenter_paths else None,
        "indenter_paths": indenter_paths,
        "model": model_name,
        "scene": scene,
        "scene_label": definition.label,
        "num_taxels": int(sensor_config.get("numTaxels", 0)),
        "indenter_mass_kg": total_mass,
        "expected_total_force_n": total_mass * GRAVITY,
        "time_codes_per_second": float(definition.time_codes_per_second),
        "overhanging": [i.name for i in indenters if i.overhangs_taxels()],
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


def _add_ramp(stage: Usd.Stage, ramp: Ramp) -> str:
    """A static incline whose top face meets the pads at the ramp's foot.

    Rotating +theta about Y takes the box's +Z face normal to
    ``(sin theta, 0, cos theta)``, which is exactly the normal of a surface
    descending toward +x — so the tilt is the whole of the geometry, and a yaw
    about Z afterwards swings that descent onto the ramp's heading. The
    translate only has to put the face where it belongs.
    """
    path = f"{SCENARIO_ROOT}/{ramp.name}"
    angle = math.radians(ramp.angle_deg)
    dx, dy = ramp.direction()

    # Midpoint of the top face, then back off along its normal by half the slab.
    # The normal of a face descending along (dx, dy) is (sin a * dx, sin a * dy,
    # cos a), so the offset follows the heading rather than staying on x.
    mid_x, mid_y, mid_z = ramp.point_at(ramp.run * 0.5)
    centre = Gf.Vec3d(
        mid_x - ramp.thickness * 0.5 * math.sin(angle) * dx,
        mid_y - ramp.thickness * 0.5 * math.sin(angle) * dy,
        mid_z - ramp.thickness * 0.5 * math.cos(angle),
    )

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    cube.AddTranslateOp().Set(centre)
    # USD applies a translate/rotate/scale op list innermost-first, so this
    # reads bottom-up: scale the slab, tilt it about its own Y, swing it to the
    # heading, then move it into place.
    cube.AddRotateZOp().Set(float(ramp.heading_deg))
    cube.AddRotateYOp().Set(float(ramp.angle_deg))
    cube.AddScaleOp().Set(Gf.Vec3f(ramp.run, ramp.width, ramp.thickness))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*ramp.color)])

    # Collider only, like the ground: static geometry is immune to the
    # many-collider parse crash and needs no rigid body.
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return path


def _refine(prim: Usd.Prim, level: int = IMPLICIT_REFINEMENT_LEVEL) -> None:
    """Ask Kit's renderer to subdivide an implicit surface before drawing it.

    Newton collides a ``UsdGeomSphere`` as an exact analytic sphere, but Kit
    draws it as a polygon mesh it tessellates itself, and at the default
    refinement an 8 mm ball filmed from 60 mm away is visibly a faceted lump.
    These two attributes are Kit's per-prim override for that tessellation;
    they are renderer-only and never reach the solver.
    """
    prim.CreateAttribute(
        "refinementEnableOverride", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(True)
    prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(level)
    )


def _add_indenter(stage: Usd.Stage, indenter: Indenter) -> str:
    """A known-mass object above the pads — the dynamic body Newton requires."""
    path = f"{SCENARIO_ROOT}/{indenter.name}"
    position = Gf.Vec3d(
        float(indenter.position[0]),
        float(indenter.position[1]),
        SENSOR_Z + indenter.drop_height,
    )

    if indenter.radius is not None and indenter.length is not None:
        geom = UsdGeom.Cylinder.Define(stage, path)
        geom.GetRadiusAttr().Set(float(indenter.radius))
        geom.GetHeightAttr().Set(float(indenter.length))
        # Author the axis along X and yaw it a quarter turn off the heading:
        # a cylinder rolls about the axis square to where it is going, and
        # leaving the USD default of Z would stand it on end.
        geom.GetAxisAttr().Set(UsdGeom.Tokens.x)
        # Authored by hand: the static UsdGeomBoundable.ComputeExtent overload
        # is not exposed in this OpenUSD build, and an X-axis cylinder's bounds
        # are just half its length along x by its radius on the other two.
        half_len, r = float(indenter.length) * 0.5, float(indenter.radius)
        geom.CreateExtentAttr().Set(
            [Gf.Vec3f(-half_len, -r, -r), Gf.Vec3f(half_len, r, r)]
        )
        geom.AddTranslateOp().Set(position)
        geom.AddRotateZOp().Set(float(indenter.heading_deg) + 90.0)
        if indenter.pitch_deg:
            # After the yaw above, the axis lies along heading + 90; a positive
            # local-Y rotation dips that end below the centre.
            geom.AddRotateYOp().Set(float(indenter.pitch_deg))
        _refine(geom.GetPrim())
    elif indenter.radius is not None:
        geom = UsdGeom.Sphere.Define(stage, path)
        geom.GetRadiusAttr().Set(float(indenter.radius))
        geom.AddTranslateOp().Set(position)
        _refine(geom.GetPrim())
    else:
        geom = UsdGeom.Cube.Define(stage, path)
        extents = tuple(float(e) for e in indenter.extents)
        # A UsdGeomCube of size s spans -s/2..s/2, so size = 2 * half extent.
        #
        # Author the real size rather than scaling a 10 mm cube down to it. Both
        # give the same collider, but on Play the viewport stops drawing the USD
        # prim and starts drawing Newton's Fabric-published body transform, which
        # carries no scale — so a scaled cube visibly snaps back to its unscaled
        # size on the first frame. Only a non-uniform footprint still needs the
        # scale op, since a cube has one size for all three axes.
        if max(extents) - min(extents) < 1e-12:
            geom.GetSizeAttr().Set(extents[0])
            geom.AddTranslateOp().Set(position)
        else:
            side = INDENTER_HALF_EXTENT * 2.0
            geom.GetSizeAttr().Set(side)
            geom.AddTranslateOp().Set(position)
            geom.AddScaleOp().Set(Gf.Vec3f(*(e / side for e in extents)))

    geom.CreateDisplayColorAttr().Set([Gf.Vec3f(*indenter.color)])

    prim = geom.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(indenter.mass))
    if indenter.contact_kd is not None:
        prim.CreateAttribute("newton:contact_kd", Sdf.ValueTypeNames.Float).Set(
            float(indenter.contact_kd)
        )
    if indenter.contact_ke is not None:
        prim.CreateAttribute("newton:contact_ke", Sdf.ValueTypeNames.Float).Set(
            float(indenter.contact_ke)
        )
    return path


def frame_sensor_camera(
    stage: Usd.Stage,
    camera_path: str = "/OmniverseKit_Persp",
    eye=CAMERA_EYE,
    target=(0.0, 0.0, SENSOR_Z),
) -> bool:
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

    transform = (
        Gf.Matrix4d()
        .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        .GetInverse()
    )

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(*CAMERA_CLIPPING))
        xform = UsdGeom.Xformable(prim)
        # Kit's camera carries translate/rotateXYZ/scale ops; replacing the op
        # order with a single transform is simpler than decomposing to Euler,
        # and the manipulator re-authors its own ops on the next orbit anyway.
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(transform)
    return True


def active_scene(stage: Usd.Stage) -> str | None:
    """Key of the scene currently built on ``stage``, or None."""
    prim = stage.GetPrimAtPath(SCENARIO_ROOT)
    if not prim.IsValid():
        return None
    key = prim.GetCustomDataByKey(f"{METADATA_NAMESPACE}:scene")
    return key if key in SCENES else None


def reset_indenters(stage: Usd.Stage) -> int:
    """Put every indenter back at its authored drop height. Returns the count.

    Note this only rewrites the **authored** pose. Newton publishes simulated
    poses to Fabric and never writes them back to the prim's xform ops, so
    during playback the authored translate still reads as the drop height and
    this call changes nothing on its own — see :func:`reset_simulation`, which
    is what a "Reset" button actually needs.
    """
    scene = active_scene(stage)
    if scene is None:
        return 0

    moved = 0
    for indenter in SCENES[scene].indenters:
        prim = stage.GetPrimAtPath(f"{SCENARIO_ROOT}/{indenter.name}")
        if not prim.IsValid():
            continue
        position = Gf.Vec3d(
            float(indenter.position[0]),
            float(indenter.position[1]),
            SENSOR_Z + indenter.drop_height,
        )
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(position)
                moved += 1
                break
    return moved


def reset_simulation(stage: Usd.Stage) -> bool:
    """Return the scene to its authored initial state.

    **Stopping the timeline is the reset.** Newton destroys its model on Stop
    and rebuilds it from USD on the next Play, so that — not rewriting prim
    transforms — is what puts the indenters back. Rewriting the authored poses
    afterwards is belt-and-braces for the case where something moved them.

    Returns True when a scenario was present to reset.
    """
    import omni.timeline

    scene = active_scene(stage)
    if scene is None:
        return False

    omni.timeline.get_timeline_interface().stop()
    # Stepping retunes the timeline rate. Restore the scene's own rate, not real
    # time — a slow-motion scene is meant to stay slow across a Reset.
    set_playback_rate(stage, SCENES[scene].time_codes_per_second)
    reset_indenters(stage)
    return True
