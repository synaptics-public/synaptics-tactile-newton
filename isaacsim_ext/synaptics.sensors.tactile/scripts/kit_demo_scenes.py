# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Check the presentation scenes land where they are supposed to, headless.

These scenes exist to be filmed, and a take where one cube quietly missed the
taxel array looks almost right — the heatmap still lights up, just dimmer and
off to one side. That is expensive to notice after recording, so assert it here
instead: ~1 minute against the real solver, before anyone hits record.

What it checks, per scene:

**Two cubes, staggered.** Both cubes must press their own end of the array (a
force-weighted centroid near the authored x), the middle must stay cold, and the
left cube must arrive first by a margin large enough to read on screen. The
stagger comes from drop height alone — nothing schedules it — so it is worth
confirming rather than assuming.

**Rolling sphere.** The contact patch must actually *travel* on a diagonal, and
keep moving while it does. A sphere that bounces, wedges between pads or stalls still
reports force, and a static hot spot is exactly the shot this scene is not meant
to produce — as does one that grinds to a halt mid-array and rocks there, which
covers respectable ground if you only measure the distance.

**Rolling cylinder.** Same, plus the contact has to stay a *line*: a cylinder
that topples onto an end and skids still travels, and still reports force, but
what it draws on the heatmap is a wandering blob rather than the bar whose
orientation is the entire reason this scene exists.

**Wire drop.** The tilt exists to produce an arrival order: one end strikes
first, then the line fills in. So the first loaded frame must centre toward
the dipped end, the settled contact must span both axes the way a 20 mm
diagonal should, and the settled contact must carry most of the wire's
weight — a wire that bounced off the array or landed flat passes none of
those.

Run it exactly like ``kit_diagnostics.py`` — same flags, with
``--exec .../scripts/kit_demo_scenes.py``.
"""

import math
import statistics
import sys
import traceback

import carb
import omni.kit.app
import omni.timeline
import omni.usd

#: Frames to run per scene. One ``app.update()`` advances
#: ``1 / timeCodesPerSecond`` of sim time, and both scenes play at 480, so this
#: is ~1.0 s of sim — past the 152 ms drop with room for the sphere to roll.
FRAMES = 500

#: Frames to time the playback rate over. Long enough that a single frame's
#: jitter cannot move the average past the 5 % tolerance.
RATE_FRAMES = 60

#: A side counts as "in contact" above this [N] — 10 % of one cube's weight, so
#: impact ringing on the far side cannot trip it early.
CONTACT_N = 0.05

#: A single taxel counts as part of a contact line above this [N]. Well under
#: an even share of 0.49 N across a dozen pads, so the ends of the line still
#: register rather than only the middle where the load concentrates.
TAXEL_CONTACT_N = 0.005

#: Sum(F) must land within this fraction of the total weight. Wider than the
#: dead-weight test's: these cubes sit near the array edge, where a little load
#: reaches the coplanar base.
TOLERANCE = 0.25

#: A cube's measured contact centre must sit within this [m] of where it was
#: authored. Half a cube width — enough to catch "landed on the wrong end" or
#: "slid off the array", not a tuning constant.
CENTROID_TOLERANCE = 0.005

#: Taxels inside this |u| [m] are the middle of the array, between the two
#: cubes. They should stay cold.
MIDDLE_HALF_WIDTH = 0.004

#: The middle may carry at most this fraction of the total.
MIDDLE_MAX_SHARE = 0.15

#: The sphere's contact patch must advance at least this far [m]. The array is
#: 29.5 mm across and the ball is authored to cross all of it and run off the
#: far end; measured travel is ~27 mm, so this leaves room for solver jitter
#: without accepting a ball that only made it partway.
MIN_ROLL_TRAVEL = 0.022

#: ...and it must average at least this [m/s] doing it. Travel alone passes a
#: ball that creeps: before the roller's contact stiffness was matched to the
#: pads it covered 20 mm, but took 760 ms over it, stalling mid-array and
#: rocking in place — which on film is the opposite of the intended shot. With
#: the contact matched it crosses at ~124 mm/s, so this separates the two
#: cleanly rather than trimming a tuned value.
MIN_ROLL_SPEED = 0.06

#: The diagonal sphere must also move across the short axis [m]: two row
#: pitches (2.5 mm each), which rejects a run that stayed in the middle row
#: without judging how far into the corners it reached. Measured 6.7 mm on
#: Isaac Sim 6.0.1 and 5.9 mm on 6.1.0.
MIN_ROLL_TRAVEL_V = 0.005

#: Straight-line distance the diagonal sphere's patch must cover [m].
#:
#: The array is **not** a rectangle: its two outer rows (v = -4.82 and +5.19)
#: carry 8 taxels against the middle rows' 12, so the corners are chamfered and
#: hold no taxels at all. A diagonal therefore enters the sensing area around
#: u = -8.6 rather than -14.75, and can never cover the u distance a straight
#: run does -- it trades length for crossing all five rows. Judging it by u
#: alone would fail a run that is doing exactly what it should, so distance
#: travelled is measured along the path. Measured 18.7 mm.
MIN_ROLL_PATH = 0.015

#: The cylinder's contact must span at least this much v [m] at peak load. It
#: is 12 mm long across a 12.9 mm array, so a proper line contact covers most
#: of the width; anything much narrower means it landed on a corner, rolled up
#: on one end, or is being treated as a point.
MIN_LINE_SPAN_V = 0.007

#: The wire's first loaded frame must centre at least this far [m] toward its
#: dipped (+u) end. Its landed reach is +-9.0 mm of u, so 4 mm is comfortably
#: end-not-middle while allowing the line to have started filling in by the
#: time the total force clears CONTACT_N.
WIRE_STRIKE_MIN_U = 0.004

#: Settled spans [m] of the wire's contact. Landed it reaches 18.1 mm of u and
#: 5.9 mm of v tip to tip; asking for roughly half of the u reach and most of
#: the covered rows still rejects a wire resting on one end or lying square to
#: the long axis.
WIRE_MIN_SPAN_U = 0.010
WIRE_MIN_SPAN_V = 0.004

#: The settled line must carry at least this fraction of the wire's weight. A
#: wire that bounced off reads ~0; one whose end came to rest on the coplanar base
#: past the array edge still carries most of it (0.6 of the weight on Isaac Sim
#: 6.1.0, all of it on 6.0.1).
WIRE_MIN_WEIGHT_FRACTION = 0.5

#: How far the measured stagger may drift from the free-fall prediction. Wide,
#: because a threshold crossing lands after true first contact by a variable
#: margin — this is checking the drop heights still produce the gap they were
#: chosen for, not timing the solver.
STAGGER_TOLERANCE = 0.4

_BANNER = "=" * 72
_failures = []


def _fresh_stage():
    from pxr import UsdGeom

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _stop() -> None:
    app = omni.kit.app.get_app()
    omni.timeline.get_timeline_interface().stop()
    for _ in range(5):
        app.update()


def _record(label: str, passed: bool, detail: str) -> None:
    if not passed:
        _failures.append(label)
    print(f"[{'PASS' if passed else 'FAIL'}] {label} — {detail}")


def _surface_u(binding):
    """Taxel u coordinates [m] — surface +u is the mounted frame's +x."""
    from synaptics.sensors.tactile.taxel_layout import surface_coords

    return surface_coords(binding.sensor.taxel_centroids)[:, 0]


def _surface_v(binding):
    """Taxel v coordinates [m] — the array's short axis."""
    from synaptics.sensors.tactile.taxel_layout import surface_coords

    return surface_coords(binding.sensor.taxel_centroids)[:, 1]


def _centroid_u(forces, u, mask=None):
    """Force-weighted contact centre [m] over the masked taxels, or None."""
    weights = forces if mask is None else forces * mask
    total = float(weights.sum())
    if total <= 1e-6:
        return None
    return float((weights * u).sum() / total)


def _run(scene: str, frames: int = FRAMES):
    """Build ``scene``, play it, and return per-frame samples.

    Samples every frame rather than only at the end: the arrival order and the
    sphere's travel are both mid-flight facts that a settled reading has lost.
    """
    from synaptics.sensors.tactile.runtime import get_active_runtime
    from synaptics.sensors.tactile.scenario import build_scenario

    stage = _fresh_stage()
    info = build_scenario(stage, "CTS0.0", frame_camera=False, scene=scene)
    runtime = get_active_runtime()
    if runtime is None:
        return info, None, []

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Sim time comes from the timeline, not from frames * 1/timeCodesPerSecond.
    # Those disagree whenever the authored rate has not reached the timeline,
    # which is the exact failure this scene's slow motion depends on not having.
    samples = []
    for _ in range(frames):
        app.update()
        if not runtime.is_bound:
            continue
        runtime.read_back()
        forces = runtime.bindings[0].latest_forces
        if forces is not None:
            samples.append((timeline.get_current_time(), forces.copy()))
    return info, runtime, samples


def check_playback_rate() -> None:
    """A frame must advance exactly 1 / timeCodesPerSecond of sim time.

    Measured rather than read back, because the two disagree in the way that
    matters: the timeline happily reports 480 while still stepping 1/60 s per
    frame if fixed time stepping is off. Nothing else fails when that happens —
    the physics is identical — so the only symptom is a video that plays 8x too
    fast, discovered after recording it.
    """
    from synaptics.sensors.tactile.scenario import SCENES, build_scenario

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()

    problems = []
    details = []
    for scene, definition in SCENES.items():
        stage = _fresh_stage()
        build_scenario(stage, "CTS0.0", frame_camera=False, scene=scene)
        timeline.play()
        for _ in range(5):  # let the rate settle before timing anything
            app.update()

        start = timeline.get_current_time()
        for _ in range(RATE_FRAMES):
            app.update()
        elapsed = timeline.get_current_time() - start
        _stop()

        want = definition.time_codes_per_second
        got = RATE_FRAMES / elapsed if elapsed > 0 else 0.0
        details.append(f"{scene}={got:.0f}")
        if abs(got - want) > 0.05 * want:
            problems.append(f"{scene}: stepping at {got:.0f} fps, scene asked for {want:.0f}")

    _record(
        "one frame == 1/timeCodesPerSecond of sim time",
        not problems,
        "; ".join(problems) if problems else ", ".join(details),
    )


def check_two_cube_stagger() -> None:
    """Both cubes press their own end, in the right order, with a cold middle."""
    scene = "two_cube_stagger"
    info, runtime, samples = _run(scene)
    if runtime is None:
        _record(scene, False, "extension runtime not active")
        return
    if not samples:
        _record(scene, False, f"never bound: {runtime.last_error}")
        _stop()
        return

    u = _surface_u(runtime.bindings[0])
    left_mask = (u < 0).astype(float)
    right_mask = (u > 0).astype(float)
    middle_mask = (abs(u) < MIDDLE_HALF_WIDTH).astype(float)

    # First moment each side takes real load: the arrival the video is built on.
    left_at = next((t for t, f in samples if float((f * left_mask).sum()) > CONTACT_N), None)
    right_at = next((t for t, f in samples if float((f * right_mask).sum()) > CONTACT_N), None)

    _, final = samples[-1]
    _stop()

    total = float(final.sum())
    expected = info["expected_total_force_n"]
    left_u = _centroid_u(final, u, left_mask)
    right_u = _centroid_u(final, u, right_mask)
    middle_share = float((final * middle_mask).sum()) / total if total > 0 else 0.0

    if left_at is None or right_at is None:
        _record(
            scene,
            False,
            f"only one side ever loaded (left={left_at}, right={right_at}) — "
            "a cube missed the array or never landed",
        )
        return

    stagger = right_at - left_at
    indenters = {i.name: i for i in _scene_indenters(scene)}
    want_left = indenters["Indenter_Left"].position[0]
    want_right = indenters["Indenter_Right"].position[0]
    want_stagger = _predicted_stagger(indenters["Indenter_Left"], indenters["Indenter_Right"])

    problems = []
    if abs(total - expected) > TOLERANCE * expected:
        problems.append(f"Sum(F)={total:.4f} N vs {expected:.4f} N expected")
    if left_u is None or abs(left_u - want_left) > CENTROID_TOLERANCE:
        problems.append(f"left patch at {_mm(left_u)}, wanted {_mm(want_left)}")
    if right_u is None or abs(right_u - want_right) > CENTROID_TOLERANCE:
        problems.append(f"right patch at {_mm(right_u)}, wanted {_mm(want_right)}")
    if middle_share > MIDDLE_MAX_SHARE:
        problems.append(f"middle carries {middle_share * 100:.0f}% (want under "
                        f"{MIDDLE_MAX_SHARE * 100:.0f}%)")
    if stagger <= 0:
        problems.append(f"right cube landed first ({stagger * 1000:.0f} ms)")
    elif abs(stagger - want_stagger) > STAGGER_TOLERANCE * want_stagger:
        problems.append(f"stagger {stagger * 1000:.0f} ms, free fall predicts "
                        f"{want_stagger * 1000:.0f} ms")

    slow_motion = info["time_codes_per_second"] / 60.0
    detail = (
        f"patches at {_mm(left_u)} / {_mm(right_u)}, middle {middle_share * 100:.0f}%, "
        f"Sum(F)={total:.4f} N vs {expected:.4f} N, stagger {stagger * 1000:.0f} ms "
        f"vs {want_stagger * 1000:.0f} ms predicted "
        f"({stagger * slow_motion:.2f} s on screen at {slow_motion:.0f}x)"
    )
    _record(scene, not problems, detail if not problems else f"{'; '.join(problems)} [{detail}]")


def check_rolling_sphere() -> None:
    """The contact patch must travel, not sit still."""
    scene = "rolling_sphere"
    info, runtime, samples = _run(scene)
    if runtime is None:
        _record(scene, False, "extension runtime not active")
        return
    if not samples:
        _record(scene, False, f"never bound: {runtime.last_error}")
        _stop()
        return

    binding = runtime.bindings[0]
    u, v = _surface_u(binding), _surface_v(binding)
    loaded = [
        (t, _centroid_u(f, u), _centroid_u(f, v))
        for t, f in samples
        if float(f.sum()) > CONTACT_N and _centroid_u(f, u) is not None
    ]
    _stop()

    if len(loaded) < 2:
        _record(scene, False, "the sphere never made sustained contact")
        return

    first_t, first_u, first_v = loaded[0]
    last_t, last_u, last_v = loaded[-1]
    travel = last_u - first_u
    # Signed, not absolute: the ramp descends toward -v (north-west start), so
    # a run that drifted toward +v is a different failure from one that never
    # left the middle.
    travel_v = last_v - first_v
    elapsed = last_t - first_t

    path = math.hypot(travel, travel_v)
    speed = path / elapsed if elapsed > 0 else 0.0

    problems = []
    if path < MIN_ROLL_PATH:
        problems.append(
            f"travelled only {path * 1000:.1f} mm along its path "
            f"(want >= {MIN_ROLL_PATH * 1000:.0f} mm)"
        )
    if speed < MIN_ROLL_SPEED:
        problems.append(
            f"crossed at {speed * 1000:.0f} mm/s "
            f"(want >= {MIN_ROLL_SPEED * 1000:.0f} mm/s) — the ball is stalling, "
            f"not rolling"
        )
    if -travel_v < MIN_ROLL_TRAVEL_V:
        problems.append(
            f"crossed only {travel_v * 1000:+.1f} mm of v "
            f"(want <= {-MIN_ROLL_TRAVEL_V * 1000:.0f} mm) — the diagonal is not "
            f"reaching the short axis"
        )

    detail = (
        f"patch travelled u {_mm(first_u)} -> {_mm(last_u)} ({travel * 1000:+.1f} mm), "
        f"v {_mm(first_v)} -> {_mm(last_v)} ({travel_v * 1000:+.1f} mm), "
        f"{path * 1000:.1f} mm of path over {elapsed * 1000:.0f} ms "
        f"({speed * 1000:.0f} mm/s)"
    )
    _record(scene, not problems, detail if not problems else f"{'; '.join(problems)} [{detail}]")


def check_rolling_cylinder() -> None:
    """The bar must roll the length of the array, pressing a line as it goes.

    Travel alone would pass a cylinder that toppled onto one end and skidded,
    which reads on the heatmap as a wandering blob — so the span of the loaded
    taxels across v is checked at the same time. That span is the entire reason
    this scene exists rather than a second sphere.
    """
    scene = "rolling_cylinder"
    info, runtime, samples = _run(scene)
    if runtime is None:
        _record(scene, False, "extension runtime not active")
        return
    if not samples:
        _record(scene, False, f"never bound: {runtime.last_error}")
        _stop()
        return

    binding = runtime.bindings[0]
    u, v = _surface_u(binding), _surface_v(binding)
    loaded = [
        (t, f, _centroid_u(f, u))
        for t, f in samples
        if float(f.sum()) > CONTACT_N and _centroid_u(f, u) is not None
    ]
    _stop()

    if len(loaded) < 2:
        _record(scene, False, "the cylinder never made sustained contact")
        return

    first_t, _, first_u = loaded[0]
    last_t, _, last_u = loaded[-1]
    travel = last_u - first_u
    elapsed = last_t - first_t
    speed = travel / elapsed if elapsed > 0 else 0.0

    # Widest line it ever pressed, measured at each frame over the taxels
    # actually carrying load rather than at one hand-picked instant.
    span, shares, counts = 0.0, [], []
    for _, forces, _ in loaded:
        hot = forces > TAXEL_CONTACT_N
        if hot.sum() < 2:
            continue
        span = max(span, float(v[hot].max() - v[hot].min()))
        # Median, not max: the worst single frame is the moment it leaves the
        # ramp edge, where the load really does pile onto one pad. What matters
        # for the shot is how the load sits while it is rolling.
        shares.append(float(forces.max() / forces.sum()))
        counts.append(int(hot.sum()))
    share = statistics.median(shares) if shares else 1.0
    lit = statistics.median(counts) if counts else 0

    problems = []
    if travel < MIN_ROLL_TRAVEL:
        problems.append(
            f"travelled only {travel * 1000:+.1f} mm "
            f"(want >= {MIN_ROLL_TRAVEL * 1000:.0f} mm)"
        )
    if speed < MIN_ROLL_SPEED:
        problems.append(
            f"crossed at {speed * 1000:.0f} mm/s "
            f"(want >= {MIN_ROLL_SPEED * 1000:.0f} mm/s) — stalling, not rolling"
        )
    if span < MIN_LINE_SPAN_V:
        problems.append(
            f"widest contact spanned {span * 1000:.1f} mm of v "
            f"(want >= {MIN_LINE_SPAN_V * 1000:.0f} mm) — this is pressing a "
            f"point, not a line"
        )

    detail = (
        f"patch travelled {_mm(first_u)} -> {_mm(last_u)} ({travel * 1000:+.1f} mm) "
        f"at {speed * 1000:.0f} mm/s, line spanned {span * 1000:.1f} mm of v, "
        f"typically {lit} taxels lit with {share * 100:.0f}% on the hottest"
    )
    _record(scene, not problems, detail if not problems else f"{'; '.join(problems)} [{detail}]")


def check_wire_drop() -> None:
    """One end strikes first, then a diagonal line carrying most of the weight."""
    scene = "wire_drop"
    info, runtime, samples = _run(scene)
    if runtime is None:
        _record(scene, False, "extension runtime not active")
        return
    if not samples:
        _record(scene, False, f"never bound: {runtime.last_error}")
        _stop()
        return

    binding = runtime.bindings[0]
    u, v = _surface_u(binding), _surface_v(binding)
    loaded = [(t, f) for t, f in samples if float(f.sum()) > CONTACT_N]
    _stop()

    if len(loaded) < 10:
        _record(scene, False, "the wire never made sustained contact")
        return

    # The strike: the first loaded frame's force-weighted centre. Positive
    # pitch dips the +u end of the axis, so that is where it must land.
    first_u = _centroid_u(loaded[0][1], u)

    # The settled line, judged over the last quarter of the loaded frames.
    tail = loaded[-max(1, len(loaded) // 4):]
    span_u = span_v = 0.0
    totals = []
    for _, forces in tail:
        totals.append(float(forces.sum()))
        hot = forces > TAXEL_CONTACT_N
        if int(hot.sum()) >= 2:
            span_u = max(span_u, float(u[hot].max() - u[hot].min()))
            span_v = max(span_v, float(v[hot].max() - v[hot].min()))
    settled = statistics.median(totals)
    expected = info["expected_total_force_n"]

    problems = []
    if first_u is None or first_u < WIRE_STRIKE_MIN_U:
        problems.append(
            f"first contact centred at {_mm(first_u)} "
            f"(want >= {WIRE_STRIKE_MIN_U * 1000:.0f} mm) — it landed flat, "
            f"not end-first"
        )
    if span_u < WIRE_MIN_SPAN_U or span_v < WIRE_MIN_SPAN_V:
        problems.append(
            f"settled contact spans {span_u * 1000:.1f} x {span_v * 1000:.1f} mm "
            f"(want >= {WIRE_MIN_SPAN_U * 1000:.0f} x "
            f"{WIRE_MIN_SPAN_V * 1000:.0f} mm) — not the diagonal line"
        )
    if not WIRE_MIN_WEIGHT_FRACTION * expected <= settled <= (1.0 + TOLERANCE) * expected:
        problems.append(
            f"settled at {settled:.3f} N (want {WIRE_MIN_WEIGHT_FRACTION * expected:.3f}"
            f"–{(1.0 + TOLERANCE) * expected:.3f} N for a {expected:.3f} N wire) — it "
            f"bounced off, most of it rests on the base, or something else is on the array"
        )

    detail = (
        f"struck at {_mm(first_u)}, settled line spans {span_u * 1000:.1f} x "
        f"{span_v * 1000:.1f} mm at {settled:.3f} N (weight {expected:.3f} N)"
    )
    _record(scene, not problems, detail if not problems else f"{'; '.join(problems)} [{detail}]")


def _scene_indenters(scene: str):
    from synaptics.sensors.tactile.scenario import SCENES

    return SCENES[scene].indenters


def _predicted_stagger(first, second) -> float:
    """Gap [s] between two free falls, from their authored heights alone."""
    from synaptics.sensors.tactile.scenario import fall_time

    return fall_time(second.drop_height, second.extents[2] * 0.5) - fall_time(
        first.drop_height, first.extents[2] * 0.5
    )


def _mm(value) -> str:
    return "n/a" if value is None else f"{value * 1000:+.1f} mm"


try:
    print(_BANNER)
    check_playback_rate()
    check_two_cube_stagger()
    check_rolling_sphere()
    check_rolling_cylinder()
    check_wire_drop()
    print(_BANNER)
    if _failures:
        print(f"DEMO SCENES FAILED: {', '.join(_failures)}")
        _status = 1
    else:
        print("DEMO SCENES PASSED")
        _status = 0
except Exception as error:  # noqa: BLE001 - report, then still shut Kit down.
    carb.log_error(f"[Synaptics Tactile] kit_demo_scenes failed: {error}")
    traceback.print_exc()
    _status = 3

sys.stdout.flush()
omni.kit.app.get_app().post_quit(_status)
