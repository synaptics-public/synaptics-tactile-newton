# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Robustness checks, headless.

Two checks that exercise code paths the dead-weight test never reaches:

**Contact-buffer headroom.** MuJoCo's defaults are ``njmax=1200 / nconmax=200``.
A press covering the whole pad generates far more contacts than the centred
10 mm drop, and a clipped contact reads as *missing force* rather than an
error — silently wrong in exactly the number this sensor exists to report. So
press every taxel at once and check the sum still equals the weight.

**Two sensors on one stage.** Newton labels shapes with their full USD prim
path, and the sensing glob is scoped per sensor prim so two sensors cannot claim
each other's taxels. This puts a second sensor beside the first and presses only
one of them.

Run it exactly like ``kit_diagnostics.py`` — same flags, with
``--exec .../scripts/kit_robustness.py``.
"""

import sys
import traceback

import carb
import omni.kit.app
import omni.timeline
import omni.usd

#: Enough for the drop plus contact settling at 1 kHz.
SETTLE_STEPS = 700

#: Sum(F) must land within this fraction of m*g.
TOLERANCE = 0.20

#: Full footprint [m] of the wide indenter: as broad as possible while staying
#: *inside* the taxel array. In the mounted frame the array spans x +-14.75 mm,
#: y +-6.45 mm, while the module base is coplanar with the force areas and
#: reaches x +-15.5 mm, y +-8.0 mm — so an indenter wider than the array rests
#: on the base and the taxels read almost nothing. That is the sensor behaving
#: correctly, not a contact-buffer failure, and it is why this footprint is
#: sized to the array rather than to the module.
WIDE_EXTENTS = (0.026, 0.011, 0.010)

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


def _play(steps: int) -> None:
    app = omni.kit.app.get_app()
    omni.timeline.get_timeline_interface().play()
    for _ in range(steps):
        app.update()


def _stop() -> None:
    app = omni.kit.app.get_app()
    omni.timeline.get_timeline_interface().stop()
    for _ in range(5):
        app.update()


def _record(label: str, passed: bool, detail: str) -> None:
    if not passed:
        _failures.append(label)
    print(f"[{'PASS' if passed else 'FAIL'}] {label} — {detail}")


def check_contact_buffer_headroom() -> None:
    """Press every taxel at once; the sum must still be the weight."""
    from synaptics.sensors.tactile.runtime import get_active_runtime
    from synaptics.sensors.tactile.scenario import build_scenario

    stage = _fresh_stage()
    info = build_scenario(stage, "CTS0.0", frame_camera=False, indenter_extents=WIDE_EXTENTS)
    runtime = get_active_runtime()
    if runtime is None:
        _record("contact-buffer headroom", False, "extension runtime not active")
        return
    _play(SETTLE_STEPS)

    if not runtime.is_bound:
        _record("contact-buffer headroom", False, f"never bound: {runtime.last_error}")
        _stop()
        return

    runtime.read_back()
    forces = runtime.bindings[0].latest_forces
    total = float(forces.sum())
    active = int((forces > 1e-4).sum())
    expected = info["expected_total_force_n"]
    _stop()

    ok = abs(total - expected) <= TOLERANCE * expected
    _record(
        "contact-buffer headroom",
        ok,
        f"{active}/{len(forces)} taxels active under a "
        f"{WIDE_EXTENTS[0] * 1000:.0f}x{WIDE_EXTENTS[1] * 1000:.0f} mm press, "
        f"Sum(F)={total:.4f} N vs m*g={expected:.4f} N "
        f"({abs(total - expected) / expected * 100:.1f}% off)",
    )


def check_two_sensors() -> None:
    """A second sensor must bind separately and read ~0 when untouched."""
    from pxr import Gf, UsdGeom

    from synaptics.sensors.tactile.runtime import get_active_runtime
    from synaptics.sensors.tactile.scenario import (
        SCENARIO_ROOT,
        SENSOR_Z,
        build_scenario,
        mount_rotation_xyzw,
    )
    from synaptics.sensors.tactile.spawn import spawn_sensor

    stage = _fresh_stage()
    info = build_scenario(stage, "CTS0.0", frame_camera=False)

    # A second sensor 60 mm to the side, mounted the same way up, well clear of
    # the indenter.
    second_path, _ = spawn_sensor(stage, "CTS0.0", parent_path=SCENARIO_ROOT)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(second_path))
    existing = xform.GetOrderedXformOps()
    translate = xform.AddTranslateOp(opSuffix="place")
    translate.Set(Gf.Vec3d(0.06, 0.0, SENSOR_Z))
    orient = xform.AddOrientOp(opSuffix="mount")
    x, y, z, w = mount_rotation_xyzw()
    orient.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xform.SetXformOpOrder([translate, orient] + list(existing))

    runtime = get_active_runtime()
    if runtime is None:
        _record("two sensors on one stage", False, "extension runtime not active")
        return
    _play(SETTLE_STEPS)

    if len(runtime.bindings) != 2:
        _record(
            "two sensors on one stage",
            False,
            f"expected 2 bindings, got {len(runtime.bindings)}: {runtime.last_error}",
        )
        _stop()
        return

    runtime.read_back()
    by_path = {b.prim_path: b for b in runtime.bindings}
    pressed = by_path[info["sensor_path"]]
    idle = by_path[second_path]
    pressed_total = float(pressed.latest_forces.sum())
    idle_total = float(idle.latest_forces.sum())
    expected = info["expected_total_force_n"]
    taxels_ok = pressed.num_taxels == idle.num_taxels == info["num_taxels"]
    _stop()

    ok = (
        taxels_ok
        and abs(pressed_total - expected) <= TOLERANCE * expected
        and idle_total < 1e-3
    )
    _record(
        "two sensors on one stage",
        ok,
        f"{pressed.num_taxels}+{idle.num_taxels} taxels; pressed "
        f"{pressed_total:.4f} N (expect {expected:.4f}), untouched "
        f"{idle_total:.6f} N",
    )


try:
    print(_BANNER)
    check_contact_buffer_headroom()
    check_two_sensors()
    print(_BANNER)
    if _failures:
        print(f"ROBUSTNESS FAILED: {', '.join(_failures)}")
        _status = 1
    else:
        print("ROBUSTNESS PASSED")
        _status = 0
except Exception as error:  # noqa: BLE001 - report, then still shut Kit down.
    carb.log_error(f"[Synaptics Tactile] kit_robustness failed: {error}")
    traceback.print_exc()
    _status = 3

sys.stdout.flush()
omni.kit.app.get_app().post_quit(_status)
