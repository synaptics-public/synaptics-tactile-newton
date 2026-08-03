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

"""Headless: build the demo scene, press Play, report what Newton produced.

This is the check a human cannot easily do by hand repeatedly. It answers, in
one run and with no GUI:

* does the whole stage — **52 static force-area colliders** — survive
  ``initialize_newton()``'s ``add_usd`` parse (the many-collider heap
  corruption, see docs/README.md "Known limitations")?
* does Newton build a model, and on which solver?
* is ``contacts.force`` allocated, i.e. is the force path actually reachable?

Run it exactly like ``kit_diagnostics.py`` — same flags, with
``--exec .../scripts/kit_scenario_test.py``.

Exit status is non-zero if the scene does not reach a state the sensor can bind to.
"""

import sys
import traceback

import carb
import omni.kit.app
import omni.timeline
import omni.usd

#: Physics steps to run after Play. The indenter needs ~0.08 s of free fall
#: from its drop height; this leaves margin for contact to settle.
STEPS = 240

_BANNER = "=" * 72


def _build_scene():
    from pxr import UsdGeom

    from synaptics.sensors.tactile.scenario import build_scenario

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    info = build_scenario(stage, "CTS0.0", frame_camera=True)
    print(
        f"scene: {info['num_taxels']} taxels, "
        f"{info['indenter_mass_kg'] * 1000:.0f} g indenter, "
        f"expect ~{info['expected_total_force_n']:.4f} N at rest"
    )
    return info


def _run_physics(steps: int) -> None:
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(steps):
        app.update()
    print(f"ran {steps} app updates with the timeline playing")


def _report(adapter) -> bool:
    record = adapter.describe()
    for key in (
        "ready",
        "init_failed",
        "active_engine",
        "solver_type",
        "device",
        "physics_dt",
        "contact_force_attribute",
        "contact_forces_available",
    ):
        print(f"  {key:26s}: {record.get(key)}")
    if record.get("contact_forces_reason"):
        print(f"  {'reason':26s}: {record['contact_forces_reason']}")

    model = adapter.model()
    if model is not None:
        print(f"  {'model bodies / shapes':26s}: {model.body_count} / {model.shape_count}")

    contacts = adapter.contacts()
    if contacts is not None:
        force = getattr(contacts, "force", None)
        print(f"  {'contacts.force':26s}: {None if force is None else force.shape}")

    if record.get("init_failed"):
        print("FAIL: Newton initialization failed — the USD parse threw and was latched.")
        return False
    if not record.get("ready"):
        print("FAIL: no Newton model after Play (body_count == 0, or physics never ran).")
        return False
    if not record.get("contact_forces_available"):
        print("FAIL: contact forces are not readable on this solver.")
        return False
    return True


try:
    print(_BANNER)
    _build_scene()

    from synaptics.sensors.tactile.adapters import get_newton_adapter

    adapter = get_newton_adapter()
    _run_physics(STEPS)

    print(_BANNER)
    print("Newton state after Play:")
    ok = _report(adapter)
    print(_BANNER)
    print("SCENARIO OK — the sensor can bind to this" if ok else "SCENARIO FAILED")
    _status = 0 if ok else 1
except Exception as error:  # noqa: BLE001 - report, then still shut Kit down.
    carb.log_error(f"[Synaptics Tactile] kit_scenario_test failed: {error}")
    traceback.print_exc()
    _status = 3

sys.stdout.flush()
omni.kit.app.get_app().post_quit(_status)
