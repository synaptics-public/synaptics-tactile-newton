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

"""Dead-weight check, headless: drop a known mass, check Sum(taxel force) ~ m*g.

Builds the demo scene, arms the runtime, plays until the indenter settles, and
compares the summed per-taxel normal force against the analytic weight. This is
the Isaac Sim mirror of ``tests/test_dead_weight.py``.

Run it exactly like ``kit_diagnostics.py`` — same flags, with
``--exec .../scripts/kit_dead_weight.py``.
"""

import sys
import traceback

import carb
import omni.kit.app
import omni.timeline
import omni.usd

#: Free fall from the drop height plus contact settling, at 1 kHz.
SETTLE_STEPS = 600

#: Tolerance on Sum(F) vs m*g. Contact stiffness and a still-jittering box put
#: this a few percent off; the point is the force path is right, not that the
#: solver is perfectly converged.
TOLERANCE = 0.20

_BANNER = "=" * 72


def _build_scene():
    from pxr import UsdGeom

    from synaptics.sensors.tactile.scenario import build_scenario

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return build_scenario(stage, "CTS0.0", frame_camera=True)


def _main() -> int:
    from synaptics.sensors.tactile.runtime import TactileRuntime, get_active_runtime

    info = _build_scene()
    expected = info["expected_total_force_n"]
    print(
        f"scene: {info['num_taxels']} taxels, "
        f"{info['indenter_mass_kg'] * 1000:.0f} g indenter, expect ~{expected:.4f} N"
    )

    # Reuse the extension's runtime when it is loaded. Standing up a second
    # one makes both bind to the same model and build their own CTSSensor.
    runtime = get_active_runtime()
    own_runtime = runtime is None
    if own_runtime:
        runtime = TactileRuntime()
        runtime.start()

    app = omni.kit.app.get_app()
    omni.timeline.get_timeline_interface().play()
    for _ in range(SETTLE_STEPS):
        app.update()

    print(_BANNER)
    if not runtime.is_bound:
        print(f"FAIL: runtime never bound a sensor. {runtime.last_error}")
        if own_runtime:
            runtime.stop()
        return 1

    runtime.read_back()
    binding = runtime.bindings[0]
    forces = binding.latest_forces
    total = float(forces.sum())
    peak = float(forces.max())
    active = int((forces > 1e-4).sum())

    print(f"steps stepped     : {runtime.step_count}")
    print(f"taxels            : {binding.num_taxels}  (active: {active})")
    print(f"Sum(taxel force)  : {total:.4f} N")
    print(f"expected m*g      : {expected:.4f} N")
    print(f"peak taxel        : {peak:.4f} N")
    print(f"net |F| vector    : {binding.latest_total_force:.4f} N")
    if expected > 0:
        print(f"relative error    : {abs(total - expected) / expected * 100:.1f} %")

    if own_runtime:
        runtime.stop()

    if active == 0:
        print("FAIL: every taxel read zero — the sensor is not seeing contact.")
        return 1
    if abs(total - expected) > TOLERANCE * expected:
        print(f"FAIL: Sum(F) is outside +-{TOLERANCE * 100:.0f}% of m*g.")
        return 1
    print("PASS — summed taxel force matches the dropped weight.")
    return 0


try:
    print(_BANNER)
    _status = _main()
except Exception as error:  # noqa: BLE001 - report, then still shut Kit down.
    carb.log_error(f"[Synaptics Tactile] kit_dead_weight failed: {error}")
    traceback.print_exc()
    _status = 3

print(_BANNER)
sys.stdout.flush()
omni.kit.app.get_app().post_quit(_status)
