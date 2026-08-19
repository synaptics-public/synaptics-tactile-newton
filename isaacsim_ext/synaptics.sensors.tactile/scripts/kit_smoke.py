# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Headless smoke test: spawn a sensor and build the panel, inside Kit.

Unlike ``kit_diagnostics.py`` this **mutates a scratch stage**, so run it
against a throwaway session only. It covers the two paths a read-only preflight
cannot reach: the ``Create → Sensors`` spawn action and the ``omni.ui`` panel's
build function.

Run it exactly like ``kit_diagnostics.py`` — same flags, with
``--exec .../scripts/kit_smoke.py``.
"""

import sys
import traceback

import carb
import omni.kit.app
import omni.usd

_BANNER = "=" * 72
_failures = []


def _check(label: str, fn):
    """Run ``fn``; record and report PASS/FAIL without aborting the run."""
    try:
        detail = fn()
    except Exception as error:  # noqa: BLE001 - one failing check must not hide the rest.
        _failures.append(label)
        print(f"[FAIL] {label}: {type(error).__name__}: {error}")
        traceback.print_exc()
        return None
    print(f"[PASS] {label}{f' — {detail}' if detail else ''}")
    return detail


def _new_stage():
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("new_stage() produced no stage")
    return "scratch stage created"


def _spawn():
    import re

    from pxr import Usd, UsdGeom

    from synaptics.sensors.tactile.spawn import find_sensor_prims, spawn_sensor

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    sensor_path, config = spawn_sensor(stage, "CTS0.0", parent_path="/World")

    prim = stage.GetPrimAtPath(sensor_path)
    if not prim.IsValid():
        raise RuntimeError(f"{sensor_path} is not a valid prim after spawn")

    # The force areas arrive through the reference, so counting them proves the
    # asset composed — and that the profile's taxel count matches the asset.
    force_area = re.compile(r"^forceArea_\d{3}$")
    found = sum(1 for child in Usd.PrimRange(prim) if force_area.match(child.GetName()))
    expected = int(config["numTaxels"])
    if found != expected:
        raise RuntimeError(
            f"profile declares {expected} taxels, asset composed {found} forceArea prims"
        )

    sensors = find_sensor_prims(stage)
    if len(sensors) != 1:
        raise RuntimeError(f"expected 1 tagged sensor prim, found {len(sensors)}")
    return f"{sensor_path}, {found} forceArea prims composed, 1 tagged prim"


def _panel():
    from synaptics.sensors.tactile.ui_builder import TactileSensorWindow

    window = TactileSensorWindow(
        models=["CTS0.0"],
        on_spawn=lambda model: f"spawn callback reached for {model}",
        on_diagnostics=lambda verbose: "diagnostics callback reached",
        on_load_scenario=lambda model: f"load-scenario callback reached for {model}",
        on_reset_scenario=lambda: "reset callback reached",
        on_read_forces=lambda: {"text": "readout reached", "sensors": [], "error": ""},
        on_warm_up=lambda: "warm-up callback reached",
        on_step=lambda n=1: f"step callback reached ({n})",
        on_play_one=lambda: "play-one callback reached",
    )
    try:
        window.toggle()
        if not window.visible:
            raise RuntimeError("window did not become visible")
        # Frame build functions run during app update, not on construction.
        for _ in range(3):
            omni.kit.app.get_app().update()
        # These are only populated by _build_frame, so a non-None model proves
        # the layout actually built.
        if window._model_combo is None or window._report_label is None:  # noqa: SLF001
            raise RuntimeError("panel widgets were not built")
        window._diagnostics_clicked(False)  # noqa: SLF001
        window._spawn_clicked()  # noqa: SLF001
        return "built, callbacks fire"
    finally:
        window.destroy()


try:
    print(_BANNER)
    _check("new stage", _new_stage)
    _check("spawn CTS0.0", _spawn)
    _check("panel builds", _panel)
    print(_BANNER)
    if _failures:
        print(f"SMOKE FAILED: {', '.join(_failures)}")
        _status = 1
    else:
        print("SMOKE PASSED")
        _status = 0
except Exception as error:  # noqa: BLE001 - report, then still shut Kit down.
    carb.log_error(f"[Synaptics Tactile] kit_smoke failed: {error}")
    traceback.print_exc()
    _status = 3

sys.stdout.flush()
omni.kit.app.get_app().post_quit(_status)
