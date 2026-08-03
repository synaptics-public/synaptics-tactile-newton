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

"""Spawns a CTS sensor prim onto the stage.

The sensor is a plain ``Xform`` referencing the baked CTS USD, tagged with
``synaptics:*`` custom data. The runtime discovers sensors by scanning for that
tag, so a sensor survives save/reload of the stage without the extension having
to persist anything of its own.
"""

from pathlib import Path

from pxr import Gf, Usd, UsdGeom

from .config import load_model_config, resolve_model_assets

#: Custom-data namespace for everything this extension writes onto a prim.
#: USD reads ``:`` in a custom-data key as a nesting separator, so writing
#: ``synaptics:model`` produces ``customData = {"synaptics": {"model": ...}}``.
METADATA_NAMESPACE = "synaptics"

#: Custom-data key that marks a prim as a Synaptics tactile sensor.
ENABLED_KEY = f"{METADATA_NAMESPACE}:enabled"


def unique_prim_path(stage: Usd.Stage, parent_path: str, base_name: str) -> str:
    """``<parent>/<base_name>``, suffixed until it names a free prim."""
    parent = parent_path.rstrip("/")
    candidate = f"{parent}/{base_name}"
    suffix = 0
    while stage.GetPrimAtPath(candidate).IsValid():
        suffix += 1
        candidate = f"{parent}/{base_name}_{suffix}"
    return candidate


def _unit_scale(stage: Usd.Stage, asset_units_per_meter: float = 1.0) -> float:
    """Scale that makes a metre-authored asset correct on this stage.

    Isaac Sim's default stage is metres, where this returns 1.0. A centimetre
    stage returns 100.0. USD does not rescale across a reference, so without
    this the sensor is silently the wrong size.
    """
    stage_meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    return float(asset_units_per_meter / stage_meters_per_unit)


def spawn_sensor(
    stage: Usd.Stage,
    model_name: str,
    parent_path: str = "/World",
    extension_root: Path | None = None,
) -> tuple[str, dict]:
    """Reference the CTS asset under ``parent_path`` and tag it.

    Args:
        stage: The open USD stage.
        model_name: A model profile name, e.g. ``"CTS0.0"``.
        parent_path: Where to put the sensor — normally the current selection.
        extension_root: Extension root holding ``data/models``; defaults to this
            file's package.

    Returns:
        ``(sensor_prim_path, runtime_config)``.

    Raises:
        FileNotFoundError: unknown model, or the core package's assets are not
            installed (run the diagnostics for the fix).
        ValueError: malformed model profile.
    """
    config = load_model_config(extension_root, model_name)
    usd_path, taxel_map_path = resolve_model_assets(config)

    sensor_path = unique_prim_path(stage, parent_path, model_name.replace(".", "_"))
    sensor_xform = UsdGeom.Xform.Define(stage, sensor_path)
    sensor_prim = sensor_xform.GetPrim()
    sensor_prim.GetReferences().AddReference(str(usd_path))

    scale = _unit_scale(stage)
    if scale != 1.0:
        sensor_xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    runtime_config = {
        "model": model_name,
        "usd": str(usd_path),
        "taxelMap": str(taxel_map_path),
        "sensingShapePattern": str(config["sensing_shape_pattern"]),
        "forceMaxN": float(config["force_max_n"]),
        "numTaxels": int(config.get("num_taxels", 0)),
        "attachPrimPath": parent_path,
        "unitScale": scale,
    }
    _write_metadata(sensor_prim, runtime_config)

    return sensor_path, runtime_config


def _write_metadata(prim: Usd.Prim, runtime_config: dict) -> None:
    prim.SetCustomDataByKey(ENABLED_KEY, True)
    for key, value in runtime_config.items():
        prim.SetCustomDataByKey(f"{METADATA_NAMESPACE}:{key}", value)


def read_metadata(prim: Usd.Prim) -> dict | None:
    """Runtime config previously written onto ``prim``, or None if untagged."""
    data = prim.GetCustomDataByKey(METADATA_NAMESPACE)
    if not isinstance(data, dict) or not data.get("enabled"):
        return None
    return dict(data)


def find_sensor_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    """Every prim on the stage tagged as a Synaptics tactile sensor."""
    return [prim for prim in stage.Traverse() if read_metadata(prim) is not None]
