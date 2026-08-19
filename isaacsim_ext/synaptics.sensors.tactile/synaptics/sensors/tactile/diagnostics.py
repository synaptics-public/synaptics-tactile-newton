# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Environment preflight for the Synaptics tactile Isaac Sim extension.

Answers "why is there no signal?" before anyone has to read a stack trace.
Runs three ways:

* CLI, from Isaac Sim's python::

      ./python.sh -m synaptics.sensors.tactile.diagnostics --verbose

* In-app, from the extension's panel (same report, rendered as text).
* Programmatically, via :func:`run_diagnostics` — a plain dict, so tests and
  CI can assert on it.

Imports nothing from ``omni`` at module scope: the CLI must work from a python
that has never booted Kit.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .adapters import get_newton_adapter

#: Isaac Sim builds this extension has been exercised against.
VALIDATED_ISAAC_BUILDS = ("6.0.1",)

#: The Newton physics backend first ships in Isaac Sim 6.0; 5.x is PhysX-only.
MINIMUM_ISAAC_VERSION = (6, 0)

#: The package holding every line of sensor math. The extension is a shell.
CORE_PACKAGE = "synaptics_tactile_newton"


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _python_record() -> dict:
    executable = sys.executable or ""
    looks_like_isaac = "isaac" in executable.lower() or "isaac" in sys.prefix.lower()
    return {
        "version": sys.version.split()[0],
        "executable": executable,
        "prefix": sys.prefix,
        "looks_like_isaac_python": looks_like_isaac,
        "in_kit": "omni.kit.app" in sys.modules,
    }


def _isaac_install_root() -> Path | None:
    """Best-effort path to the Isaac Sim install (the folder holding VERSION)."""
    isaac_path = os.environ.get("ISAAC_PATH", "").strip()
    if isaac_path and (Path(isaac_path) / "VERSION").is_file():
        return Path(isaac_path)

    try:
        import isaacsim

        start = Path(isaacsim.__file__).resolve().parent
    except Exception:  # noqa: BLE001 - probing must never raise.
        start = Path(sys.prefix)

    for candidate in (start, *start.parents):
        if (candidate / "VERSION").is_file() and (candidate / "exts").is_dir():
            return candidate
    return None


def _parse_isaac_version(raw: str) -> dict:
    """Split ``6.0.1-rc.7+release.42383.32955d8d.gl`` into its parts.

    The ``+release.…`` build metadata is the authoritative signal that a build
    is a shipped release; the ``-rc.N`` prerelease tag survives promotion and
    does **not** mean the build is a release candidate.
    """
    version, _, buildtag = raw.partition("+")
    core, _, prerelease = version.partition("-")
    parts = core.split(".")
    numeric = []
    for part in parts[:3]:
        try:
            numeric.append(int(part))
        except ValueError:
            numeric.append(0)
    while len(numeric) < 3:
        numeric.append(0)
    return {
        "raw": raw,
        "core": core,
        "prerelease": prerelease,
        "buildtag": buildtag,
        "major": numeric[0],
        "minor": numeric[1],
        "patch": numeric[2],
        "is_release_build": buildtag.startswith("release."),
    }


def _check_isaac_sim() -> dict:
    root = _isaac_install_root()
    if root is None:
        return {
            "available": False,
            "error": (
                "Could not find an Isaac Sim VERSION file. Run this from Isaac "
                "Sim's python (./python.sh) or set ISAAC_PATH."
            ),
        }

    raw = (root / "VERSION").read_text(encoding="utf-8").strip().splitlines()[0]
    record = _parse_isaac_version(raw)
    record["available"] = True
    record["path"] = str(root)
    record["meets_minimum"] = (record["major"], record["minor"]) >= MINIMUM_ISAAC_VERSION
    record["validated"] = record["core"] in VALIDATED_ISAAC_BUILDS
    return record


def _check_newton_adapter() -> dict:
    try:
        adapter = get_newton_adapter()
    except ImportError as error:
        return {"available": False, "error": str(error)}

    record = {"available": True}
    try:
        record.update(adapter.describe())
    except Exception as error:  # noqa: BLE001 - reporting must not raise.
        record["describe_error"] = str(error)
    return record


def _module_version(module_name: str) -> dict:
    try:
        module = __import__(module_name)
    except Exception as error:  # noqa: BLE001 - a broken install must report, not crash.
        return {"available": False, "error": f"{type(error).__name__}: {error}"}

    version = getattr(module, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as metadata_version

            version = metadata_version(module_name)
        except Exception:  # noqa: BLE001
            version = "unknown"
    return {
        "available": True,
        "version": str(version),
        "path": getattr(module, "__file__", "") or "",
    }


def _check_usd() -> dict:
    try:
        from pxr import Usd
    except Exception as error:  # noqa: BLE001
        return {"available": False, "error": f"{type(error).__name__}: {error}"}

    # Usd.GetVersion() reports OpenUSD 25.11 as (0, 25, 11) — the leading 0 is
    # the pre-1.0 major, so the year.month pair is elements 1 and 2.
    try:
        parts = tuple(int(part) for part in Usd.GetVersion())
    except Exception:  # noqa: BLE001 - older/newer pxr may not expose GetVersion.
        parts = ()

    if parts[:1] == (0,):
        version = ".".join(str(part) for part in parts[1:])
        comparable = parts[1:3]
    elif parts:
        version = ".".join(str(part) for part in parts)
        comparable = parts[0:2]
    else:
        version = "unknown"
        comparable = ()

    # OpenUSD < 26.5 carries the parallel physics-parse heap corruption that
    # limits us to statically mounted sensors. See docs/README.md.
    many_collider_crash = comparable < (26, 5) if comparable else None

    return {
        "available": True,
        "version": version,
        "many_collider_parse_crash": many_collider_crash,
        "work_thread_limit": os.environ.get("PXR_WORK_THREAD_LIMIT", "(unset)"),
    }


def _check_assets() -> dict:
    from .config import available_models, load_model_config, package_root, resolve_model_assets

    root = package_root()
    models = available_models(root)
    record = {"extension_root": str(root), "models": models, "available": bool(models)}
    if not models:
        record["error"] = f"No model profiles found under {root / 'data' / 'models'}."
        return record

    resolved = {}
    for model_name in models:
        try:
            config = load_model_config(root, model_name)
            usd_path, taxel_map_path = resolve_model_assets(config)
            resolved[model_name] = {
                "available": True,
                "usd": str(usd_path),
                "taxel_map": str(taxel_map_path),
                "num_taxels": config.get("num_taxels"),
                "force_max_n": config.get("force_max_n"),
            }
        except (FileNotFoundError, ValueError) as error:
            resolved[model_name] = {"available": False, "error": str(error)}
    record["resolved"] = resolved
    record["available"] = all(entry["available"] for entry in resolved.values())
    return record


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def run_diagnostics() -> dict:
    """Collect the full preflight report."""
    isaac_sim = _check_isaac_sim()
    adapter = _check_newton_adapter()
    packages = {
        "newton": _module_version("newton"),
        "warp": _module_version("warp"),
        CORE_PACKAGE: _module_version(CORE_PACKAGE),
        "usd": _check_usd(),
    }
    assets = _check_assets()

    passed = bool(
        isaac_sim.get("available")
        and isaac_sim.get("meets_minimum")
        and adapter.get("available")
        and packages["newton"]["available"]
        and packages["warp"]["available"]
        and packages[CORE_PACKAGE]["available"]
        and assets.get("available")
    )

    return {
        "passed": passed,
        "python": _python_record(),
        "isaac_sim": isaac_sim,
        "newton_backend": adapter,
        "packages": packages,
        "assets": assets,
    }


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "WARN" if warn else "PASS"
    return "FAIL"


def format_diagnostics(report: dict, verbose: bool = False) -> str:
    lines = [
        "Synaptics Tactile Sensor — Isaac Sim Diagnostics",
        "===============================================",
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        "",
    ]

    python = report["python"]
    lines += [
        f"[INFO] Python {python['version']}",
        f"  Executable: {python['executable']}",
    ]
    if verbose:
        lines.append(f"  Prefix: {python['prefix']}")
    lines.append(f"  Running inside Kit: {'yes' if python['in_kit'] else 'no'}")
    lines.append("")

    isaac = report["isaac_sim"]
    if not isaac.get("available"):
        lines += ["[FAIL] Isaac Sim", f"  {isaac.get('error')}", ""]
    else:
        ok = bool(isaac["meets_minimum"])
        lines += [
            f"[{_status(ok, warn=not isaac['validated'])}] Isaac Sim {isaac['core']}",
            f"  VERSION: {isaac['raw']}",
            f"  Build: {'release' if isaac['is_release_build'] else isaac['buildtag'] or 'unknown'}",
            f"  Path: {isaac['path']}",
        ]
        if not ok:
            lines.append(
                "  The Newton physics backend requires Isaac Sim "
                f"{MINIMUM_ISAAC_VERSION[0]}.{MINIMUM_ISAAC_VERSION[1]} or newer."
            )
        elif not isaac["validated"]:
            lines.append(
                f"  Not a validated build. Validated: {', '.join(VALIDATED_ISAAC_BUILDS)}."
            )
        lines.append("")

    backend = report["newton_backend"]
    if not backend.get("available"):
        lines += ["[FAIL] Newton backend", f"  {backend.get('error')}", ""]
    else:
        lines += [
            "[PASS] Newton backend",
            f"  Adapter: {backend.get('adapter')} (Isaac Sim {backend.get('isaac_version')}.x)",
            f"  Extension: {backend.get('extension_dependency')}",
            f"  Active physics engine: {backend.get('active_engine')}",
        ]
        if backend.get("ready"):
            lines += [
                f"  Simulation: initialized on {backend.get('device')} "
                f"(dt={backend.get('physics_dt')})",
                f"  Solver: {backend.get('solver_type')}",
                f"  contacts.force allocated: {backend.get('contact_force_attribute')}",
            ]
            if backend.get("contact_forces_available"):
                lines.append("  [PASS] Contact forces are readable on this solver.")
            else:
                lines.append(f"  [FAIL] {backend.get('contact_forces_reason')}")
        elif backend.get("init_failed"):
            lines.append(
                "  [FAIL] Newton initialization failed on this stage — Isaac Sim "
                "latched the error and disabled physics. Check the console for "
                "the USD parse error."
            )
        else:
            lines.append(
                "  Simulation: not initialized. Newton builds its model lazily on "
                "the first step after Play; solver and contact checks stay "
                "unknown until then."
            )
        lines.append("")

    packages = report["packages"]
    for name in ("newton", "warp", CORE_PACKAGE):
        record = packages[name]
        if record["available"]:
            lines.append(f"[PASS] {name} {record['version']}")
            if verbose:
                lines.append(f"  {record['path']}")
        else:
            lines.append(f"[FAIL] {name}: {record['error']}")
            if name == CORE_PACKAGE:
                lines.append(
                    "  All sensor math lives in this package. Install it into "
                    "Isaac Sim's python: "
                    "./python.sh -m pip install synaptics-tactile-newton"
                )

    usd = packages["usd"]
    if usd["available"]:
        crash = usd.get("many_collider_parse_crash")
        lines.append(f"[{_status(True, warn=bool(crash))}] OpenUSD {usd['version']}")
        if verbose or crash:
            lines.append(f"  PXR_WORK_THREAD_LIMIT: {usd['work_thread_limit']}")
        if crash:
            lines.append(
                "  OpenUSD < 26.5: parallel physics parsing corrupts the heap when "
                "one rigid body owns many colliders. Statically mounted sensors "
                "are unaffected; a robot-mounted CTS is not supported on this "
                "build (needs Newton 1.5.0)."
            )
    else:
        lines.append(f"[FAIL] OpenUSD: {usd['error']}")
    lines.append("")

    assets = report["assets"]
    lines.append(f"[{_status(bool(assets.get('available')))}] Sensor models")
    if assets.get("error"):
        lines.append(f"  {assets['error']}")
    for model_name, entry in (assets.get("resolved") or {}).items():
        if entry["available"]:
            lines.append(
                f"  {model_name}: {entry['num_taxels']} taxels, "
                f"force_max={entry['force_max_n']} N"
            )
            if verbose:
                lines.append(f"    USD:       {entry['usd']}")
                lines.append(f"    Taxel map: {entry['taxel_map']}")
        else:
            lines.append(f"  {model_name}: UNRESOLVED — {entry['error']}")

    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the environment for the Synaptics tactile Isaac Sim extension."
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Include paths and extra detail.")
    args = parser.parse_args(argv)

    report = run_diagnostics()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(format_diagnostics(report, verbose=args.verbose))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
