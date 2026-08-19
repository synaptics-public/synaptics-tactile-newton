# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Pytest fixtures and CLI options for the tactile-sensor validation suite.

These tests run the real Newton/MuJoCo solver against the CTS sensor USD, so
they are slow and need the ``newton`` stack (and ideally a CUDA device). They
are marked ``sim`` and skip cleanly when the stack or the USD asset is missing.

Run a subset::

    .venv-newton/bin/python -m pytest tests/test_dead_weight.py -v
    .venv-newton/bin/python -m pytest -m sim --device cuda:0
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

_ARTIFACTS = Path(__file__).parent / "_artifacts"

# Named sensor assets selectable via ``--asset``. Only one ships today; the
# actual paths are not resolved here (that would import warp/newton at
# collection time and break the graceful skip) — ``sensor_rig`` holds them.
ASSET_NAMES = ["generic_sensor"]
DEFAULT_ASSET = "generic_sensor"


def pytest_addoption(parser):
    group = parser.getgroup("tactile-validation")
    group.addoption("--device", default=None, help="Warp device, e.g. cuda:0 or cpu")
    group.addoption("--force-max", type=float, default=10.0, help="Taxel sat. [N]")
    group.addoption("--substeps", type=int, default=4)
    group.addoption("--dt", type=float, default=1.0 / 480.0)
    group.addoption("--nconmax", type=int, default=512, help="MuJoCo contacts/world")
    group.addoption(
        "--asset",
        default=DEFAULT_ASSET,
        choices=ASSET_NAMES,
        help=f"Named sensor asset to load (default: {DEFAULT_ASSET}).",
    )
    group.addoption("--sensor-usd", default=None, help="Override sensor USD path")
    group.addoption("--taxel-map", default=None, help="Override taxel-map path")
    group.addoption(
        "--viewer",
        action="store_true",
        help="Stream the run to the Newton web viewer (default: headless).",
    )
    group.addoption(
        "--gui",
        action="store_true",
        help="Use Newton's native OpenGL viewer window (ViewerGL) instead of the "
             "Rerun web viewer. Interactive (WASD/mouse camera, pause/step). "
             "Implies --viewer. Requires a display (X11 / RDP / local).",
    )
    group.addoption("--viewer-port", type=int, default=9090, help="Web viewer port")
    group.addoption("--viewer-rrd", default=None, help="Also record to this .rrd file")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "sim: slow Newton simulation test (needs newton + CTS USD)"
    )


@pytest.fixture(scope="session")
def sim_config(request):
    """Build a :class:`SimConfig` from CLI options, skipping if unavailable."""
    pytest.importorskip("warp", reason="warp not installed")
    pytest.importorskip("newton", reason="newton not installed")
    from sensor_rig import SimConfig, DEFAULT_SENSOR_USD, DEFAULT_TAXEL_MAP

    # ``sensor_rig``'s constants are the single source of truth for the paths.
    assets = {"generic_sensor": (DEFAULT_SENSOR_USD, DEFAULT_TAXEL_MAP)}

    opt = request.config.getoption
    asset_usd, asset_map = assets[opt("--asset")]
    usd = opt("--sensor-usd") or str(asset_usd)
    taxel_map = opt("--taxel-map") or str(asset_map)
    if not os.path.exists(usd):
        pytest.skip(f"sensor USD not found: {usd}")
    if not os.path.exists(taxel_map):
        pytest.skip(f"force-area/taxel map not found: {taxel_map}")

    return SimConfig(
        dt=opt("--dt"),
        substeps=opt("--substeps"),
        nconmax=opt("--nconmax"),
        force_max=opt("--force-max"),
        device=opt("--device"),
        sensor_usd=usd,
        taxel_map=taxel_map,
        viewer=opt("--viewer") or opt("--gui"),
        gui=opt("--gui"),
        viewer_port=opt("--viewer-port"),
        viewer_rrd=opt("--viewer-rrd"),
    )


@pytest.fixture
def make_rig(sim_config):
    """Factory returning a fresh, unfinalized ``SensorRig`` per call."""
    from sensor_rig import SensorRigFactory

    return SensorRigFactory(config=sim_config)


@pytest.fixture
def artifact_csv():
    """Return a writer ``fn(name, rows)`` that drops a CSV in ``tests/_artifacts``."""

    def _write(name: str, rows: list[dict]) -> Path:
        if not rows:
            return _ARTIFACTS / f"{name}.csv"
        _ARTIFACTS.mkdir(exist_ok=True)
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        path = _ARTIFACTS / f"{name}.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        return path

    return _write


@pytest.fixture(autouse=True, scope="session")
def _viewer_session(request):
    """When ``--viewer``/``--gui`` is set, keep the viewer open at the end of the
    session until the user closes it (Ctrl+C)."""
    yield
    viewer_on = request.config.getoption("--viewer") or request.config.getoption("--gui")
    if not viewer_on:
        return
    try:
        from sensor_rig import SensorRig

        viewer = getattr(SensorRig, "_VIEWER", None)
    except Exception:  # noqa: BLE001
        viewer = None
    if viewer is None:
        return

    if request.config.getoption("--gui"):
        # Native OpenGL window: pump frames so it stays interactive, replaying
        # the last logged state.
        print("\nGL viewer holding — close the window (or Ctrl+C) to exit.")
        last = getattr(SensorRig, "_LAST", None)
        try:
            while viewer.is_running():
                t = last[0] if last else 0.0
                viewer.begin_frame(t)
                if last:
                    viewer.log_state(last[1])
                    viewer.log_contacts(last[2], last[1])
                viewer.end_frame()
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()
        return

    import time

    port = request.config.getoption("--viewer-port")
    print(f"\nViewer holding at http://localhost:{port} — Ctrl+C to exit.")
    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

