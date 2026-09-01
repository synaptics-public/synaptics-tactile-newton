# Changelog

All notable changes to the `synaptics-tactile-newton` package are documented in
this file. It is scoped to the **package** — the sensor model, the Isaac Sim
extension, the examples and the test suite.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/) (see
`pyproject.toml` for the current version).

## [1.0.0] - 2026-09-01

- Add simple Winkler based model to provide better force data in Newton model

## [0.4.0] - 2026-08-28

- Add more scenarios to Isaac Sim extension: diagonal sphere, rolling cylinder, wire drop
- Heatmap tooltips: live force readout, fixed colors
- Add run_isaaclab_demo.sh: one-command Isaac Lab demo launcher
- Make the Kit extension self-contained to launch
- Pin warp-lang to the validated 1.13.0 in setup_newton.sh
- Cleanup documentation

## [0.3.0] - 2026-08-17

- Update license to BSD license.

## [0.2.0] - 2026-07-31

### Added

- **Isaac Sim Kit extension** (`isaacsim_ext/synaptics.sensors.tactile`), a shell
  over this package that brings the sensor into the Isaac Sim GUI: a
  `Create → Sensors` spawn menu, a one-click demo scene, a live per-taxel
  heatmap with saturation indication, single-physics-step transport controls,
  and a PASS/FAIL environment preflight. No signal maths lives in the extension.
  Requires Isaac Sim 6.0.1 on its Newton experience, MuJoCo GPU solver.
- Five headless Kit harnesses under the extension's `scripts/`, each exiting
  non-zero on failure: environment preflight, spawn/panel smoke, scene bring-up,
  a dead-weight check mirroring `tests/test_dead_weight.py`, and robustness
  (full-array press, two sensors on one stage).

### Documentation

- README: how to install the extension via Kit's extension search path, without
  copying anything into Isaac Sim's own `exts/`. Note that `PYTHONPATH` has no
  effect on Kit's embedded Python.
- README: presses wider than the taxel array under-read, because the module base
  is coplanar with the force areas in the shipped asset.
- README: the extension mounts the sensor statically, for the same OpenUSD
  multi-collider reason the Isaac Lab example builds from boxes.

## [0.1.1] - 2026-07-29

### Changed

- Default per-taxel saturation force (`force_max`) raised from 10 N to 100 N
  across the core sensor, the Isaac Lab config, and the examples to match
  expected specification of sensor.
- Isaac Lab batching now splits the core's env-major taxel force and world
  positions into the per-env buffers with on-GPU scatter kernels instead of a
  host round-trip, and the debug-vis markers reuse the world positions the
  sensor already computes on-GPU each step.
- Removed --debug_vis from IsaacLab demo due to complexity and not being useful.
- taxel_map.json stores units of "mm" inside of it.

### Documentation

- README: note that the Rerun web viewer needs **both** port 9090 (page) and
  9876 (gRPC scene stream) forwarded on remote/headless hosts, or it loads
  blank.

### Fixed

- In examples slow camera speeds are actually applied.

## [0.1.0] - 2026-07-02

### Added

- Initial release: `CTSSensor` / `CTSOutput` / `print_forces` core API on top
  of Newton.
- Optional Isaac Lab wrapper (`synaptics_tactile_newton.isaaclab`) behind the
  `isaaclab` extra.
- Baked CTS USD asset + taxel map shipped inside the package.
- Examples: `standalone_newton.py`, `isaaclab_task.py`.
- Test suite covering dead-weight totals, per-taxel indentation/spatial
  response, force-area coverage, and saturation.
