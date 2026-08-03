# Changelog

All notable changes to the Synaptics tactile sensor Isaac Sim extension.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

First release: spawn a CTS sensor in Isaac Sim, press it, and read per-taxel
force live.

### Added

- Kit extension `synaptics.sensors.tactile` for Isaac Sim 6.0.1 on the Newton
  backend. All sensor maths comes from the `synaptics_tactile_newton` package;
  the extension only binds it to Isaac Sim.
- `Create → Sensors → Synaptics Tactile Sensor → <model>` spawns the sensor
  under the selected prim and tags it with `synaptics:*` custom data, so a
  spawned sensor survives saving and reloading the stage.
- `Window → Synaptics Tactile Sensor` panel: `Load Scenario` builds a complete
  demo scene (ground, light, sensor mounted pads-up, 50 g indenter, framed
  camera); `Warm up kernels` compiles Warp's contact kernels on demand;
  `Play 1` / `Step 1` / `Step 5` walk the simulation a physics step at a time;
  `Reset` rewinds the scene and restores real-time playback.
- Live per-taxel heatmap: one cell per force area laid out from the taxel map,
  coloured on a scale that auto-ranges to the current peak, with saturated
  cells drawn distinctly. Cells scale with the panel and the panel scrolls.
- Per-step runtime binding `CTSSensor` to the live Newton model on Play and
  tearing it down on Stop. It refuses to arm on a solver that cannot produce
  contact forces, and reports why rather than reading zeros.
- Environment preflight reporting Isaac Sim build, Newton backend state,
  solver, `contacts.force` allocation, OpenUSD version and asset resolution —
  from the panel or headless.
- Five headless Kit harnesses under `scripts/`, each exiting non-zero on
  failure: preflight, spawn/panel smoke, scene bring-up, a dead-weight check
  (Σ taxel force ≈ *m·g*), and robustness (full-array press, two sensors on
  one stage).
- Sensor model profiles under `data/models/`; `CTS0.0` (52 taxels).
- Version-compatibility boundary under `adapters/`, and a per-Isaac-version
  dependency manifest under `packaging/isaac-6.0.1/`.

### Known limitations

- **Static / fixture mounts only.** Under OpenUSD < 26.5 the parallel physics
  parse corrupts the heap when one rigid body owns many colliders, and this
  module has 52. Robot-link mounting needs OpenUSD ≥ 26.5, arriving with
  Newton 1.5.0.
- **MuJoCo GPU solver only.** Isaac Sim only calls `solver.update_contacts()`
  on that path, so XPBD and MuJoCo-on-CPU produce no contact forces.
- **Presses wider than the taxel array under-read**, because the module base is
  coplanar with the force areas in the shipped asset and carries the load.
- A scene must contain at least one dynamic body: Newton builds no model at all
  when `body_count == 0`, which looks exactly like a broken sensor.
