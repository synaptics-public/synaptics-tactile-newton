# Changelog

All notable changes to the Synaptics tactile sensor Isaac Sim extension.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.1] - 2026-09-13

### Fixed

- A stage without a sensor is scanned once per Play, not on every physics
  step; a solver refusal is logged once.

### Changed

- Validated on Isaac Sim 6.1.0 (Newton 1.5.0) alongside 6.0.1; the demo-scene
  harness now judges the sphere by row pitch and the wire by carrying most of
  its weight.

## [1.0.0] - 2026-09-01

First release: spawn a CTS sensor in Isaac Sim, press it, and read per-taxel
force live. Isaac Sim 6.0.1 on the Newton backend; all sensor maths comes from
the `synaptics_tactile_newton` package.

### Added

- Example scenes: staggered cubes (two sizes), a diagonal rolling sphere, a
  rolling cylinder, and a tilted wire drop.
- `Load Scenario` builds a complete demo scene — ground, light, sensor mounted
  pads-up, indenters, camera — for the scene picked in the `Scene` dropdown.
- Diagnostics: an environment preflight (panel or headless) and headless Kit
  harnesses under `scripts/`, all exiting non-zero on failure.
- `Play 1` / `Step 1` / `Step 5` walk the simulation a physics step at a
  time; `Reset` rewinds the scene.
- Live per-taxel heatmap with an auto-ranging scale and distinct saturated
  cells, plus sensor path, readout mode, summed force and step count.
- Winkler readout correction, on by default. The engine's per-taxel split under
  a resting object is statically indeterminate — a flat cube reads a ragged
  profile that no contact tuning fixes — so the runtime redistributes it as a
  Winkler foundation per pressing body, preserving each body's engine total. It
  reads pressing-body geometry off the Newton model, so nothing needs
  configuring; a model with no supported pressing shape keeps the raw engine
  readout. The panel headline and the diagnostics summary both name the mode in
  force.
