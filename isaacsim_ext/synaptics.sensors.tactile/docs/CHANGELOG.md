# Changelog

All notable changes to the Synaptics tactile sensor Isaac Sim extension.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

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
  cells, plus sensor path, summed force and step count.
