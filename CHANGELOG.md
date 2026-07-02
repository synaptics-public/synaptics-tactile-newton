# Changelog

All notable changes to the `synaptics-tactile-newton` package are documented in
this file. This file is scoped to the **package** (what ships in the release
repo), not the playground/exploration work that lives alongside it here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/) (see
`pyproject.toml` for the current version).

## [Unreleased]

## [0.1.0] - 2026-07-02

### Added

- Initial release: `CTSSensor` / `CTSOutput` / `print_forces` core API on top
  of Newton.
- Optional Isaac Lab wrapper (`synaptics_tactile_newton.isaaclab`) behind the
  `isaaclab` extra.
- Baked CTS USD asset + taxel map shipped inside the package.
- Examples: `standalone_newton.py`, `isaaclab_task.py`, `isaaclab_task_demo.py`.
- Test suite covering dead-weight totals, per-taxel indentation/spatial
  response, force-area coverage, and saturation.
