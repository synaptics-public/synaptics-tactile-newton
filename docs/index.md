# Synaptics Tactile Newton

`synaptics-tactile-newton` is a Newton-native model of the **Synaptics Capacitive
Tactile Sensor (CTS)**. It turns per-contact forces computed by the
[Newton](https://github.com/newton-physics/newton) physics engine into a
per-taxel normal-force readout you can use directly as a robot-learning
observation — on GPU, with no CPU round-trip.

The core package is Newton-only. An **optional** Isaac Lab wrapper is provided
for multi-environment reinforcement-learning tasks and is never pulled in by the
core install.

---

## What's in this package

```
synaptics_tactile_newton/        # the deliverable package
  sensor.py                      #   CTSSensor — the sensor model
  output.py                      #   CTSOutput — the per-taxel force bundle
  display.py                     #   format_forces / print_forces helpers
  kernels.py                     #   Warp signal-generation kernels
  isaaclab/                      #   OPTIONAL Isaac Lab wrapper (guarded import)
  assets/                        #   baked CTS USD geometry + taxel map
examples/                        # standalone Newton, Isaac Lab task, calibration
tests/                           # pytest suite
docs/                            # this document
```

Top-level API:

```python
from synaptics_tactile_newton import CTSSensor, CTSOutput, print_forces
```

- **`CTSSensor`** — attaches to a Newton `Model`, matches the sensor's force-area
  shapes, and each step reduces the contacts on those shapes to a per-taxel
  normal force.
- **`CTSOutput`** — the result bundle (all GPU `wp.array`):
  - `force` — per-taxel normal force `[N]`, shape `(num_taxels,)`
  - `total_force` — net force vector `[N]` on the sensing surface, shape `(3,)`
- **`print_forces` / `format_forces`** — pretty-print the taxel grid to the
  terminal.

The **baked CTS USD asset** and its **taxel map** (taxel names, centroids, press
axis) ship inside `synaptics_tactile_newton/assets/` and travel with the wheel.

---

## Getting started

The package runs on a CUDA GPU (Warp/Newton are GPU engines) and is only
tested on **Linux** (Ubuntu 24.04). Isaac Sim / Isaac Lab (needed for the
optional Isaac Lab example) do not support macOS at all, and this project's
setup script and docs assume a Linux shell — Windows is untested and not
supported here.

### Quick setup

`setup_newton.sh` creates a Python 3.12 virtual environment (`.venv-newton`),
installs Newton, the viewers, and this package in editable mode:

```bash
./setup_newton.sh
source .venv-newton/bin/activate
```

### Manual install

```bash
pip install synaptics-tactile-newton                 # core (Newton only)
pip install synaptics-tactile-newton[isaaclab]       # + Isaac Lab wrapper
pip install synaptics-tactile-newton[viewer]         # + Rerun web viewer
```

Extras: `isaaclab` (RL wrapper, needs Isaac Lab 3.0+), `viewer` (`rerun-sdk`),
`test` (`pytest`).

### Minimal usage

```python
from synaptics_tactile_newton import CTSSensor, print_forces

sensor = CTSSensor(
    model,                                   # a finalized Newton Model
    sensing_shape_pattern="*/forceArea_*",   # which shapes are taxels
    taxel_map="path/to/cts0.0_taxel_map.json",
    force_max=10.0,                          # per-taxel saturation [N]
)

# inside the sim loop, after solver + contact update:
sensor.update(state, contacts)
print_forces(sensor.data)                    # per-taxel grid + total
forces = sensor.data.force.numpy()           # (num_taxels,) normal force [N]
```

---

## Examples

All three live in `examples/`.

| Example | What it does |
|---|---|
| `standalone_newton.py` | Newton-only drop test. Loads the CTS sensor, drops a known-mass cube, and prints per-taxel / total forces. Rerun web viewer on by default (`--no-viewer` to disable). The fastest way to see the sensor work. |
| `isaaclab_task.py` | Multi-environment Isaac Lab task. Clones the sensor across environments and reads each environment's taxel forces independently — the RL-ready path. Requires Isaac Lab and a sourced Isaac Sim environment. |
| `calibration.py` | Calibration / validation harness: drops objects of known mass and compares measured per-taxel force against the expected `m·g`. |

Run the standalone example:

```bash
python examples/standalone_newton.py
python examples/standalone_newton.py --steps 400 --no-viewer
```

Run the Isaac Lab task with the Isaac Lab launcher (it sources Isaac Sim's
environment, which sets `EXP_PATH`) and your Newton venv active:

```bash
source .venv-newton/bin/activate
cd /path/to/IsaacLab            # must contain the _isaac_sim symlink
./isaaclab.sh -p /path/to/examples/isaaclab_task.py --num_envs 10 --headless
```

---

## Testing

```bash
pip install synaptics-tactile-newton[test]
pytest
```

The suite covers dead-weight totals, per-taxel indentation and spatial response,
force-area coverage, and saturation. Tests marked `sim` run a real Newton
simulation and need the bundled CTS USD.

---

## Limitations & known issues

- **GPU required.** Warp and Newton execute on CUDA; there is no CPU fallback for
  the sensor.

- **The Isaac Lab example builds the sensor from primitive boxes, not from a
  USD-referenced dynamic rigid body.** OpenUSD's parallel physics parse
  (`UsdPhysics.LoadUsdPhysicsFromRange`) has a thread-safety race that corrupts
  the heap when a single rigid body owns many (~30+) colliders — which the CTS
  sensor does (one body, ~50 force areas). Isaac Lab's per-environment cloner
  re-parses that body and reliably hits the race. The documented single-thread
  workaround (`PXR_WORK_THREAD_LIMIT=1`) **cannot be applied inside Isaac Sim/Kit**
  because a core Kit extension forces `PXR_WORK_THREAD_LIMIT=16` before USD
  initializes — see
  [isaac-sim/IsaacSim#692](https://github.com/isaac-sim/IsaacSim/issues/692). The
  example therefore reads the force-area geometry with `UsdGeom` bounds only and
  replays it as boxes on a Newton body, so no sensor physics schema is ever
  parsed. This is a solid, portable path; referencing the multi-collider sensor
  as a dynamic USD rigid body is not supported on the current stack.

- **The box-built sensor body is kinematic (perfectly rigid).** For stable
  contact the solver step must stay below roughly `sqrt(m / contact_ke)`; a body
  gets a per-world shape index (so forces report in every environment), but with
  a coarse step a light object will bounce instead of settling. Use a fine step,
  e.g. `SimulationCfg(dt=1/480)` with `NewtonCfg(num_substeps=4)`, and give the
  MuJoCo-Warp solver enough buffer headroom (`MJWarpSolverCfg(njmax=512,
  nconmax=256)`) when scaling up environments.

---

## Tested against

This release was validated on the following stack:

| Component | Version |
|---|---|
| Python | 3.12 |
| OS | Ubuntu 24.04 |
| Newton | 1.2.0 |
| Warp (`warp-lang`) | 1.13.0 |
| `usd-core` | 25.11 |
| Isaac Sim (for the Isaac Lab example) | 6.0.0 |
| Isaac Lab (for the Isaac Lab example) | 3.0 |
| GPU | NVIDIA RTX (CUDA) |

Other versions may work but are untested. The Isaac Sim / Isaac Lab versions
apply only to the optional Isaac Lab example; the core Newton package and the
standalone example do not need them.
