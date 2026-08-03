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
isaacsim_ext/                    # OPTIONAL Isaac Sim Kit extension
  synaptics.sensors.tactile/     #   spawn menu, live taxel panel, diagnostics
examples/                        # standalone Newton, Isaac Lab task
tests/                           # pytest suite
README.md                        # this document
CHANGELOG.md                     # release history
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

The package runs on a CUDA GPU (Warp/Newton are GPU engines) and is only tested
on **Linux** (Ubuntu 24.04). Isaac Sim / Isaac Lab do not support macOS, and
Windows is untested and not supported here.

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
pip install synaptics-tactile-newton[viewer]         # + Rerun web viewer
pip install synaptics-tactile-newton[test]           # + pytest
```

Extras: `viewer` (`rerun-sdk`), `test` (`pytest`), `isaaclab` (RL wrapper). The
`isaaclab` extra needs Isaac Lab 3.0+, which is **not on PyPI yet**, so
`pip install synaptics-tactile-newton[isaaclab]` cannot resolve from an index
today — install Isaac Lab from a local source checkout instead (see
[Optional: Isaac Lab integration](#optional-isaac-lab-integration)).

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

## Optional: Isaac Lab integration

The `isaaclab_*` examples and the `synaptics_tactile_newton.isaaclab` wrapper run
the sensor across many parallel environments on the Isaac Lab **Newton backend**.
This path is optional and has extra prerequisites — the core Newton package and
the standalone example do **not** need any of it.

**Prerequisites (installed separately, not by `setup_newton.sh` by default):**

- A local checkout of **Isaac Lab 3.0+** (the first release with the Newton
  backend). 3.0 is not on PyPI yet, so it must be installed from source — not
  `pip install isaaclab`.
- A matching **Isaac Sim** with the `_isaac_sim` symlink in the Isaac Lab repo
  (its `setup_conda_env.sh` sets `EXP_PATH`, which the launcher needs).

**Install Isaac Lab into the same `.venv-newton`.** The example venv must contain
*both* this package and Isaac Lab. The simplest way is to point `setup_newton.sh`
at your Isaac Lab checkout:

```bash
./setup_newton.sh --with-isaaclab /path/to/IsaacLab
```

This runs the normal Newton setup, then installs the Isaac Lab `isaaclab`,
`isaaclab_ppisp`, `isaaclab_newton`, and `isaaclab_physx` extensions editable
into `.venv-newton`, plus `isaaclab_visualizers` so the demo's `--viz rerun` and
`--viz newton` viewers work out of the box. Note it pulls a large dependency
tree (PyTorch + CUDA) and may downgrade `warp-lang` / `usd-core` to the versions
Isaac Lab pins.

Already have a `.venv-newton`? Install them by hand instead:

```bash
source .venv-newton/bin/activate
pip install -e /path/to/IsaacLab/source/isaaclab \
             -e /path/to/IsaacLab/source/isaaclab_ppisp \
             -e /path/to/IsaacLab/source/isaaclab_newton \
             -e /path/to/IsaacLab/source/isaaclab_physx \
             -e /path/to/IsaacLab/source/isaaclab_visualizers
pip install PyOpenGL-accelerate      # only needed for the --viz newton window
```

---

## Optional: Isaac Sim extension

`isaacsim_ext/synaptics.sensors.tactile` is a Kit extension that brings the same
sensor into the Isaac Sim GUI: a `Create → Sensors` spawn menu, a one-click demo
scene, and a live per-taxel heatmap. It is a **shell over this package** — every
force value comes from `CTSSensor`, so the extension can never drift from the
model the tests cover.

Requires Isaac Sim **6.0.1** started on its **Newton** experience
(`./isaac-sim.newton.sh`); the default app disables the Newton backend and runs
PhysX.

**Install without copying anything into Isaac Sim.** Kit treats any folder as an
extension search path, so point it at this repo and leave the Isaac install
untouched:

```bash
# make this package importable inside Kit, then load the extension from here
<isaac-sim>/isaac-sim.newton.sh \
    --/app/python/extraPaths/0=/path/to/synaptics-tactile-newton \
    --ext-folder /path/to/synaptics-tactile-newton/isaacsim_ext \
    --enable synaptics.sensors.tactile
```

For a permanent install, `pip install` this package into Isaac Sim's python
(`<isaac-sim>/python.sh -m pip install .`) and add the `isaacsim_ext` folder
under *Window → Extensions → ⚙ → extension search paths*.

> `PYTHONPATH` has **no effect** — Kit's embedded Python ignores it. Copying the
> extension into Isaac Sim's own `exts/` folder also works but is not
> recommended: it duplicates the code and has to be redone on every upgrade.

Then `Window → Synaptics Tactile Sensor` → **Load Scenario** → **Play**.

Full documentation, including the diagnostics and the headless test harnesses,
is in
[`isaacsim_ext/synaptics.sensors.tactile/docs/README.md`](isaacsim_ext/synaptics.sensors.tactile/docs/README.md).

---

## Examples

All of these live in `examples/`.

| Example | What it does |
|---|---|
| `standalone_newton.py` | Newton-only drop test. Loads the CTS sensor, drops a known-mass cube, and prints per-taxel / total forces. Rerun web viewer on by default (`--no-viewer` to disable). The fastest way to see the sensor work. |
| `isaaclab_task.py` | Multi-environment Isaac Lab task. Clones the sensor across environments and reads each environment's taxel forces independently — the RL-ready path. Requires Isaac Lab and a sourced Isaac Sim environment. |
| `isaaclab_task_demo.py` | Richer, runnable Isaac Lab demo built on the box-built sensor body. Drops a selectable object (`--object cube/sphere/cylinder`, optionally off-center via `--offset_x/--offset_y`) onto the pads, prints a per-environment force summary, and optionally saves per-step artifacts (`force.npy`, `total_force.npy`, `positions_w.npy`, `readout.csv`) plus a force-field heatmap PNG. Also exposes `--force_max`, `--steps`, and `--save_dir`. |

Run the standalone example:

```bash
python examples/standalone_newton.py
python examples/standalone_newton.py --steps 400 --no-viewer
```

The Rerun web viewer serves its page on port **9090** but streams the actual
scene data over gRPC on port **9876**. When running on a remote/headless host
(SSH, VS Code Remote, a container), forward **both** ports to your local
machine or the viewer will load an empty page:

```bash
ssh -L 9090:localhost:9090 -L 9876:localhost:9876 user@host
```

(VS Code Remote usually auto-forwards 9090; add 9876 manually in the Ports
panel if the viewer stays blank.) Prefer no live viewer at all? Record to a file
with `--rrd out.rrd` and open it in the native Rerun desktop app.

Run the Isaac Lab task with the Isaac Lab launcher (it sources Isaac Sim's
environment, which sets `EXP_PATH`). This requires Isaac Lab installed into your
`.venv-newton` and a sourced Isaac Sim — see
[Optional: Isaac Lab integration](#optional-isaac-lab-integration):

```bash
source .venv-newton/bin/activate
cd /path/to/IsaacLab            # must contain the _isaac_sim symlink
./isaaclab.sh -p /path/to/examples/isaaclab_task.py --num_envs 10 --viz none
```

Run the richer demo (same launch requirements) with a selectable indenter
and artifact/heatmap capture:

```bash
./isaaclab.sh -p /path/to/examples/isaaclab_task_demo.py \
    --num_envs 4 --object sphere --offset_x 0.004 \
    --steps 600 --save_dir ./cts_demo_artifacts --heatmap --headless
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

The Isaac Sim extension has its own harnesses, which run **inside Kit** because
a bare `python.sh` cannot import `pxr`, `warp`, `newton` or the Newton backend.
Each exits non-zero on failure, so they double as CI gates — see
[the extension's README](isaacsim_ext/synaptics.sensors.tactile/docs/README.md):

| Harness | Checks |
|---|---|
| `kit_diagnostics.py` | environment preflight: versions, backend, solver, assets |
| `kit_smoke.py` | spawn a sensor, build the panel |
| `kit_scenario_test.py` | the demo scene reaches a state the sensor can bind to |
| `kit_dead_weight.py` | drop a known mass, Σ taxel force ≈ *m·g* |
| `kit_robustness.py` | full-array press; two sensors on one stage |

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

- **In the Isaac Sim extension the sensor is statically mounted**, for the same
  OpenUSD reason as above: static colliders are unaffected by that race at any
  count, so a bench-mounted sensor pressed by a moving indenter works today
  while a robot-link-mounted one does not. That needs OpenUSD ≥ 26.5, which
  arrives with Newton 1.5.0.

- **Presses wider than the taxel array under-read.** In the shipped asset the
  module base is coplanar with the force areas and extends further out, so a
  flat indenter overhanging the array rests on the base and the taxels barely
  register. Keep contact inside the array (x ±14.75 mm, y ±6.45 mm on CTS0.0).
  Real hardware has a rubber pad proud of the base, so this is an asset-fidelity
  gap rather than a physical one.

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
| Isaac Sim (Isaac Lab example) | 6.0.0 |
| Isaac Sim (Kit extension) | **6.0.1**, Newton experience |
| `isaacsim.physics.newton` (Kit extension) | 0.8.x |
| Isaac Lab (for the Isaac Lab example) | 3.0 |
| GPU | NVIDIA RTX (CUDA) |

Other versions may work but are untested. The Isaac Sim / Isaac Lab versions
apply only to the optional integrations; the core Newton package and the
standalone example do not need them.

The Kit extension needs Isaac Sim **6.0** or newer, because the Newton backend
(`isaacsim.physics.newton`) does not exist before it — 5.x is PhysX-only.
