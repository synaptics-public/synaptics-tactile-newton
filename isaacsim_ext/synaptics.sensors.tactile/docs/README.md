# Synaptics Tactile Sensor (CTS) — Isaac Sim extension

A Kit extension that brings the Synaptics Capacitive Tactile Sensor into
Isaac Sim, bound to the **Newton** physics backend.

The extension is a shell. Every line of contact-to-signal math lives in the
[`synaptics_tactile_newton`](https://github.com/synaptics-public/synaptics-tactile-newton)
python package, which the Isaac Lab wrapper and the standalone Newton examples
use unchanged. That is deliberate: it keeps the sensor model usable outside
Isaac Sim and keeps a Kit version bump from touching sensor behaviour.

## Support matrix

| | Supported |
|---|---|
| Isaac Sim | **6.0.1** (validated). 6.0 is the floor — `isaacsim.physics.newton` does not exist in 5.x, which is PhysX-only. |
| Physics backend | Newton (`isaacsim.physics.newton` 0.8.x) |
| Solver | **MuJoCo, GPU.** Isaac Sim only calls `solver.update_contacts()` on that path, so XPBD and MuJoCo-on-CPU produce no contact forces. The extension refuses to arm on those rather than reporting zeros. |
| Sensor mount | **Static / fixture** (bench-mounted sensor, indenter presses it). A robot-link-mounted sensor is not supported yet — see [Known limitations](#known-limitations). |
| Newton | 1.2.x (1.2.1 ships with Isaac Sim 6.0.1) |

## Install

1. **Add this folder to Kit's extension search paths.** Either
   *Window → Extensions → ⚙ → add the folder that contains
   `synaptics.sensors.tactile`*, or pass it at launch:

   ```bash
   --ext-folder /path/to/synaptics-tactile-newton/isaacsim_ext --enable synaptics.sensors.tactile
   ```

   Loaded from a repo checkout, the extension puts the repo root on Kit's
   python path itself, so the core `synaptics_tactile_newton` package — the
   sensor model and the baked assets — resolves with no further setup. Only if
   the extension folder lives away from the repo, install the core package
   into Isaac Sim's python instead:

   ```bash
   cd <isaac-sim-root>
   ./python.sh -m pip install /path/to/synaptics-tactile-newton
   ```

   **`PYTHONPATH` has no effect** — Kit's embedded Python ignores it.
   `SYNAPTICS_TACTILE_ASSET_DIR` resolves the *assets* alone, which is not
   enough: the sensor model itself is imported from this package.

2. **Launch the Newton experience.** Isaac Sim's default app runs PhysX and
   disables the Newton backend, so start Isaac Sim with:

   ```bash
   ./isaac-sim.newton.sh
   ```

   (`apps/isaacsim.exp.full.newton.kit` — it enables `isaacsim.physics.newton`
   and disables the PhysX extensions.)

## Use

* **`Create → Sensors → Synaptics Tactile Sensor → CTS0.0`** references the
  sensor asset under the currently selected prim (or `/World`), and tags it
  with `synaptics:*` custom data. The tag is what the runtime looks for, so a
  spawned sensor survives saving and reloading the stage.
* **`Window → Synaptics Tactile Sensor`** opens the panel. **`Load Scenario`**
  is the fastest way in: it builds a complete, correct scene — ground, light,
  the sensor mounted pads-up, one or more indenters above it, and a camera
  that can actually resolve a 4 mm part — then you press **Play**. `Reset`
  stops the timeline and returns the indenters to their drop heights, ready for
  another Play.
* **The `Scene` dropdown** picks what `Load Scenario` builds:

  | Scene | What it does |
  |---|---|
  | `Two cubes, staggered` (default) | Two 50 g / 6 mm cubes onto opposite ends of the array. Released together, but the blue one falls from twice the height and lands ~35 ms later. |
  | `Two larger cubes, staggered` | The same shot with 100 g / 8 mm cubes, for a heavier reading. |
  | `Rolling sphere (diagonal)` | A 50 g / 8 mm sphere rolls down a ramp set 18° off the long axis, so its patch crosses the array corner to corner — moving in **both** u and v rather than straight down the middle row. |
  | `Rolling cylinder` | A 50 g / 8 mm × 12 mm cylinder rolls the length of the array, its axis square to its travel. Presses a **line** across the full width instead of a point: the shot that shows the sensor resolving what orientation something has, not just where it is. |
  | `Wire drop (awkward angle)` | A 2 mm × 20 mm rigid wire dropped tilted, its axis 18° off the long axis. One end strikes first, then the rest slaps down — a point that grows into a diagonal line, settling at *m·g*. |

  The measurement scene is no longer in the dropdown: one 50 g cube sitting
  still makes a poor demo, but it is the only scene whose reading is verifiable
  (Σ taxel force settles at *m·g* = 0.4905 N), so it survives as `single_drop`
  for the headless harnesses, which still build it as their default.

  The presentation scenes play at **8× slow motion**. The whole drop is over
  in about a tenth of a second at real speed, so at 60 fps it is a blink — which
  is also why they are no use as measurements. The staggered scenes do settle
  at 2·*m·g*, but only because they tune
  two things Isaac Sim's defaults get wrong for it: a stiffer
  `newton:contact_kd` on the cubes (holding 50 g in a 6 mm box makes them
  ~230,000 kg/m³, and at the default softness the higher cube sinks into the pads
  on impact and is thrown clear off the sensor), and a wider contact buffer
  (`nconmax = 512`; the default 200 overflows at ~330 contacts, which is logged
  but not raised, and the dropped contacts read as missing force). That is a
  tuned demo, not a measurement. The wire is denser still for its contact area,
  so it reuses the cubes' `contact_kd`.

  **The array is not a rectangle.** Its two outer rows (v = −4.82 and +5.19 mm)
  carry 8 taxels against the middle rows' 12, so the corners are chamfered and
  hold no taxels at all. A diagonal run therefore enters the sensing area at
  about u = −8.6 mm rather than −14.75, and covers less of the long axis than a
  straight one does — it trades length for crossing all five rows. That is why
  `kit_demo_scenes.py` judges the sphere by distance along its path rather than
  by u alone; measuring u would fail a run that is doing exactly what it should.

  The sphere scene needs a third: `newton:contact_ke = 2.5e5` on the ball, the
  same stiffness every collider in `cts0.0.usd` carries. Newton's importer
  defaults an indenter to `2.5e3`, and because MuJoCo mixes the pair's `solref`
  a default-soft ball halves the stiffness of every pad contact it makes. A
  resting object does not care — the contact still balances its weight — but a
  rolling one sinks ~0.13 mm into a dimple it has to climb out of continuously,
  and the damping in that contact bleeds off its energy: 0.24 m/s² of drag,
  which stopped the ball 8 mm short of the array edge and left it rocking in
  place. Note that this is *contact* stiffness, not the ball's own; the collider
  Newton builds from a `UsdGeomSphere` is analytic, never a faceted mesh.
* **The per-taxel heatmap** refreshes on its own (~10 Hz) once the timeline is
  playing — one cell per force area, laid out from the taxel map, coloured by
  force. The scale **auto-ranges to the current frame's peak** and prints its
  value, because a 100 N per-taxel limit against a ~0.04 N peak would render
  every cell black. Cells at the model's saturation limit are drawn in magenta
  and called out in the headline, so a clamped reading never passes for a
  correct one. Above the grid: the sensor path, summed force, net |F| and the
  step count — a settled 50 g object reads ~0.49 N there.
* **`Play 1` → `Step 1` / `Step 5`** walk through a contact event a physics step
  at a time. `Play 1` starts a stopped simulation and pauses after one step, so
  you are immediately ready to Step; `Step N` then advances exactly N steps.
  `Reset` restores real-time playback along with the scene.

  Isaac Sim has no control for this: per `omni.kit.loop-isaac`'s docs, "timeline
  tick advance and physics stepping are separate systems", so the Animation
  Timeline's next-frame button moves the playhead without stepping physics. A
  timeline frame is `1/timeCodesPerSecond` of sim time — 17 physics steps at the
  default 60 fps against a 1 kHz solver — so these buttons retune the timeline
  rate to make a frame equal the requested number of steps. That also leaves
  playback in slow motion, which is usually what you want while stepping.
* **`Warm up kernels`** compiles Warp's MuJoCo contact kernels on demand,
  instead of stalling your first Play for ~30 s.

**Stopping the timeline is what resets the scene.** Newton publishes simulated
poses to Fabric and never writes them back to the prim's transform, so rewriting
that transform mid-play changes nothing — the model has to be rebuilt from USD,
which is exactly what Stop → Play does.

The sensor binds on the **first physics step after Play** (Newton builds its
model lazily), and unbinds on Stop. Expect a ~30 s pause on the very first Play
of a session while Warp compiles the MuJoCo narrow-phase kernels.

`Add sensor only` places a bare sensor instead. Three things it leaves to you,
all of which `Load Scenario` handles:

* **Orient it.** The asset's sensing face points along +Y, so on a Z-up stage it
  spawns standing on edge. Set `Rotate X = 90` so the pads face up.
* **You will not see it** at the default camera — it is 31 × 16 × 4 mm, and
  Isaac Sim's viewport camera has a **1 cm near clip**, so "frame selected"
  puts the sensor inside the near plane. Set the clipping range to
  `(0.0001, 50)` first.
* **Add a dynamic body.** Newton returns *no model at all* from a stage whose
  `body_count` is zero, and this sensor is deliberately static — so a
  fixture-only stage looks exactly like a broken sensor.

## Diagnostics

The preflight answers "why is there no signal?" — Isaac Sim version, Newton
backend state, solver, whether `contacts.force` is allocated, OpenUSD version,
and whether the core package and its assets resolve.

Run it from the panel, or headless:

```bash
ISAAC=<isaac-sim-root>
REPO=<this-repo>
$ISAAC/kit/kit $ISAAC/apps/isaacsim.exp.base.kit \
    --no-window \
    --enable isaacsim.physics.newton \
    --enable isaacsim.physics.newton.tensors \
    --ext-folder $REPO/isaacsim_ext \
    --enable synaptics.sensors.tactile \
    --exec $REPO/isaacsim_ext/synaptics.sensors.tactile/scripts/kit_diagnostics.py
```

That boots in under a minute and exits non-zero on failure, so it doubles as a
CI gate — but check the printed verdict too: Kit occasionally crashes on its own
way out, *after* the report is complete, which shows up as a non-zero exit from
a run that actually passed. (The full `isaacsim.exp.full.newton.kit` experience works too, but
takes many minutes to start. `isaacsim.physics.newton.tensors` only silences an
unrelated `SimulationManager` traceback on Play.)

Swap the `--exec` script for any of the others; they all run the same way:

| Harness | Checks | Mutates the stage |
|---|---|---|
| `kit_diagnostics.py` | versions, Newton backend, solver, `contacts.force`, assets | no |
| `kit_smoke.py` | spawn a sensor, compose 52 force areas, build the panel | yes |
| `kit_scenario_test.py` | the demo scene reaches a state the sensor can bind to | yes |
| `kit_dead_weight.py` | drop a known mass, Σ taxel force ≈ *m·g* | yes |
| `kit_robustness.py` | full-array press; two sensors on one stage | yes |
| `kit_demo_scenes.py` | the presentation scenes land where intended, and play at the rate they ask for | yes |

The ones that mutate the stage replace it outright, so run them in a throwaway
session.

`./python.sh -m synaptics.sensors.tactile.diagnostics` also works, but a bare
Isaac python cannot import `pxr`, `warp`, `newton` or the Newton backend —
Kit adds those paths as it enables extensions — so it can only report on the
core package and the model profiles.

## Sensor models

A model is a JSON profile in `data/models/`; adding a sensor variant is a data
change, not a code change. `CTS0.0` is the 52-taxel generic module.

The assets a profile names (`cts0.0.usd`, `cts0.0_taxel_map.json`) are resolved
from the installed `synaptics_tactile_newton` package, not from this extension,
so Isaac Sim, Isaac Lab and the standalone examples all read the same bytes.

## Known limitations

**Statically mounted sensors only.** Under OpenUSD < 26.5 — which Isaac Sim
6.0.1 bundles (25.11) — the parallel physics parse
(`UsdPhysics.LoadUsdPhysicsFromRange`) corrupts the heap when one rigid body
owns many colliders. The CTS module has 52. Static colliders are unaffected at
any count, and the shipped asset carries `PhysicsCollisionAPI` without
`RigidBodyAPI`, so fixture-style scenes work today. Mounting the sensor on a
robot link needs OpenUSD ≥ 26.5, which arrives with Newton 1.5.0.

**MuJoCo GPU solver only.** See the support matrix. XPBD implements
`update_contacts()` but Isaac Sim never calls it.

**Broad presses under-read.** In the shipped asset the module's base
(`force_base`) is **coplanar** with the force areas — both top out at the same
height — and extends further out: the taxel array spans x ±14.75 mm, y ±6.45 mm
once mounted, while the base spans ±15.5 × ±8.0 mm. A flat indenter wider than
the array therefore rests on the base, which carries the load, and the taxels
register almost nothing: a 30 × 15 mm press reads 3/52 taxels and ~3 % of the
weight. Sized to the array (26 × 11 mm) it reads 47/52 and ~83 %.

That missing ~17 % is the base, not a measurement fault: summing the vertical
reaction on every sensor shape accounts for the weight exactly — 0.4103 N on the
taxels plus 0.0802 N on `force_base` is 0.4905 N, precisely *m·g*, with nothing
on the holder or connectors. The split is unchanged whether or not the contact
buffer overflows.

Keep calibration presses and whole-pad demos **inside the taxel array**. On real
hardware a rubber pad sits proud of the base, so this is an asset-fidelity gap
rather than a physical one.

**Do not raise `nconmax` past ~768.** Widening MuJoCo's contact buffer is the
right fix for `Number of Newton contacts (N) exceeded MJWarp limit`, and the
demo scenes ask for 512. But at **1024** every contact force reads zero: the
physics stays correct — objects rest exactly where they should — while the
entire force readout dies silently, with the runtime still bound and reporting
no error. Raising `njmax` alongside does not help. Measured working: 200, 512,
640, 768.

## Layout

```
config/extension.toml           Kit metadata + dependencies
data/models/<model>.json        sensor profiles
docs/                           this file, CHANGELOG
packaging/isaac-<version>/      per-Isaac-version dependency manifests
synaptics/sensors/tactile/
  extension.py                  Kit entry point: menus, panel, spawn action
  spawn.py                      references the asset, writes synaptics:* metadata
  config.py                     model profiles and asset resolution
  diagnostics.py                environment preflight (also a CLI)
  ui_builder.py                 the omni.ui panel
  adapters/                     everything version-specific about the Newton backend
scripts/kit_diagnostics.py      headless preflight, run inside Kit via --exec
scripts/kit_smoke.py            headless spawn + panel smoke test (mutates the stage)
scripts/kit_demo_scenes.py      headless check on the presentation scenes (mutates the stage)
```
