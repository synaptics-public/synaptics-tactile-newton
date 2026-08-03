# `isaacsim_ext/` — Isaac Sim Kit extensions

Kit extension search path. Point Isaac Sim at **this** folder; it contains one
extension per subdirectory, named after its extension id.

| Extension | What it is |
|---|---|
| [`synaptics.sensors.tactile`](synaptics.sensors.tactile/docs/README.md) | The CTS tactile sensor for Isaac Sim, on the Newton backend |

```bash
ISAAC=../IsaacSim_6.0.1
$ISAAC/isaac-sim.newton.sh --ext-folder $PWD/isaacsim_ext --enable synaptics.sensors.tactile
```

Use the **Newton** experience (`isaac-sim.newton.sh`): Isaac Sim's default app
runs PhysX and disables the Newton backend these extensions bind to.

Kit treats every subdirectory of a search path as an extension, so this folder
holds **only** extension directories — helper scripts live inside the
extension they belong to.

Design rules for anything added here:

- **No sensor math.** It belongs in `synaptics_tactile_newton`, where
  `pytest tests/` can exercise it. An extension is a shell over that package.
- **Everything version-specific goes behind `adapters/`.**
  `isaacsim.physics.newton` is pre-1.0 and its contact handling already changed
  between Isaac Sim 6.0.0-rc.22 and 6.0.1.
- **No repo-local helper files.** Everything here must work from a plain
  checkout — no paths outside `isaacsim_ext/` and `synaptics_tactile_newton/`.
