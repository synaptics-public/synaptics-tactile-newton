# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Console display helpers for Synaptics CTS sensor output.

These read the GPU tensors back to the CPU (a device sync) and format them for
humans, so they are for debugging/inspection only — never a hot RL loop.
"""

from .output import CTSOutput

# Active-taxel threshold [N] and taxels-per-row for the rank-1 grid.
_ACTIVE_EPS = 1e-4
_COLS_PER_ROW = 8


def _format_single(forces, step) -> str:
    """rank-1 (single sensor): summary header + numbered grid, 8 per row."""
    prefix = "" if step is None else f"step {step:5d} | "
    active = int((forces > _ACTIVE_EPS).sum())
    header = f"{prefix}\u03a3 {forces.sum():6.2f} N, {active}/{len(forces)} active"

    lines = [header]
    for start in range(0, len(forces), _COLS_PER_ROW):
        chunk = forces[start:start + _COLS_PER_ROW]
        cells = "  ".join(
            f"{start + i:02d}:{f:5.2f}" for i, f in enumerate(chunk)
        )
        lines.append("  " + cells)
    return "\n".join(lines)


def _format_batched(forces, step) -> str:
    """rank-2 (per-env): one sum + active-taxel count per env."""
    prefix = "" if step is None else f"step {step:5d} | "
    per_env = "  ".join(
        f"env{e}: {forces[e].sum():6.2f} N ({int((forces[e] > _ACTIVE_EPS).sum()):2d})"
        for e in range(forces.shape[0])
    )
    return f"{prefix}{per_env}"


def format_forces(output: CTSOutput, step=None) -> str:
    """Build a human-readable string for the sensor output.

    Rank 1 (single sensor) renders a per-taxel grid, 8 per row, under a
    ``Σ total, k/N active`` header; rank 2 ``(num_envs, num_taxels)`` renders one
    ``sum (active)`` summary per env. ``step`` optionally prefixes each line.
    """
    forces = output.force.numpy()
    if forces.ndim == 1:
        return _format_single(forces, step)
    if forces.ndim == 2:
        return _format_batched(forces, step)
    raise ValueError(
        f"output.force must be rank 1 or 2, got shape {forces.shape}"
    )


def print_forces(output: CTSOutput, step=None) -> None:
    """Format the sensor output and print it as one atomic block."""
    print(format_forces(output, step))
