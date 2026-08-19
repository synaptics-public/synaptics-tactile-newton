# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Arrange taxels into a 2-D grid for display.

Pure geometry on the taxel centroids — no physics, and nothing Kit-specific, so
it is testable without booting Isaac Sim.

The sensing plane is spanned by the two axes that are *not* the press axis. For
the CTS the press axis is local -Y, so surface *u* is local **+X** and surface
*v* is local **-Z**. The calibration rig uses the same mapping, so a panel cell
and a calibration sweep coordinate name the same physical spot.
"""

import numpy as np


def surface_coords(centroids) -> np.ndarray:
    """Project body-local centroids onto the sensing surface.

    Args:
        centroids: ``(n, 3)`` body-local taxel centroids, in metres.

    Returns:
        ``(n, 2)`` array of ``(u, v)`` surface coordinates, same units as the
        input. Only relative positions matter for layout, so the units carry
        through untouched.
    """
    centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
    return np.column_stack((centroids[:, 0], -centroids[:, 2]))


#: Gaps below this fraction of the coordinate's full range are treated as
#: within-line jitter rather than a step to the next lattice line.
_JITTER_FRACTION = 0.01


def _cluster(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Group near-equal coordinates into lattice lines.

    The force-area grid is **slightly rotated** in the sensing plane (~0.065 deg
    for the CTS: consecutive cells along one row drift ~0.003 mm sideways while
    stepping 2.5 mm along the row), so taxels on one line do not share an exact
    coordinate.

    Splitting at half the *median* gap does not work: most gaps are the tiny
    within-line ones, so the median collapses towards zero and every taxel
    becomes its own line. Instead the pitch is the **smallest gap that is large
    relative to the coordinate's range**, which ignores jitter without needing
    the pitch supplied.
    """
    order = np.argsort(values)
    gaps = np.diff(values[order])
    span = float(values.max() - values.min())

    significant = gaps[gaps > _JITTER_FRACTION * span] if span > 0 else np.empty(0)
    threshold = float(significant.min()) * 0.5 if significant.size else np.inf

    labels = np.empty(values.size, dtype=int)
    line = 0
    labels[order[0]] = 0
    for previous, index in zip(order[:-1], order[1:]):
        if values[index] - values[previous] > threshold:
            line += 1
        labels[index] = line
    return labels, line + 1


def grid_layout(centroids) -> dict:
    """Assign each taxel a ``(row, col)`` cell.

    Rows run along the surface's ``v`` axis, top row first, so the panel reads
    the same way round as looking down at the sensor.

    Returns:
        dict with ``rows``, ``cols``, ``cells`` (``rows x cols`` of taxel index
        or ``None``), and ``row_of`` / ``col_of`` per taxel.
    """
    coords = surface_coords(centroids)
    col_of, cols = _cluster(coords[:, 0])
    row_of, rows = _cluster(coords[:, 1])
    row_of = (rows - 1) - row_of  # highest v first, so +v renders upward

    cells = [[None] * cols for _ in range(rows)]
    collisions = 0
    for index, (row, col) in enumerate(zip(row_of, col_of)):
        if cells[row][col] is None:
            cells[row][col] = index
        else:
            collisions += 1

    return {
        "rows": int(rows),
        "cols": int(cols),
        "cells": cells,
        "row_of": row_of.tolist(),
        "col_of": col_of.tolist(),
        "collisions": collisions,
    }
