# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Synaptics tactile sensor output dataclass."""

from dataclasses import dataclass

import warp as wp


@dataclass
class CTSOutput:
    """Synaptics CTS sensor output.

    All arrays live on GPU (wp.array) so they can be used directly as RL
    observations without a CPU roundtrip.
    """

    force: wp.array
    """Per-taxel normal force [N], shape (num_taxels,)."""

    total_force: wp.array
    """Net force vector [N] on the sensing surface, shape (3,)."""

    positions_w: wp.array | None = None
    """Per-taxel position in world frame [m], shape (num_taxels, 3).

    Each taxel's sensing-shape position transformed by its body's live pose, so
    it tracks the sensor as it moves (world-static shapes are already in world
    frame). Derived from the Newton shape transforms, so it is always available
    and independent of the taxel map."""
