# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.
#
# This material contains information that is proprietary to Synaptics
# Incorporated. The holder of this document shall treat all information
# contained herein as confidential, shall use the information only for its
# intended purpose and in conjunction with Synaptics products, and shall protect
# the information in whole or part from duplication, disclosure to any other
# party, or dissemination in any media without the written permission of
# Synaptics Incorporated.
#
# Information contained herein is provided AS-IS, with no express or implied
# warranties. SYNAPTICS HEREBY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED,
# INCLUDING, WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE, AND NONINFRINGEMENT. SYNAPTICS ASSUMES NO LIABILITY
# WHATSOEVER, INCLUDING NO LIABILTY FOR INTELLECTUAL PROPERTY INFRINGEMENT, FOR
# ANY DAMAGES, INCLUDING ANY SPECIAL, PUNITIVE, INCIDENTAL, OR CONSEQUENTIAL
# DAMAGES RESULTING FROM THE USE OF THE INFORMATION CONTAINED HEREIN. This
# material conveys no express or implied licenses to any intellectual property
# rights belonging to Synaptics or any other party. Synaptics may, from time to
# time and at its sole option, update the information contained herein without
# notice.
#
# Synaptics Incorporated
# 1109 McKay Drive
# San Jose, CA 95131
# (408) 904-1100

"""Live per-taxel force heatmap.

Presentation only: the colour ramp and the cell layout live here, the forces
come from ``CTSSensor``.

The colour scale **auto-ranges to the current frame's peak**. A per-taxel
saturation limit of 100 N against a 50 g indenter's ~0.04 N peak would otherwise
render every cell black; auto-ranging shows the contact *shape*, and the scale
value is printed so the absolute magnitude is never implied.
"""

import carb
import numpy as np
import omni.ui as ui

from .taxel_layout import grid_layout

#: A taxel this close to its configured maximum is clamped by the sensor model,
#: so it is drawn distinctly — a saturated reading otherwise looks like a
#: correct one.
SATURATION_FRACTION = 0.99

#: Floor on the auto-range scale [N], so sensor noise does not fill the grid
#: with colour when nothing is touching it.
MIN_SCALE_N = 0.005

#: Colour ramp as (position, (r, g, b)), interpolated piecewise.
_RAMP = (
    (0.00, (34, 34, 42)),
    (0.25, (30, 80, 170)),
    (0.50, (30, 160, 160)),
    (0.75, (220, 190, 60)),
    (1.00, (220, 60, 50)),
)

_SATURATED_RGB = (255, 80, 235)
_EMPTY_RGB = (22, 22, 26)

#: Cell edge in pixels, clamped so the grid stays usable in a narrow panel and
#: does not swallow a wide one.
MIN_CELL = 8
MAX_CELL = 160
DEFAULT_CELL = 22

_CELL_GAP = 2


def _abgr(rgb) -> int:
    """omni.ui packs colours as 0xAABBGGRR."""
    red, green, blue = (max(0, min(255, int(round(c)))) for c in rgb)
    return 0xFF000000 | (blue << 16) | (green << 8) | red


def force_color(force: float, scale: float, force_max: float) -> int:
    """Colour for one taxel reading, as an ``omni.ui`` packed colour."""
    if force_max > 0.0 and force >= force_max * SATURATION_FRACTION:
        return _abgr(_SATURATED_RGB)

    if scale <= 0.0:
        position = 0.0
    else:
        position = min(max(force / scale, 0.0), 1.0)

    for (low, low_rgb), (high, high_rgb) in zip(_RAMP, _RAMP[1:]):
        if position <= high:
            span = high - low
            blend = 0.0 if span <= 0.0 else (position - low) / span
            return _abgr(
                tuple(a + (b - a) * blend for a, b in zip(low_rgb, high_rgb))
            )
    return _abgr(_RAMP[-1][1])


class TaxelGrid:
    """A grid of coloured cells, one per taxel, rebuilt when the layout changes."""

    def __init__(self):
        self._layout = None
        self._centroids_signature = None
        self._cells = {}      # taxel index -> ui.Rectangle
        self._rows = []       # ui.HStack per grid row, so heights track width
        self._spacers = []    # placeholders where the pad has no taxel
        self._cell_px = DEFAULT_CELL
        self._dropped = 0
        self._caption = None

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(self, centroids, names=None) -> None:
        """Lay out cells for ``centroids``. Call inside a ui container."""
        self._layout = grid_layout(centroids)
        self._centroids_signature = self._signature(centroids)
        self._cells = {}

        self._rows = []
        self._spacers = []
        rows, cols = self._layout["rows"], self._layout["cols"]
        cell = self._cell_px
        with ui.VStack(spacing=_CELL_GAP, height=0):
            for row in range(rows):
                stack = ui.HStack(spacing=_CELL_GAP, height=cell)
                self._rows.append(stack)
                with stack:
                    for col in range(cols):
                        index = self._layout["cells"][row][col]
                        if index is None:
                            # Keep the footprint's real shape: the CTS pad has
                            # no taxel at its rounded corners.
                            self._spacers.append(ui.Spacer())
                            continue
                        label = None if names is None else str(names[index])
                        # No explicit width: the HStack divides the row evenly,
                        # so cells follow the panel instead of forcing it wider.
                        rect = ui.Rectangle(
                            height=cell,
                            style={"background_color": _abgr(_EMPTY_RGB)},
                            tooltip=label or f"taxel {index}",
                        )
                        self._cells[index] = rect
            self._caption = ui.Label("", height=0)

        dropped = self._layout.get("collisions", 0)
        if dropped:
            # A cell that never got drawn is indistinguishable from one reading
            # zero, so say it out loud rather than quietly showing 51 of 52.
            carb.log_warn(
                f"[Synaptics Tactile] taxel layout dropped {dropped} taxel(s): "
                f"two centroids landed in the same grid cell."
            )
            self._dropped = dropped

    def matches(self, centroids) -> bool:
        """True when an existing layout already fits these centroids."""
        return (
            self._layout is not None
            and self._signature(centroids) == self._centroids_signature
        )

    @staticmethod
    def _signature(centroids):
        """Key the layout on the actual geometry.

        Keying on the taxel *count* alone would reuse one sensor's layout for a
        different sensor that happens to have as many taxels.
        """
        coords = np.asarray(centroids, dtype=np.float64)
        return (coords.shape, hash(np.round(coords, 9).tobytes()))

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, forces, force_max: float) -> dict:
        """Recolour every cell. Returns the stats the panel reports."""
        peak = float(forces.max()) if len(forces) else 0.0
        scale = max(peak, MIN_SCALE_N)
        saturated = (
            int((forces >= force_max * SATURATION_FRACTION).sum()) if force_max > 0.0 else 0
        )

        for index, rect in self._cells.items():
            if index >= len(forces):
                continue
            rect.set_style(
                {"background_color": force_color(float(forces[index]), scale, force_max)}
            )

        if self._caption is not None:
            self._caption.text = (
                f"colour scale 0 – {scale:.4f} N   |   peak {peak:.4f} N"
                + (f"   |   SATURATED: {saturated} taxel(s)" if saturated else "")
                + (f"   |   {self._dropped} taxel(s) not shown" if self._dropped else "")
            )
        return {"peak": peak, "scale": scale, "saturated": saturated}

    def sync_cell_height(self) -> bool:
        """Make cells square against their measured width. True if it changed.

        Cell *width* is elastic — the row divides itself evenly — so the panel
        drives the grid rather than the other way round. Only the height has to
        be set, and it is read back from a laid-out cell.

        Driving this from the frame's ``computed_width`` instead would feed back
        on itself: the frame measures its content, so wider cells report a wider
        frame, which widens the cells again until they hit the clamp.

        The pad's taxels are square, and a stretched grid would misrepresent
        where a contact is — hence square rather than "fill the panel".
        """
        if not self._cells:
            return False

        sample = next(iter(self._cells.values()))
        measured = float(getattr(sample, "computed_width", 0.0) or 0.0)
        if measured <= 0.0:
            return False

        cell = max(MIN_CELL, min(MAX_CELL, int(measured)))
        if cell == self._cell_px:
            return False

        self._cell_px = cell
        for rect in self._cells.values():
            rect.height = ui.Pixel(cell)
        for stack in self._rows:
            stack.height = ui.Pixel(cell)
        return True

    def clear(self) -> None:
        for rect in self._cells.values():
            rect.set_style({"background_color": _abgr(_EMPTY_RGB)})
        if self._caption is not None:
            self._caption.text = ""

    def destroy(self) -> None:
        self._cells = {}
        self._rows = []
        self._spacers = []
        self._caption = None
        self._layout = None
