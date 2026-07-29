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

"""Newton-native Synaptics Capacitive Tactile Sensor (CTS)."""

import json
from pathlib import Path

import warp as wp
import numpy as np
from newton.sensors import SensorContact

from .output import CTSOutput
from .kernels import _extract_normal_forces, _taxel_world_positions, _reduce_total_force


#: Length units a taxel map may declare in its ``"units"`` field, and their
#: factor to metres.
_CENTROID_UNITS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "mm": 1.0e-3, "millimeter": 1.0e-3, "millimeters": 1.0e-3,
    "millimetre": 1.0e-3, "millimetres": 1.0e-3,
    "cm": 1.0e-2, "centimeter": 1.0e-2, "centimeters": 1.0e-2,
    "centimetre": 1.0e-2, "centimetres": 1.0e-2,
}


def centroid_scale_to_meters(taxel_map: dict) -> float:
    """Factor converting a taxel map's centroids to metres.

    The map is a CAD export, so it is authored in the CAD's units (FreeCAD:
    millimetres) and names them in its ``"units"`` field. Every consumer should
    resolve the unit through this function rather than assuming one — that is
    what keeps ``CTSSensor.taxel_centroids`` SI alongside the sensor's other
    output, and what lets a map authored in different units drop straight in.

    A map with no ``"units"`` predates the field and is millimetres.

    Raises:
        ValueError: the map declares a unit we do not know.
    """
    units = taxel_map.get("units")
    if units is None:
        return 1.0e-3

    scale = _CENTROID_UNITS.get(str(units).strip().lower())
    if scale is None:
        raise ValueError(
            f"taxel map declares unknown units {units!r}; expected one of "
            f"{', '.join(sorted(set(_CENTROID_UNITS)))}"
        )
    return scale


class CTSSensor:
    """Synaptics Capacitive Tactile Sensor (CTS) for Newton.

    Consumes Newton contact data via ``SensorContact`` and launches Warp kernels
    to produce the Synaptics per-taxel output.

    Args:
        model: Finalized Newton Model (after ``builder.finalize()``).
        sensing_shape_pattern: Glob matching the sensing pad shapes in the USD
            hierarchy (e.g. ``"*/forceArea_*"``).
        sensing_axis: Body-local (x, y, z) press axis shared by every force area
            (the sensor normal). The contact force is projected onto it (shear
            discarded) and clamped to ``[0, force_max]``. Normalized internally;
            the kernel rotates it into world frame each step by the body pose.
            Default +Z. The sign must be baked into the taxel-map axis so
            compression projects positive; there is no runtime inversion flag.
        force_max: Per-taxel saturation force [N]. Default 100.0.
        taxel_map: Optional taxel map (``<out>_taxel_map.json`` path or parsed
            dict). When given it OVERRIDES ``sensing_axis`` with the map's shared
            press axis and exposes per-taxel names/centroids (matched to shapes
            BY NAME, so robust to shape ordering). Centroids are converted from
            the map's declared ``"units"`` (default millimetres) to metres.
        mount_rotation: Optional (x, y, z, w) quaternion baked once into the
            press axis at construction. Use only for a sensor FIXED to the world
            (all taxel shapes static, body == -1) whose USD was loaded with a
            reorienting rotation — pass that same rotation. Leave None for a
            sensor on a moving body (the kernel rotates the axis by the live
            pose instead).

    Data model:
        ``data`` is the only device-resident (GPU ``wp.array``) surface. Every
        ``taxel_*`` property is static host metadata: resolved once at
        construction and free to re-read (never re-copied from the device).
    """

    def __init__(
        self,
        model,
        sensing_shape_pattern: str,
        sensing_axis=(0.0, 0.0, 1.0),
        force_max: float = 100.0,
        taxel_map=None,
        mount_rotation=None,
    ):
        self._sensor_contact = SensorContact(
            model, sensing_obj_shapes=sensing_shape_pattern
        )
        self._num_taxels = len(self._sensor_contact.sensing_obj_idx)
        self._force_max = force_max
        device = model.device

        # Optional taxel map: OVERRIDES sensing_axis with its shared press axis
        # and provides per-taxel names/centroids (resolved by the helper).
        self._taxel_names = None
        self._taxel_centroids = None
        if taxel_map is not None:
            sensing_axis = self._resolve_taxel_map(model, taxel_map)

        # Normalize the shared press axis.
        axis = np.asarray(sensing_axis, dtype=np.float32).reshape(-1)
        if axis.shape != (3,):
            raise ValueError("sensing_axis must be a 3-vector (x, y, z)")
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            raise ValueError("sensing_axis must be non-zero")
        axis = axis / norm

        # For a world-static sensor the kernel cannot rotate the axis (no body
        # pose to read), so bake the USD's reorienting rotation into it here.
        if mount_rotation is not None:
            q = np.asarray(mount_rotation, np.float32)
            rotated = wp.quat_rotate(
                wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
                wp.vec3(float(axis[0]), float(axis[1]), float(axis[2])),
            )
            axis = np.array(rotated, dtype=np.float32)
        self._sensing_axis = wp.vec3(float(axis[0]), float(axis[1]), float(axis[2]))

        # Per-taxel body index (so the kernel can rotate the axis by the live
        # pose). Static/world-attached shapes have body -1.
        sensing_idx = np.asarray(self._sensor_contact.sensing_obj_idx, dtype=np.int64)
        if self._sensor_contact.sensing_obj_type == "body":
            taxel_bodies = sensing_idx.astype(np.int32)
        else:
            shape_body = model.shape_body.numpy()
            taxel_bodies = shape_body[sensing_idx].astype(np.int32)
        # Same per-taxel body index on both sides of the device boundary:
        # ``_wp`` feeds the kernels (GPU), ``_np`` backs the host property.
        self._taxel_bodies_wp = wp.array(taxel_bodies, dtype=wp.int32, device=device)
        self._taxel_bodies_np = taxel_bodies

        # Taxel-to-shape index mapping (identity for now — one taxel per shape)
        self._taxel_indices = wp.array(
            np.arange(self._num_taxels, dtype=np.int32), dtype=wp.int32, device=device
        )

        # Per-taxel body-local position (each sensing shape's own transform, in
        # meters) for the world-position kernel. Whole-body sensing has no
        # per-shape transform, so the taxel sits at the body origin.
        if self._sensor_contact.sensing_obj_type == "body":
            local_pos = np.zeros((self._num_taxels, 3), dtype=np.float32)
        else:
            shape_tf = model.shape_transform.numpy()
            local_pos = shape_tf[sensing_idx, :3].astype(np.float32)
        self._taxel_local_pos = wp.array(local_pos, dtype=wp.vec3, device=device)

        # Output tensors (GPU)
        self._data = CTSOutput(
            force=wp.zeros(self._num_taxels, dtype=wp.float32, device=device),
            total_force=wp.zeros(3, dtype=wp.float32, device=device),
            positions_w=wp.zeros(self._num_taxels, dtype=wp.vec3, device=device),
        )

        print(
            f"CTSSensor: {self._num_taxels} taxels, "
            f"axis={tuple(float(v) for v in axis)}, "
            f"force_max={force_max} N"
        )

    @staticmethod
    def _load_taxel_map(taxel_map):
        """Load and validate the taxel map (path or parsed dict).

        Requires the two keys this sensor uses: ``"axis"`` (the shared press
        axis 3-vector) and ``"taxels"`` (per-force-area entries keyed by name,
        each with a ``"centroid"``). The optional ``"units"`` key names the
        centroids' length unit (default ``"mm"``, the CAD export frame).
        """
        if isinstance(taxel_map, (str, Path)):
            with open(taxel_map, "r") as fh:
                taxel_map = json.load(fh)
        if not isinstance(taxel_map, dict):
            raise TypeError(
                "taxel_map must be a path to a taxel-map JSON file or a dict, "
                f"got {type(taxel_map).__name__}"
            )
        if taxel_map.get("axis") is None:
            raise KeyError(
                "taxel map has no top-level 'axis' (the single shared press axis)"
            )
        if not isinstance(taxel_map.get("taxels"), dict):
            raise KeyError(
                "taxel map has no top-level 'taxels' dict (per-force-area "
                "centroids keyed by name)"
            )
        return taxel_map

    def _resolve_taxel_map(self, model, taxel_map):
        """Read the shared press axis and cache per-taxel names/centroids.

        Names and centroids are matched to the sensing shapes BY NAME so they
        line up with the output rows regardless of shape ordering. Returns the
        shared axis as a (3,) array.
        """
        taxel_map = self._load_taxel_map(taxel_map)
        axis = taxel_map["axis"]
        taxels = taxel_map["taxels"]

        # Resolve each shape's taxel name (its USD label basename) and look the
        # centroid up BY NAME so centroids[row] lines up with forces()[row].
        labels = list(model.shape_label)
        names = [
            str(labels[i]).rsplit("/", 1)[-1]
            for i in self._sensor_contact.sensing_obj_idx
        ]
        centroid_scale = centroid_scale_to_meters(taxel_map)
        centroids = np.empty((self._num_taxels, 3), dtype=np.float32)
        for row, name in enumerate(names):
            entry = taxels.get(name)
            if entry is None:
                raise KeyError(
                    f"taxel map has no entry for sensing shape '{name}' "
                    f"(map has {len(taxels)} entries: "
                    f"{', '.join(sorted(taxels)[:5])}...)"
                )
            if "centroid" not in entry:
                raise KeyError(
                    f"taxel map entry '{name}' has no 'centroid'"
                )
            centroids[row] = np.asarray(entry["centroid"], dtype=np.float64) * centroid_scale
        self._taxel_names = names
        self._taxel_centroids = centroids
        return np.asarray(axis, dtype=np.float32)

    @property
    def num_taxels(self) -> int:
        return self._num_taxels

    @property
    def taxel_bodies(self):
        """Per-taxel body index (host ``int32``); ``-1`` for static/world shapes."""
        return self._taxel_bodies_np

    @property
    def taxel_names(self):
        """Per-taxel force-area names in sensing/output row order (or None)."""
        return self._taxel_names

    @property
    def taxel_centroids(self):
        """Per-taxel (num_taxels, 3) body-local centroids **in metres**, or None.

        The taxel map authors these in its own ``"units"`` (millimetres, from
        the CAD export); they are converted on load so this is SI like the rest
        of the sensor's output.
        """
        return self._taxel_centroids

    @property
    def data(self) -> CTSOutput:
        """Current sensor output (GPU tensors)."""
        return self._data

    def update(self, state, contacts):
        """Read contact forces and produce sensor output.

        Call this once per simulation step, after the solver step.

        Args:
            state: Current Newton simulation state.
            contacts: Current Newton contacts.
        """
        self._sensor_contact.update(state, contacts)

        wp.launch(
            _extract_normal_forces,
            dim=self._num_taxels,
            inputs=[
                self._sensor_contact.total_force,
                self._taxel_indices,
                self._taxel_bodies_wp,
                self._sensing_axis,
                state.body_q,
                self._force_max,
                self._data.force,
            ],
        )

        # World-frame per-taxel positions.
        wp.launch(
            _taxel_world_positions,
            dim=self._num_taxels,
            inputs=[
                self._taxel_local_pos,
                self._taxel_bodies_wp,
                state.body_q,
                self._data.positions_w,
            ],
        )

        # Net force vector on the whole sensing surface (all taxels -> one group).
        self.reduce_total_force(state, self._num_taxels, self._data.total_force)

    def reduce_total_force(self, state, taxels_per_group, out_total):
        """Reduce the current per-taxel normal force into net force vectors [N].

        Sums each taxel's clamped force along the world press axis in contiguous
        groups of ``taxels_per_group``: pass ``num_taxels`` for the single (3,)
        surface total, or the per-env count for the batched wrapper. ``out_total``
        is written in place (flattened length ``num_groups * 3``). Call after
        ``update`` has filled ``data.force`` for the step.
        """
        out_flat = out_total.flatten()
        out_flat.zero_()
        wp.launch(
            _reduce_total_force,
            dim=self._num_taxels,
            inputs=[
                self._data.force,
                self._taxel_bodies_wp,
                self._sensing_axis,
                state.body_q,
                int(taxels_per_group),
                out_flat,
            ],
        )
