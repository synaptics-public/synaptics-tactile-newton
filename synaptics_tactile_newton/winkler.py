# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Winkler-foundation readout for :class:`CTSSensor` (the "F1 readout split").

Rigid contact only guarantees the *net* wrench on a body — the split of that
wrench across a row of coplanar taxels is statically indeterminate, so the
engine's per-taxel forces under a resting object are an accident of where the
contact-manifold generator put its points (a flat cube reads ``2, 8, 7, 8, 9``
instead of flat). No amount of contact-parameter tuning fixes that.

This module leaves the engine's dynamics (normal response *and* friction)
completely untouched and fixes only the *readout*: after ``CTSSensor.update``
has produced the engine's per-taxel forces, :class:`WinklerReadout` recomputes
the spatial distribution as a Winkler foundation — a bed of independent
springs, the correct model for a thin pad on a rigid backing — and
renormalizes it so each pressing body's total matches what the engine already
delivered. Flat face -> flat profile; tilted face -> linear gradient; sphere ->
a small round patch peaking under the centre. Totals unchanged.

The Winkler model itself is object-agnostic: pressure depends only on the
local indentation, ``p = k * w``. The pressing object enters solely through
its surface — the indentation profile is the object's shape stamped into the
bed — so the readout derives each taxel's gap from the *collision shapes the
Newton model already carries* (boxes and spheres today) and needs no
per-object configuration. Per taxel and pressing body the virtual penetration
is

    delta_i = max(0, delta0 - (gap_i - min_j gap_j))

where ``gap_i`` is the signed distance from the body's surface to the taxel
reference point. Referencing gaps to the body's *deepest* taxel cancels both
the engine's (tiny, noisy) true penetration and the common offset between the
taxel reference point and the physical pad tip; ``delta0`` then dwarfs the
sub-micron geometric noise that makes the raw engine split meaningless (the
conditioning requirement delta* >> epsilon). For a flat face ``delta0`` is
pure conditioning — any value well above the noise gives the same profile.
For a curved surface it also sets the apparent contact-patch size
(``a ~ sqrt(2 R delta0)`` for a sphere of radius R), standing in for the
pad's physical compliance — calibrate it if patch width matters.

Limitations: pressing bodies are seen through their BOX, SPHERE, CAPSULE and
CYLINDER collision shapes only (a body with none of those is invisible to the
correction and its taxels keep the raw engine forces); overlapping contact
patches from two bodies are split by nearest-surface ownership; single-world
models only (the batched Isaac Lab wrapper is not handled).
"""

from __future__ import annotations

import numpy as np

from newton import GeoType

#: Shape types the gap computation understands. Scale layout per type:
#: BOX = half extents, the others = (radius, half_height, -); capsule and
#: cylinder axes run along the shape's local Z.
_SUPPORTED_GEO = (
    int(GeoType.BOX),
    int(GeoType.SPHERE),
    int(GeoType.CAPSULE),
    int(GeoType.CYLINDER),
)


class WinklerReadout:
    """Redistribute a :class:`CTSSensor`'s per-taxel forces via a Winkler model.

    Purely a readout post-processor: memoryless, host-side, and applied *after*
    ``sensor.update``. It rewrites ``sensor.data.force`` in place, preserving
    each pressing body's engine total (and therefore the overall total)
    exactly, so every downstream consumer (totals, display, panels) sees the
    corrected distribution.

    Configuration-free: candidate pressing bodies and their surface geometry
    are read off the finalized model at construction — every dynamic body with
    a box, sphere, capsule or cylinder collision shape, excluding the bodies
    that carry the sensor's own taxels.

    Args:
        sensor: The :class:`CTSSensor` whose readout to correct.
        model: The finalized Newton model the sensor was built against.
        delta0: Virtual engagement depth [m]. Taxels whose gap to a body
            exceeds that body's deepest taxel by more than this read zero for
            it; within it, weight falls off linearly (the Winkler spring law).
            Default 0.3 mm.
        include_margin: A body only participates when its closest approach is
            within this of the globally closest body [m] — what keeps an
            object *hovering* above the array from claiming taxels. Taxel
            reference points share one common tip offset, so this comparison
            between bodies is offset-free. Default: ``delta0``.
        aperture: Taxels are areas, not points: each taxel's weight is the
            Winkler weight averaged over a 3x3 sample grid spanning this width
            [m] in the sensor plane, so a thin or diagonal contact that misses
            a covered cell's *centroid* still registers on that cell (a 2 mm
            wire lying diagonally otherwise reads a row of holes). ``"auto"``
            (default) uses the median nearest-neighbour taxel spacing — the
            cell pitch; ``None``/``0`` restores point sampling at centroids.
            The sample offsets lie in the plane perpendicular to the sensor's
            ``sensing_axis`` (fallback: the taxel cloud's own plane), resolved
            once at the first update — a sensor that *rotates* mid-run should
            use point sampling.
    """

    def __init__(
        self,
        sensor,
        model,
        delta0: float = 3.0e-4,
        include_margin=None,
        aperture="auto",
    ):
        self._sensor = sensor
        self._delta0 = float(delta0)
        if self._delta0 <= 0.0:
            raise ValueError("delta0 must be positive")
        self._include_margin = (
            self._delta0 if include_margin is None else float(include_margin)
        )

        # Static host snapshot of every candidate pressing shape: dynamic
        # bodies only, sensor's own bodies excluded, supported types only.
        taxel_bodies = set(int(b) for b in sensor.taxel_bodies)
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_tf = model.shape_transform.numpy()

        shapes = []  # (body, geo_type, scale(3,), local_pos(3,), local_quat(4,))
        for s in range(len(shape_body)):
            body = int(shape_body[s])
            if body < 0 or body in taxel_bodies:
                continue
            geo = int(shape_type[s])
            if geo not in _SUPPORTED_GEO:
                continue
            shapes.append(
                (
                    body,
                    geo,
                    shape_scale[s].astype(np.float64),
                    shape_tf[s, 0:3].astype(np.float64),
                    shape_tf[s, 3:7].astype(np.float64),
                )
            )
        self._shapes = shapes
        self._bodies = sorted({s[0] for s in shapes})

        self._aperture = aperture
        self._offsets = None  # resolved sample offsets, (K, 3); lazy

        # Debug/introspection snapshots of the last update.
        self.gaps = None      # (num_bodies, num_taxels) signed distances [m]
        self.owners = None    # (num_taxels,) owning body index, -1 = none
        self.weights = None   # (num_taxels,) owner's Winkler weight [m]

    @property
    def active(self) -> bool:
        """False when the model offered no supported pressing shape at all."""
        return bool(self._shapes)

    @property
    def bodies(self) -> list:
        """Body indices this readout watches (host ints)."""
        return list(self._bodies)

    def _sample_offsets(self, taxels: np.ndarray) -> np.ndarray:
        """Resolve the aperture sample offsets, (K, 3); K=1 for point sampling.

        A 3x3 grid at the centres of the cell's thirds, spanning ``aperture``
        in the sensor plane. The in-plane basis comes from the sensor's press
        axis when it exposes one, else from the taxel cloud's own principal
        plane. Resolved once and cached (see the ``aperture`` doc for the
        static-orientation assumption).
        """
        if self._offsets is not None:
            return self._offsets

        width = self._aperture
        if width == "auto":
            if len(taxels) < 2:
                width = 0.0
            else:
                diff = taxels[:, np.newaxis, :] - taxels[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, np.inf)
                width = float(np.median(dist.min(axis=1)))
        width = 0.0 if width is None else float(width)
        if width <= 0.0:
            self._offsets = np.zeros((1, 3))
            return self._offsets

        axis = getattr(self._sensor, "sensing_axis", None)
        if axis is not None:
            normal = np.asarray(axis, dtype=np.float64).reshape(3)
        else:
            centred = taxels - taxels.mean(axis=0)
            _, sv, vt = np.linalg.svd(centred, full_matrices=False)
            # Smallest principal direction = the plane normal; degenerate
            # (collinear) clouds fall back to global z.
            normal = vt[2] if len(sv) > 2 and sv[1] > 1e-9 else np.array([0.0, 0.0, 1.0])
        normal = normal / np.linalg.norm(normal)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(normal[0]) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        u = seed - np.dot(seed, normal) * normal
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)

        step = width / 3.0
        grid = np.array([-step, 0.0, step])
        self._offsets = np.array(
            [a * u + b * v for a in grid for b in grid]
        )
        return self._offsets

    def update(self, state) -> np.ndarray:
        """Recompute the per-taxel split from the current pose and write it back.

        Call once per readout, after ``sensor.update(state, contacts)``. Reads
        body poses from ``state.body_q`` and the taxel positions from the
        sensor's own output. Returns the per-taxel forces (host array); when no
        supported pressing body exists the engine forces pass through
        untouched.
        """
        sensor = self._sensor
        engine = sensor.data.force.numpy()
        if not self._shapes:
            return engine

        taxels = sensor.data.positions_w.numpy().astype(np.float64)
        body_q = state.body_q.numpy().astype(np.float64)

        # Aperture sampling: evaluate every taxel at K in-plane sample points
        # so the weight integrates over the cell area rather than sampling its
        # centroid (K=1 for point sampling).
        offsets = self._sample_offsets(taxels)
        num_samples = len(offsets)
        points = (taxels[np.newaxis, :, :] + offsets[:, np.newaxis, :]).reshape(-1, 3)

        # Signed distance from each candidate body's surface to each sample
        # point (minimum over the body's shapes), (num_bodies, K, num_taxels).
        num_bodies = len(self._bodies)
        body_row = {b: i for i, b in enumerate(self._bodies)}
        gaps_s = np.full((num_bodies, num_samples, len(taxels)), np.inf)
        for body, geo, scale, local_pos, local_quat in self._shapes:
            q = body_q[body]
            rot = _quat_to_matrix(q[3:7]) @ _quat_to_matrix(local_quat)
            origin = q[0:3] + _quat_to_matrix(q[3:7]) @ local_pos
            local = (points - origin) @ rot  # rows are R^T (p - origin)
            if geo == int(GeoType.SPHERE):
                sd = np.linalg.norm(local, axis=1) - scale[0]
            elif geo == int(GeoType.CAPSULE):
                # Distance to the core segment along local Z, minus the radius.
                radius, half_height = scale[0], scale[1]
                core = local.copy()
                core[:, 2] -= np.clip(local[:, 2], -half_height, half_height)
                sd = np.linalg.norm(core, axis=1) - radius
            elif geo == int(GeoType.CYLINDER):
                radius, half_height = scale[0], scale[1]
                d_radial = np.linalg.norm(local[:, 0:2], axis=1) - radius
                d_axial = np.abs(local[:, 2]) - half_height
                outside = np.sqrt(
                    np.maximum(d_radial, 0.0) ** 2 + np.maximum(d_axial, 0.0) ** 2
                )
                inside = np.minimum(np.maximum(d_radial, d_axial), 0.0)
                sd = outside + inside
            else:  # BOX; scale = half extents
                excess = np.abs(local) - scale[np.newaxis, :]
                outside = np.linalg.norm(np.maximum(excess, 0.0), axis=1)
                inside = np.minimum(excess.max(axis=1), 0.0)
                sd = outside + inside
            row = body_row[body]
            gaps_s[row] = np.minimum(gaps_s[row], sd.reshape(num_samples, -1))

        # Bodies actually bearing on the array: within include_margin of the
        # globally closest approach. The taxel reference points share one tip
        # offset, so this between-body comparison is offset-free; a body
        # hovering above the surface is excluded here.
        gaps = gaps_s.min(axis=1)  # per-taxel closest sample, (bodies, taxels)
        min_gap = gaps.min(axis=1)
        active = min_gap <= min_gap.min() + self._include_margin

        # Winkler weight per (active body, taxel): the spring law evaluated at
        # every sample and averaged over the cell, then nearest-surface
        # ownership so overlapping patches are split rather than double-counted.
        weights = np.zeros_like(gaps)
        for row in np.flatnonzero(active):
            weights[row] = np.maximum(
                0.0, self._delta0 - (gaps_s[row] - min_gap[row])
            ).mean(axis=0)
        owners_row = weights.argmax(axis=0)
        owned = weights[owners_row, np.arange(len(taxels))] > 0.0

        # Redistribute each body's own engine total across its taxels. Taxels
        # no body claims keep the raw engine force, so the overall sum is
        # preserved even for contact the model cannot explain.
        out = engine.astype(np.float64).copy()
        for row in np.flatnonzero(active):
            mask = owned & (owners_row == row)
            if not mask.any():
                continue
            group_total = float(engine[mask].sum())
            w = weights[row, mask]
            out[mask] = group_total * w / w.sum() if group_total > 0.0 else 0.0

        np.clip(out, 0.0, sensor.force_max, out=out)
        self.gaps = gaps
        self.owners = np.where(owned, np.array(self._bodies)[owners_row], -1)
        self.weights = np.where(owned, weights[owners_row, np.arange(len(taxels))], 0.0)

        out32 = out.astype(np.float32)
        sensor.data.force.assign(out32)
        return out32


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
