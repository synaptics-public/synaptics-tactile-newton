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

"""Warp kernels for Synaptics tactile signal generation."""

import warp as wp


@wp.kernel
def _extract_normal_forces(
    contact_forces: wp.array(dtype=wp.vec3),
    taxel_indices: wp.array(dtype=wp.int32),
    taxel_bodies: wp.array(dtype=wp.int32),
    sensing_axis_local: wp.vec3,
    body_q: wp.array(dtype=wp.transform),
    force_max: wp.float32,
    out_force: wp.array(dtype=wp.float32),
):
    """Project each taxel's contact force onto the shared press axis and clamp.

    The press axis (the sensor normal, shared by every force area) is stored in
    the sensor body's local frame, so it is rotated into world frame by the
    taxel's body pose (static shapes, body < 0, skip this) before the world-frame
    contact force is projected onto it. The result is clamped to ``[0, force_max]``
    — tension becomes 0, saturation is capped — and shear is discarded.

    Args:
        contact_forces: Per-shape world-frame force vectors, shape (N, 3).
        taxel_indices: Maps taxel index -> contact_forces row.
        taxel_bodies: Body index per taxel (-1 = static, no rotation).
        sensing_axis_local: Shared unit press axis in body-local frame (sign
            chosen so compression projects positive).
        body_q: Per-body world transforms.
        force_max: Saturation force [N].
        out_force: Output per-taxel axial force [N], shape (num_taxels,).
    """
    i = wp.tid()
    idx = taxel_indices[i]
    axis = sensing_axis_local
    b = taxel_bodies[i]
    if b >= 0:
        axis = wp.quat_rotate(wp.transform_get_rotation(body_q[b]), axis)
    f_axial = wp.dot(contact_forces[idx], axis)
    out_force[i] = wp.clamp(f_axial, 0.0, force_max)


@wp.kernel
def _taxel_world_positions(
    local_pos: wp.array(dtype=wp.vec3),
    taxel_bodies: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    out_positions: wp.array(dtype=wp.vec3),
):
    """Transform each taxel's body-local position into world frame.

    Applies the taxel body's current world transform so the output tracks the
    sensor as it moves; world-static shapes (body < 0) pass through unchanged.

    Args:
        local_pos: Per-taxel body-local positions, shape (num_taxels,).
        taxel_bodies: Body index per taxel (-1 = static / world-attached).
        body_q: Per-body world transforms.
        out_positions: Output per-taxel world position [m], shape (num_taxels,).
    """
    i = wp.tid()
    c = local_pos[i]
    b = taxel_bodies[i]
    if b >= 0:
        out_positions[i] = wp.transform_point(body_q[b], c)
    else:
        out_positions[i] = c


@wp.kernel
def _reduce_total_force(
    per_taxel_force: wp.array(dtype=wp.float32),
    taxel_bodies: wp.array(dtype=wp.int32),
    sensing_axis_local: wp.vec3,
    body_q: wp.array(dtype=wp.transform),
    taxels_per_group: wp.int32,
    out_total_flat: wp.array(dtype=wp.float32),
):
    """Reduce per-taxel normal force into per-group net force vectors [N].

    Each taxel's clamped scalar force is put back onto the world-frame press axis
    and accumulated into its group, giving the normal net force vector. Taxels
    are grouped into contiguous blocks of ``taxels_per_group``: one group for a
    standalone sensor, one per environment for the batched wrapper (env-major).

    Args:
        per_taxel_force: Per-taxel clamped normal force [N], shape (num_taxels,).
        taxel_bodies: Body index per taxel (-1 = static; axis used as-is).
        sensing_axis_local: Shared unit press axis in body-local frame.
        body_q: Per-body world transforms.
        taxels_per_group: Contiguous taxels per group (env).
        out_total_flat: Flat output totals, shape (num_groups * 3,); atomically
            accumulated per group. Must be zeroed by the caller.
    """
    i = wp.tid()
    axis = sensing_axis_local
    b = taxel_bodies[i]
    if b >= 0:
        axis = wp.quat_rotate(wp.transform_get_rotation(body_q[b]), axis)
    f = per_taxel_force[i]
    base = (i / taxels_per_group) * 3
    wp.atomic_add(out_total_flat, base + 0, f * axis[0])
    wp.atomic_add(out_total_flat, base + 1, f * axis[1])
    wp.atomic_add(out_total_flat, base + 2, f * axis[2])


@wp.kernel
def _scatter_env_force(
    src: wp.array(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    taxels_per_env: wp.int32,
    dst: wp.array2d(dtype=wp.float32),
):
    """Copy per-taxel force from the flat core array into each env's row.

    For env ``e``, the taxels live at ``src[e*taxels_per_env : (e+1)*taxels_per_env]``
    and are written to ``dst[e]``. Only envs whose ``env_mask[e]`` is True are
    touched; the rest keep their previous values. Runs on-device so the batching
    step needs no host round-trip.

    Args:
        src: Flat per-taxel force [N], env-major, shape (num_envs * taxels_per_env,).
        env_mask: Per-env update flags, shape (num_envs,); False leaves the row.
        taxels_per_env: Contiguous taxels per env.
        dst: Batched output, shape (num_envs, taxels_per_env).
    """
    env, t = wp.tid()
    if env_mask[env]:
        dst[env, t] = src[env * taxels_per_env + t]


@wp.kernel
def _scatter_env_positions(
    src: wp.array(dtype=wp.vec3),
    env_mask: wp.array(dtype=wp.bool),
    taxels_per_env: wp.int32,
    dst: wp.array3d(dtype=wp.float32),
):
    """Same copy as :func:`_scatter_env_force`, but for world-frame taxel positions.

    ``src`` holds one ``vec3`` per taxel; each component is written into the
    trailing axis of ``dst[e]`` (shape ``(taxels_per_env, 3)``). Masking and
    env-major layout are identical to :func:`_scatter_env_force` — see there for
    the layout sketch.

    Args:
        src: Flat per-taxel world positions [m], env-major, shape
            (num_envs * taxels_per_env,).
        env_mask: Per-env update flags, shape (num_envs,); False leaves the row.
        taxels_per_env: Contiguous taxels per env.
        dst: Batched output, shape (num_envs, taxels_per_env, 3).
    """
    env, t = wp.tid()
    if env_mask[env]:
        p = src[env * taxels_per_env + t]
        dst[env, t, 0] = p[0]
        dst[env, t, 1] = p[1]
        dst[env, t, 2] = p[2]
