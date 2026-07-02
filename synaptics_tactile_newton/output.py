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
