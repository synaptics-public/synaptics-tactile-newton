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

"""Optional Isaac Lab wrapper for the Synaptics Newton tactile sensor.

The only part of ``synaptics_tactile_newton`` that depends on Isaac Lab; the
top-level package never imports it, so the core Newton sensor stays free of the
heavy dependency. Requires **Isaac Lab 3.0+** (the first release with the Newton
backend). Install via the optional extra::

    pip install synaptics-tactile-newton[isaaclab]
"""

try:
    import isaaclab  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without Isaac Lab
    raise ImportError(
        "synaptics_tactile_newton.isaaclab requires Isaac Lab 3.0+ (the first "
        "release with the Newton physics backend). It is an optional dependency.\n"
        "Install it with:\n"
        "    pip install synaptics-tactile-newton[isaaclab]\n"
        "or install Isaac Lab directly. The core Newton sensor "
        "(synaptics_tactile_newton.CTSSensor) does NOT need Isaac Lab."
    ) from exc

from .sensor import CTSSensorIsaacLab
from .sensor_cfg import CTSSensorCfg

__all__ = ["CTSSensorIsaacLab", "CTSSensorCfg"]
