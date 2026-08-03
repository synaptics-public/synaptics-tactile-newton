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

"""Synaptics Capacitive Tactile Sensor (CTS) extension for Isaac Sim.

Kit discovers :class:`SynapticsTactileSensorExtension` here. The import is
guarded so that ``python -m synaptics.sensors.tactile.diagnostics`` still works
from a python that has never booted Kit — the diagnostics CLI is the first
thing anyone runs when the extension will not load.
"""

try:
    from .extension import SynapticsTactileSensorExtension
except ModuleNotFoundError as error:
    if not str(error.name or "").startswith(("omni", "carb", "pxr")):
        raise
    SynapticsTactileSensorExtension = None

__all__ = ["SynapticsTactileSensorExtension"]
