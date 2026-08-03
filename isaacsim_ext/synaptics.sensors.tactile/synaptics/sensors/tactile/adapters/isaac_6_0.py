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

"""Isaac Sim 6.0 Newton backend adapter (reference target: 6.0.1)."""

from .base import NewtonBackendAdapter


class IsaacSim60Adapter(NewtonBackendAdapter):
    """Binds to ``isaacsim.physics.newton`` 0.8.x as shipped in Isaac Sim 6.0.1.

    The whole 6.0.x surface we use is implemented in
    :class:`~.base.NewtonBackendAdapter`; this subclass only names the module
    and the version it was validated against. When a future Isaac Sim moves the
    Newton backend's Python surface, add a sibling adapter and override the few
    methods that moved — do not branch inside the base class.
    """

    isaac_version = "6.0"
    extension_dependency = "isaacsim.physics.newton"
    backend_module_candidates = ("isaacsim.physics.newton",)

    #: Isaac Sim builds this adapter has actually been exercised against.
    validated_isaac_builds = ("6.0.1",)
