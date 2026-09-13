# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Isaac Sim 6.x Newton backend adapter (validated on 6.0.1 and 6.1.0)."""

from .base import NewtonBackendAdapter


class IsaacSim60Adapter(NewtonBackendAdapter):
    """Binds to ``isaacsim.physics.newton`` as shipped in Isaac Sim 6.0.1
    (0.8.x) and 6.1.0 (1.0.x).

    The whole 6.x surface we use is implemented in
    :class:`~.base.NewtonBackendAdapter`; this subclass only names the module
    and the version it was validated against. When a future Isaac Sim moves the
    Newton backend's Python surface, add a sibling adapter and override the few
    methods that moved — do not branch inside the base class.
    """

    isaac_version = "6"
    extension_dependency = "isaacsim.physics.newton"
    backend_module_candidates = ("isaacsim.physics.newton",)

    #: Isaac Sim builds this adapter has actually been exercised against.
    validated_isaac_builds = ("6.0.1", "6.1.0")
