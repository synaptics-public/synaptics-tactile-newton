# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

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
