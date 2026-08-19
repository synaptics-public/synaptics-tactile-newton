# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Isaac Sim version-compatibility adapters."""

from .base import NewtonBackendAdapter
from .detect import ISAAC_VERSION_ENV_VAR, get_newton_adapter
from .isaac_6_0 import IsaacSim60Adapter

__all__ = [
    "NewtonBackendAdapter",
    "IsaacSim60Adapter",
    "get_newton_adapter",
    "ISAAC_VERSION_ENV_VAR",
]
