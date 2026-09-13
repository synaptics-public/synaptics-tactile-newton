# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Selects the Newton backend adapter for the running Isaac Sim."""

import os

from .isaac_6_0 import IsaacSim60Adapter

#: Force one adapter (release packages, CI) instead of auto-detecting.
ISAAC_VERSION_ENV_VAR = "SYNAPTICS_TACTILE_ISAAC_VERSION"

#: Newest first. Isaac Sim < 6.0 has no Newton backend at all, so there is
#: deliberately no 5.x entry here — see docs/README.md, "Support matrix".
_ADAPTERS = (IsaacSim60Adapter,)


def get_newton_adapter(preferred_version: str | None = None):
    """Return the first available adapter for the active Isaac Sim runtime.

    Raises:
        ImportError: when no supported Newton backend is importable — which on a
            correctly installed Isaac Sim 6.x means the extension was loaded
            without its ``isaacsim.physics.newton`` dependency.
    """
    requested = (preferred_version or os.environ.get(ISAAC_VERSION_ENV_VAR, "")).strip()
    if requested:
        candidates = tuple(
            adapter for adapter in _ADAPTERS if requested.startswith(adapter.isaac_version)
        )
        if not candidates:
            supported = ", ".join(adapter.isaac_version for adapter in _ADAPTERS)
            raise ImportError(
                f"{ISAAC_VERSION_ENV_VAR}={requested} names an unsupported Isaac "
                f"Sim version. Supported: {supported}."
            )
    else:
        candidates = _ADAPTERS

    import_errors = []
    for adapter_cls in candidates:
        adapter = adapter_cls()
        try:
            adapter.backend()
            return adapter
        except ImportError as error:
            import_errors.append(str(error))

    raise ImportError(
        "No supported Isaac Sim Newton backend is available. The Newton backend "
        "(isaacsim.physics.newton) ships with Isaac Sim 6.0 and later; earlier "
        "versions are PhysX-only and cannot run this sensor. "
        f"Set {ISAAC_VERSION_ENV_VAR} to force one adapter if detection is "
        f"ambiguous. Details: {'; '.join(import_errors)}"
    )
