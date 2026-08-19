# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

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
