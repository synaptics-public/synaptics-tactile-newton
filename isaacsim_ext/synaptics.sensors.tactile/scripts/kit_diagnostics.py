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

"""Headless preflight: load the extension inside Kit and print the diagnostics.

A bare ``./python.sh`` cannot import ``pxr``, ``warp``, ``newton`` or
``isaacsim.physics.newton`` — those module paths are added by Kit as it enables
extensions. So the authoritative diagnostics run happens inside Kit, either
from the extension's panel or from here.

Run it on the **base** experience with the Newton backend enabled — that boots
in under a minute, where the full app takes many minutes and is not needed to
answer any of these questions::

    ISAAC=../IsaacSim_6.0.1
    EXT=isaacsim_ext/synaptics.sensors.tactile
    $ISAAC/kit/kit $ISAAC/apps/isaacsim.exp.base.kit \\
        --no-window \\
        --/app/python/extraPaths/0=$PWD \\
        --enable isaacsim.physics.newton \\
        --enable isaacsim.physics.newton.tensors \\
        --ext-folder isaacsim_ext \\
        --enable synaptics.sensors.tactile \\
        --exec $EXT/scripts/kit_diagnostics.py

Notes on the invocation:

* ``--ext-folder`` must point at ``isaacsim_ext`` itself — Kit treats every
  subdirectory of a search path as an extension.
* ``--enable isaacsim.physics.newton`` is what the GUI's
  ``isaac-sim.newton.sh`` does via its own experience file; the *default* full
  app disables the Newton backend and runs PhysX.
* ``--exec`` runs after Kit finishes starting extensions, so the extension's
  python module is importable here.

Exit status is non-zero when the preflight fails, so this doubles as a CI gate.
"""

import sys

import carb
import omni.kit.app

_BANNER = "=" * 72


def _main() -> int:
    app = omni.kit.app.get_app()
    manager = app.get_extension_manager()

    extension_id = None
    for extension in manager.get_extensions():
        if extension["name"] == "synaptics.sensors.tactile":
            extension_id = extension["id"]
            break

    print(_BANNER)
    if extension_id is None:
        print("FAIL: synaptics.sensors.tactile is not registered.")
        print("      Check --ext-folder points at the folder CONTAINING the")
        print("      synaptics.sensors.tactile directory.")
        return 2

    enabled = manager.is_extension_enabled(extension_id)
    print(f"Extension: {extension_id} (enabled={enabled})")
    if not enabled:
        print("FAIL: the extension is registered but did not enable.")
        return 2

    from synaptics.sensors.tactile.diagnostics import format_diagnostics, run_diagnostics

    report = run_diagnostics()
    print(_BANNER)
    print(format_diagnostics(report, verbose=True))
    print(_BANNER)
    return 0 if report["passed"] else 1


try:
    _status = _main()
except Exception as error:  # noqa: BLE001 - report, then still shut Kit down.
    carb.log_error(f"[Synaptics Tactile] kit_diagnostics failed: {error}")
    import traceback

    traceback.print_exc()
    _status = 3

sys.stdout.flush()
omni.kit.app.get_app().post_quit(_status)
