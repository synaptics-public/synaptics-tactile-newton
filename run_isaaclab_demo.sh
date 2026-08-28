#!/usr/bin/env bash
# Run the Isaac Lab demo with the environment it needs.
# Usage: ./run_isaaclab_demo.sh /path/to/IsaacLab [demo flags...]
# No -u: Isaac Sim's setup_python_env.sh appends to possibly-unset variables.
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB="${1:?usage: $0 /path/to/IsaacLab [demo flags...]}"; shift
ISAAC_SIM="$(readlink -f "${ISAACLAB}/_isaac_sim")"
if [ ! -f "${ISAAC_SIM}/setup_python_env.sh" ]; then
    echo "ERROR: '${ISAACLAB}/_isaac_sim' does not point at an Isaac Sim install."
    exit 1
fi

source "${REPO_DIR}/.venv-newton/bin/activate"
export ISAAC_PATH="${ISAAC_SIM}" CARB_APP_PATH="${ISAAC_SIM}/kit" EXP_PATH="${ISAAC_SIM}/apps"
source "${ISAAC_SIM}/setup_python_env.sh"

DEMO="${REPO_DIR}/examples/isaaclab_task.py"
cd "${ISAACLAB}"
exec ./isaaclab.sh -p "${DEMO}" "$@"
