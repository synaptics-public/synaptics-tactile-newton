#!/usr/bin/env bash
# Setup Newton physics engine environment.
# Creates a Python 3.12 venv and installs newton with examples.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv-newton"

# --- Parse arguments ---
# Optional: --with-isaaclab <path> also installs Isaac Lab (editable, from a
# local source checkout) into the venv so the isaaclab_* examples can run. Isaac
# Lab 3.0+ (first release with the Newton backend) is not on PyPI, so it can
# only be installed from a local checkout.
ISAACLAB_PATH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --with-isaaclab)
            ISAACLAB_PATH="${2:-}"
            if [ -z "${ISAACLAB_PATH}" ]; then
                echo "ERROR: --with-isaaclab requires a path to your Isaac Lab checkout."
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--with-isaaclab /path/to/IsaacLab]"
            echo "  --with-isaaclab PATH  Also install Isaac Lab (editable) from PATH"
            echo "                        into the venv, for the isaaclab_* examples."
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1' (see --help)."
            exit 1
            ;;
    esac
done

echo "=== Newton Physics Engine Setup ==="

# --- Check Python 3.12 ---
PYTHON="python3.12"
if ! command -v "${PYTHON}" &>/dev/null; then
    echo "Python 3.12 not found, falling back to python3.11..."
    PYTHON="python3.11"
    if ! command -v "${PYTHON}" &>/dev/null; then
        echo "ERROR: Need Python 3.11+ (3.12 preferred). Install with:"
        echo "  sudo apt install python3.12 python3.12-venv"
        exit 1
    fi
fi

echo "  Python: $(${PYTHON} --version)"

# --- Create venv ---
if [ -d "${VENV_DIR}" ]; then
    echo "  Venv already exists at ${VENV_DIR}"
else
    echo "  Creating venv at ${VENV_DIR}..."
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

# --- Activate and install ---
source "${VENV_DIR}/bin/activate"

echo "  Upgrading pip..."
pip install --upgrade pip -q

# Pin Newton AND warp-lang to the validated "Tested against" stack. Newton
# 1.2.0 leaves warp-lang unbounded, and a floating resolve picks up warp 1.16+,
# whose codegen rejects mujoco-warp's kernels (WarpCodegenKeyError in
# _sensor_pos) — the standalone example then crashes on startup. 1.13.0 is also
# the version Isaac Lab pins, so the --with-isaaclab path installs without a
# resolver conflict.
NEWTON_VERSION="1.2.0"
WARP_VERSION="1.13.0"
echo "  Installing newton[examples]==${NEWTON_VERSION} (warp-lang==${WARP_VERSION})..."
pip install "newton[examples]==${NEWTON_VERSION}" "warp-lang==${WARP_VERSION}" -q

echo "  Installing this package (synaptics-tactile-newton) in editable mode, with the optional viewer (rerun-sdk) and test (pytest) extras..."
pip install -e "${SCRIPT_DIR}[viewer,test]" -q

# --- Optional: Isaac Lab (for the isaaclab_* examples) ---
if [ -n "${ISAACLAB_PATH}" ]; then
    ISAACLAB_PATH="$(cd "${ISAACLAB_PATH}" 2>/dev/null && pwd || true)"
    if [ -z "${ISAACLAB_PATH}" ] || [ ! -d "${ISAACLAB_PATH}/source/isaaclab" ]; then
        echo "ERROR: '${ISAACLAB_PATH:-<empty>}' is not an Isaac Lab checkout"
        echo "       (expected a 'source/isaaclab' directory inside it)."
        exit 1
    fi
    echo ""
    echo "  Installing Isaac Lab (editable) from ${ISAACLAB_PATH}..."
    echo "  NOTE: this pulls a large dependency tree (PyTorch + CUDA) and may"
    echo "        downgrade warp-lang / usd-core to the versions Isaac Lab pins."
    # isaaclab_ppisp is required by isaaclab_newton and is not on PyPI, so it
    # must be installed editable from the local checkout too. isaaclab_visualizers
    # provides the --viz backends; its base install needs only isaaclab + numpy
    # (the heavy newton[sim] git pin lives in its extras, which we don't request),
    # so it adds the viewers without a Newton rebuild.
    pip install -e "${ISAACLAB_PATH}/source/isaaclab" \
                -e "${ISAACLAB_PATH}/source/isaaclab_ppisp" \
                -e "${ISAACLAB_PATH}/source/isaaclab_newton" \
                -e "${ISAACLAB_PATH}/source/isaaclab_physx" \
                -e "${ISAACLAB_PATH}/source/isaaclab_visualizers" -q

    # Runtime deps for the visualizer backends we support out of the box:
    #   --viz rerun  -> rerun-sdk (already installed via this package's [viewer] extra)
    #   --viz newton -> imgui-bundle (pulled above) + PyOpenGL-accelerate (below)
    # (--viz kit needs a matching Isaac Sim build and is not set up here.)
    echo "  Enabling the rerun and newton visualizer backends..."
    pip install PyOpenGL-accelerate -q
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate with:"
echo "  source .venv-newton/bin/activate"
echo ""
echo "Run the standalone sensor example:"
echo "  python examples/standalone_newton.py"
echo ""
if [ -n "${ISAACLAB_PATH}" ]; then
    DEMO="${SCRIPT_DIR}/examples/isaaclab_task.py"
    echo "Isaac Lab installed. Run an Isaac Lab example (from your Isaac Lab checkout):"
    echo "  cd ${ISAACLAB_PATH} && ./isaaclab.sh -p ${DEMO} --viz rerun"
    echo "  (visualizer backends: --viz rerun | newton | none)"
    echo "  NOTE: with Isaac Sim 6.0.1 export the Isaac Sim env vars first —"
    echo "        see 'Examples' in the README."
    echo ""
else
    echo "For the Isaac Lab examples, re-run with:  ./setup_newton.sh --with-isaaclab /path/to/IsaacLab"
    echo ""
fi
echo "Newton version: $(python -c 'import newton; print(newton.__version__)')"
