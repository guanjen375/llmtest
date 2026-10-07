#!/usr/bin/env bash
# Build the whole vllm-openreg environment under this directory.
# No sudo, nothing outside this directory except uv's own cache/python dirs.
# Idempotent: re-run it after pulling changes or changing a knob.
#
#   ./setup.sh
#   TORCH_BACKEND=cu130 ./setup.sh     # override uv's --torch-backend
#
# Knobs (environment variables):
#   PYTHON_VERSION  python for the venv              (default 3.12)
#   VLLM_SPEC       requirement passed to uv         (default vllm==0.31.0)
#   TORCH_BACKEND   uv --torch-backend               (default auto)
#   PYTORCH_REPO    where to clone PyTorch from      (default github.com/pytorch/pytorch)
#   UV_VERSION      uv release to download if no uv  (default latest)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
# The plugin subclasses vLLM internals (worker, model runner, attention
# backend), so it is validated against one vLLM release; bump deliberately.
VLLM_SPEC="${VLLM_SPEC:-vllm==0.31.0}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"
PYTORCH_REPO="${PYTORCH_REPO:-https://github.com/pytorch/pytorch.git}"

VENV="$ROOT/.venv"
TOOLS="$ROOT/.tools/bin"
PT_SRC="$ROOT/third_party/pytorch"
OPENREG_DIR="test/cpp_extensions/open_registration_extension"

log() { printf '\n[setup] %s\n' "$*"; }
die() { printf '\n[setup] ERROR: %s\n' "$*" >&2; exit 1; }

# shellcheck disable=SC1091
. "$ROOT/env.sh"

# ---------------------------------------------------------------- 0. prerequisites
for tool in git curl tar c++; do
    command -v "$tool" >/dev/null 2>&1 \
        || die "'$tool' not found. It cannot be installed without root from here; ask for it (c++ = g++ or clang++)."
done

# ---------------------------------------------------------------- 1. uv
if ! command -v uv >/dev/null 2>&1; then
    case "$(uname -m)" in
        x86_64) uv_arch=x86_64 ;;
        aarch64 | arm64) uv_arch=aarch64 ;;
        *) die "no uv binary for $(uname -m); install uv manually and re-run" ;;
    esac
    if [ -n "${UV_VERSION:-}" ]; then
        uv_url="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$uv_arch-unknown-linux-gnu.tar.gz"
    else
        uv_url="https://github.com/astral-sh/uv/releases/latest/download/uv-$uv_arch-unknown-linux-gnu.tar.gz"
    fi
    log "uv not on PATH, downloading it into $TOOLS"
    mkdir -p "$TOOLS"
    curl -fsSL "$uv_url" | tar -xz -C "$TOOLS" --strip-components=1
fi
log "using $(uv --version) ($(command -v uv))"

# ---------------------------------------------------------------- 2. venv + vLLM
if [ ! -x "$VENV/bin/python" ]; then
    log "creating venv (python $PYTHON_VERSION) at $VENV"
    uv venv --python "$PYTHON_VERSION" "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"

log "uv pip install $VLLM_SPEC --torch-backend=$TORCH_BACKEND"
uv pip install "$VLLM_SPEC" --torch-backend="$TORCH_BACKEND"

# ---------------------------------------------------------------- 3. PyTorch source at the installed torch's tag
TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
TORCH_SHA="$(python -c 'import torch; print(torch.version.git_version)')"
TORCH_TAG="v${TORCH_VERSION%%+*}"
log "installed torch $TORCH_VERSION (commit $TORCH_SHA) -> PyTorch tag $TORCH_TAG"

if [ -d "$PT_SRC/.git" ]; then
    have="$(git -C "$PT_SRC" rev-parse HEAD)"
    [ "$have" = "$TORCH_SHA" ] \
        || die "$PT_SRC is at $have, but torch was built from $TORCH_SHA. Remove that directory and re-run."
else
    log "sparse-cloning $PYTORCH_REPO @ $TORCH_TAG (only $OPENREG_DIR)"
    mkdir -p "$(dirname "$PT_SRC")"
    git clone --depth 1 --branch "$TORCH_TAG" --filter=blob:none --sparse \
        -c advice.detachedHead=false "$PYTORCH_REPO" "$PT_SRC"
    git -C "$PT_SRC" sparse-checkout set "$OPENREG_DIR"
    have="$(git -C "$PT_SRC" rev-parse HEAD)"
    [ "$have" = "$TORCH_SHA" ] \
        || die "tag $TORCH_TAG is $have, but torch reports commit $TORCH_SHA; not building against mismatched sources."
fi

# ---------------------------------------------------------------- 4. build torch_openreg against that torch
OPENREG_SRC="$PT_SRC/$OPENREG_DIR/torch_openreg"
log "building torch_openreg from $OPENREG_SRC"

# cmake and ninja come from PyPI, so the host only has to provide the compiler.
uv pip install setuptools wheel cmake ninja

if [ "$(python -c 'import torch; print(torch.version.cuda or "")')" != "" ] \
    && ! command -v nvcc >/dev/null 2>&1 && [ ! -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    log "WARNING: torch is a CUDA build and no CUDA toolkit (nvcc) was found."
    log "         CMake's find_package(Torch) needs one, so the build below will likely fail."
    log "         On a host without the toolkit: remove $VENV and re-run with TORCH_BACKEND=cpu."
fi

# torch_openreg's setup.py runs cmake itself with a fixed argument list, and the
# project's USE_TEST option (default ON) needs the googletest submodule, which
# the sparse clone does not have. Configure a fresh build directory first: the
# option is then OFF in the CMake cache when setup.py configures it again.
rm -rf "$OPENREG_SRC/build"
cmake -S "$OPENREG_SRC" -B "$OPENREG_SRC/build" -G Ninja \
    -DUSE_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX="$OPENREG_SRC/torch_openreg" \
    -DPYTHON_INCLUDE_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
    -DPYTORCH_INSTALL_DIR="$(python -c 'import os, torch; print(os.path.dirname(os.path.realpath(torch.__file__)))')"

# --no-build-isolation: compile against the venv's torch, not a fresh one from PyPI.
# --reinstall/--refresh: never reuse a wheel that was built against another torch.
uv pip install --no-build-isolation --no-deps \
    --reinstall-package torch-openreg --refresh-package torch-openreg "$OPENREG_SRC"

# ---------------------------------------------------------------- 5. the plugin itself
log "installing the vllm-openreg plugin (editable)"
uv pip install --no-build-isolation --no-deps -e "$ROOT"

# ---------------------------------------------------------------- 6. smoke tests
log "checking the openreg device"
python "$ROOT/scripts/check_openreg.py"
VLLM_OPENREG=1 CUDA_VISIBLE_DEVICES=-1 python -c '
from vllm.platforms import current_platform
assert current_platform.device_type == "openreg", current_platform
print("vLLM platform with VLLM_OPENREG=1:", type(current_platform).__name__)'

log "done. Next: scripts/serve.sh openreg   (MODEL=/path/to/model to override the default)"
