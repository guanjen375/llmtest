# Source this file (setup.sh and scripts/*.sh do). It only sets up the shell:
#   - project-local tools (.tools/bin) and the venv (.venv) go on PATH
#   - TLS clients are pointed at the system CA store instead of their bundled ones
# shellcheck shell=bash

OPENREG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENREG_ROOT

# uv ships its own root store and Python uses certifi; behind a TLS-inspecting
# proxy only the system store knows the proxy CA. (git and curl already use it.)
if [ -z "${SSL_CERT_FILE:-}" ]; then
    for _ca in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt; do
        if [ -r "$_ca" ]; then
            export SSL_CERT_FILE="$_ca"
            break
        fi
    done
    unset _ca
fi
if [ -n "${SSL_CERT_FILE:-}" ]; then
    export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
fi
# UV_SYSTEM_CERTS is the current name, UV_NATIVE_TLS the pre-0.12 one.
export UV_SYSTEM_CERTS=1 UV_NATIVE_TLS=1

# `import torch` loads every installed device backend, and a loaded OpenReg
# becomes torch.accelerator's device even when CUDA is available, which breaks
# vLLM on a GPU ("cuda doesn't match the current accelerator openreg"). The
# plugin imports torch_openreg itself when VLLM_OPENREG=1, so autoload stays off.
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

case ":$PATH:" in
    *":$OPENREG_ROOT/.tools/bin:"*) ;;
    *) export PATH="$OPENREG_ROOT/.tools/bin:$PATH" ;;
esac
if [ -f "$OPENREG_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$OPENREG_ROOT/.venv/bin/activate"
fi
