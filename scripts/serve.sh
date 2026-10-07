#!/usr/bin/env bash
# Start `vllm serve` on the simulated device or on a GPU, with identical settings.
#
#   scripts/serve.sh openreg [extra vllm serve args]
#   scripts/serve.sh cuda    [extra vllm serve args]
#
# Environment: MODEL (default ~/models/Qwen3-0.6B), PORT (default 8000),
#              CUDA_VISIBLE_DEVICES (cuda only, default 0),
#              TRITON_F32_DEFAULT (cuda only, default ieee).
set -euo pipefail

target="${1:-}"
[ $# -gt 0 ] && shift
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

MODEL="${MODEL:-$HOME/models/Qwen3-0.6B}"
PORT="${PORT:-8000}"

case "$target" in
    openreg)
        export VLLM_OPENREG=1
        # Hide every GPU: the reference GPU must stay out of this run, and vLLM
        # only turns its Triton kernels off when Triton finds no usable GPU.
        # It has to be -1; an empty value makes vLLM keep Triton enabled.
        export CUDA_VISIBLE_DEVICES=-1
        ;;
    cuda)
        unset VLLM_OPENREG
        # OpenReg must not be loaded at all in a CUDA run (see env.sh).
        export TORCH_DEVICE_BACKEND_AUTOLOAD=0
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        # With --dtype float32 vLLM picks its Triton attention backend, whose
        # dot products default to TF32 (10-bit mantissa). Ask for real float32,
        # otherwise the reference itself is off by ~5e-3 in logprob.
        export TRITON_F32_DEFAULT="${TRITON_F32_DEFAULT:-ieee}"
        ;;
    *)
        echo "usage: $0 openreg|cuda [vllm serve args...]" >&2
        exit 2
        ;;
esac

# No telemetry and, for a model directory on disk, no Hugging Face lookups.
export VLLM_NO_USAGE_STATS=1
if [ -d "$MODEL" ]; then
    export HF_HUB_OFFLINE=1
fi

# Everything that can change the generated tokens is fixed here for both targets,
# including the scheduler limits that vLLM otherwise derives from the device.
exec vllm serve "$MODEL" \
    --served-model-name model \
    --host 127.0.0.1 --port "$PORT" \
    --dtype float32 \
    --enforce-eager \
    --seed 0 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --kv-cache-memory-bytes $((512 * 1024 * 1024)) \
    "$@"
