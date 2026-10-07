"""Kernels the openreg device needs beyond what OpenReg and vLLM provide."""

import torch

# ----------------------------------------------------------------------------
# aten operators, registered for the PrivateUse1 dispatch key on import
# ----------------------------------------------------------------------------

# OpenReg runs nearly every operator through PyTorch's generic CPU fallback, and
# that fallback puts the result on the device of the *first* tensor argument.
# `scalar - tensor` is evaluated as sub(scalar_as_cpu_tensor, tensor), so on
# OpenReg its result lands on the CPU. vLLM's sampler computes `1 - top_p`.
_aten = torch.library.Library("aten", "IMPL")  # must stay referenced


def _rsub_scalar(self: torch.Tensor, other, alpha=1) -> torch.Tensor:
    """``other - alpha * self`` with the device tensor first in every op."""
    return torch.add(torch.mul(self, -alpha), other)


_aten.impl("rsub.Scalar", _rsub_scalar, "PrivateUse1")


# The CPU fallback moves tensors to the CPU but passes a generator through as it
# is, and the CPU kernel refuses a generator tagged with another device. An
# OpenReg generator is a CPU generator with an openreg device tag, so its state
# can drive a CPU one. vLLM samples a request that has its own seed this way.
def _exponential_(self: torch.Tensor, lambd=1.0, *, generator=None) -> torch.Tensor:
    host = torch.empty(self.shape, dtype=self.dtype)
    if generator is None or generator.device.type == "cpu":
        host.exponential_(lambd, generator=generator)
    else:
        host_generator = torch.Generator()
        host_generator.set_state(generator.get_state())
        host.exponential_(lambd, generator=host_generator)
        generator.set_state(host_generator.get_state())
    return self.copy_(host)


_aten.impl("exponential_", _exponential_, "PrivateUse1")


# ----------------------------------------------------------------------------
# Replacements for the Triton kernels vLLM launches outside the model
# ----------------------------------------------------------------------------
# Triton only targets GPUs, so OpenRegPlatform.register_triton_kernel_overrides()
# swaps these in. The parameter names must match the Triton kernel's exactly:
# vLLM then forwards the launch arguments by keyword.


def compute_slot_mapping(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc_ptr: torch.Tensor,  # [num_reqs + 1], int32
    positions_ptr: torch.Tensor,  # [num_tokens], int64
    block_table_ptr: torch.Tensor,  # [max_num_reqs, max_num_blocks_per_req], int32
    block_table_stride: int,
    block_size: int,
    slot_mapping_ptr: torch.Tensor,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: int,
    BLOCKS_PER_KV_BLOCK: int,
    TOTAL_CP_WORLD_SIZE: int,
    TOTAL_CP_RANK: int,
    CP_KV_CACHE_INTERLEAVE_SIZE: int,
    PAD_ID: int,
    BLOCK_SIZE: int,
) -> None:
    """Replaces vllm.v1.worker.block_table.ComputeSlotMappingKernel.kernel.

    For every scheduled token, turn (request, position) into its KV cache slot:
    ``block_table[request, position // block_size] * block_size + position % block_size``.
    """
    assert TOTAL_CP_WORLD_SIZE == 1, "context parallelism is not supported on openreg"
    assert BLOCKS_PER_KV_BLOCK == 1 and block_size == KV_CACHE_BLOCK_SIZE, (
        "the openreg attention backend accepts any block size, so the KV cache "
        "manager's blocks are never split into smaller kernel blocks"
    )

    query_start_loc = query_start_loc_ptr.long()
    num_reqs = query_start_loc.shape[0] - 1
    req_of_token = torch.repeat_interleave(
        torch.arange(num_reqs, device=query_start_loc.device),
        query_start_loc[1:] - query_start_loc[:-1],
    )
    positions = positions_ptr[:num_tokens]
    block_numbers = block_table_ptr[req_of_token, positions // block_size].long()
    slot_mapping_ptr[:num_tokens] = block_numbers * block_size + positions % block_size
    # The tail is padding (used by CUDA graphs); mark it so that it is never written.
    slot_mapping_ptr[num_tokens:max_num_tokens] = PAD_ID
