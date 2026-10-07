"""The simplest possible paged attention, written with plain torch ops only.

No torch.nn.functional.scaled_dot_product_attention: OpenReg registers a stub
for it that returns uninitialised memory.

Per-layer KV cache, as vLLM hands it to the backend (a logical view, whatever the
physical layout): ``[num_blocks, num_kv_heads, block_size, 2 * head_size]``.
This backend stores the key in the first ``head_size`` entries of the last
dimension and the value in the rest.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout


@register_backend(AttentionBackendEnum.CUSTOM)
class OpenRegAttentionBackend(AttentionBackend):
    # vLLM calls impl.do_kv_cache_update() before impl.forward().
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float32]
    supported_kv_cache_dtypes: ClassVar[list] = ["auto"]

    @staticmethod
    def get_name() -> str:
        # The Attention layer looks this name up in AttentionBackendEnum; CUSTOM
        # is the member reserved for out-of-tree backends.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["OpenRegAttentionImpl"]:
        return OpenRegAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["OpenRegAttentionMetadataBuilder"]:
        return OpenRegAttentionMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Only the logical view is indexed, so any layout would do; the identity
        # layout keeps every per-layer view contiguous.
        return (KVCacheLayout.LBHNC,)


@dataclass
class OpenRegAttentionMetadata(AttentionMetadata):
    # Host-side copies: the loop bounds in forward() must be Python ints.
    query_start_loc: list[int]  # [num_reqs + 1] offset of each request's tokens
    seq_lens: list[int]  # [num_reqs] context length, this step's tokens included
    block_table: torch.Tensor  # [num_reqs, max_blocks_per_req] on the device


class OpenRegAttentionMetadataBuilder(AttentionMetadataBuilder[OpenRegAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> OpenRegAttentionMetadata:
        assert common_attn_metadata.causal is True, "only causal attention is implemented"
        return OpenRegAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc_cpu.tolist(),
            seq_lens=common_attn_metadata.seq_lens.cpu().tolist(),
            block_table=common_attn_metadata.block_table_tensor,
        )


class OpenRegAttentionImpl(AttentionImpl[OpenRegAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        unsupported = {
            "alibi_slopes": alibi_slopes,
            "sliding_window": sliding_window,
            "logits_soft_cap": logits_soft_cap,
            "kv_sharing_target_layer_name": kv_sharing_target_layer_name,
        }
        unsupported = {k: v for k, v in unsupported.items() if v is not None}
        if unsupported or attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"openreg attention: unsupported {unsupported or attn_type}"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,  # [num_tokens], slot = block * block_size + offset
    ) -> None:
        # Padding tokens, and every token of a warm-up run, carry slot -1.
        tokens = (slot_mapping >= 0).nonzero(as_tuple=True)[0]
        if tokens.numel() == 0:
            return
        slots = slot_mapping[tokens]
        block_size = kv_cache.shape[2]
        kv = torch.cat([key[tokens], value[tokens]], dim=-1)
        kv_cache[slots // block_size, :, slots % block_size] = kv.to(kv_cache.dtype)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: OpenRegAttentionMetadata | None,
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:  # warm-up run: there is no batch to attend over
            return output

        block_size = kv_cache.shape[2]
        head_size = self.head_size
        kv_group = self.num_heads // self.num_kv_heads
        starts = attn_metadata.query_start_loc

        # This step's keys and values are already in the cache, so each request
        # reads its whole context (prompt, earlier steps and now) back from its blocks.
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            start, end = starts[i], starts[i + 1]
            query_len = end - start
            if query_len == 0:
                continue
            num_blocks = -(-seq_len // block_size)
            blocks = kv_cache.index_select(
                0, attn_metadata.block_table[i, :num_blocks].long()
            )
            # [num_blocks, kv_heads, block_size, 2D] -> [kv_heads, seq_len, 2D]
            context = blocks.transpose(0, 1).reshape(self.num_kv_heads, -1, 2 * head_size)
            context = context[:, :seq_len].to(query.dtype)
            k, v = context[..., :head_size], context[..., head_size:]
            if kv_group > 1:  # grouped-query attention
                k = k.repeat_interleave(kv_group, dim=0)
                v = v.repeat_interleave(kv_group, dim=0)

            q = query[start:end].transpose(0, 1)  # [heads, query_len, D]
            scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
            if query_len > 1:
                # Query j is the token at position seq_len - query_len + j and
                # must not see the keys after it.
                future = torch.ones(
                    query_len, seq_len, dtype=torch.bool, device=query.device
                ).triu(seq_len - query_len + 1)
                scores = scores.masked_fill(future, float("-inf"))
            attended = torch.matmul(torch.softmax(scores, dim=-1), v)
            output[start:end] = attended.transpose(0, 1)
        return output
