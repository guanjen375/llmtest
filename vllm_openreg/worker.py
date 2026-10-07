"""Worker for the openreg device.

vLLM's GPU worker and its V1 model runner are reused as they are: everything they
do through torch.accelerator / torch.Event already works on OpenReg. Only the
CUDA-specific parts of device bring-up are replaced here.
"""

from contextlib import AbstractContextManager, nullcontext

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager


class OpenRegWorker(Worker):
    def init_device(self) -> None:
        self.device = torch.device(f"openreg:{self.local_rank}")
        current_platform.set_device(self.device)
        current_platform.check_if_supports_dtype(self.model_config.dtype)

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        set_random_seed(self.model_config.seed)
        init_workspace_manager(self.device, num_ubatches=1)

        # Without Triton (the platform insists on that) vLLM itself selects V1.
        assert not self.use_v2_model_runner, "model runner V2 needs Triton"
        self.model_runner = GPUModelRunner(self.vllm_config, self.device)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # The base class would route a non-CUDA device to the CuMem allocator.
        return nullcontext()

    def determine_available_memory(self) -> int:
        # The base class profiles a forward pass against the device's free memory.
        # OpenReg reports none, so the platform fixed the KV cache budget instead.
        kv_cache_memory_bytes = self.cache_config.kv_cache_memory_bytes
        assert kv_cache_memory_bytes is not None
        return kv_cache_memory_bytes
