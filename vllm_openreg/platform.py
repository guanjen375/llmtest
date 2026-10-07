"""vLLM platform for OpenReg, PyTorch's reference PrivateUse1 backend.

OpenReg simulates an accelerator on the host CPU: device memory is mprotect()ed
host memory, and every aten op without a native kernel is copied to the CPU,
computed there and copied back. Slow, but functionally complete.
"""

import os
from typing import TYPE_CHECKING

import torch
import torch_openreg  # noqa: F401  (names PrivateUse1 "openreg", adds torch.openreg)

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

from . import ENV_SWITCH, kernels

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger(__name__)

# The simulated device reports no capacity (torch.accelerator.get_memory_info is
# unimplemented for OpenReg), so the KV cache gets a fixed budget instead of a
# share of "free device memory". Override with --kv-cache-memory-bytes.
DEFAULT_KV_CACHE_BYTES = 1 << 30


class OpenRegPlatform(Platform):
    _enum = PlatformEnum.OOT
    # vLLM builds torch.device(f"{device_name}:{rank}"), so both must be the torch name.
    device_name: str = "openreg"
    device_type: str = "openreg"
    dispatch_key: str = "PrivateUse1"
    # Only one device is supported here; TP > 1 would use OpenReg's own "occl".
    dist_backend: str = "gloo"
    # Keeps vLLM's standalone @torch.compile helpers away from inductor.
    simple_compile_backend: str = "eager"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        # The only dtype this plugin is validated with; "--dtype auto" resolves to it.
        return [torch.float32]

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        if dtype != torch.float32:
            raise ValueError(f"openreg only supports float32, got {dtype}")

    # ------------------------------------------------------------ configuration

    @classmethod
    def pre_register_and_update(
        cls, parser: "FlexibleArgumentParser | None" = None
    ) -> None:
        # Read while VllmConfig is built and again by torch in the engine process,
        # so it has to be in the environment before either happens.
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    @classmethod
    def validate_environ(cls, hard_fail: bool) -> None:
        # Platform.validate_environ, minus this plugin's own switch.
        for env in os.environ:
            if (
                env.startswith("VLLM_")
                and env != ENV_SWITCH
                and env not in envs.environment_variables
            ):
                if hard_fail:
                    raise ValueError(f"Unknown vLLM environment variable detected: {env}")
                logger.warning("Unknown vLLM environment variable detected: %s", env)

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        # Same effect as --enforce-eager. This hook runs before VllmConfig derives
        # its compilation defaults, so nothing downstream sees a compiled setup.
        from vllm.config import CompilationMode, CUDAGraphMode

        if vllm_config.model_config is not None:
            vllm_config.model_config.enforce_eager = True
        vllm_config.compilation_config.mode = CompilationMode.NONE
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        vllm_config.kernel_config.enable_jit_warmup = False

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # Import lazily: vllm.triton_utils needs current_platform, which is still
        # being resolved while this module is imported.
        from vllm.triton_utils import HAS_TRITON

        if HAS_TRITON:
            # vLLM would pick model runner V2 and the Triton sampler, and Triton
            # would launch GPU kernels on pointers into OpenReg's memory.
            raise RuntimeError(
                "Triton is active in this process, but openreg cannot run Triton "
                "kernels. Start with CUDA_VISIBLE_DEVICES=-1 (an empty value is not "
                "enough: vLLM then keeps Triton enabled)."
            )

        parallel_config = vllm_config.parallel_config
        if parallel_config.world_size > 1:
            raise NotImplementedError("vllm-openreg runs on a single openreg device")
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_openreg.worker.OpenRegWorker"

        # The runner's async path is written against torch.cuda streams and events.
        vllm_config.scheduler_config.async_scheduling = False

        cache_config = vllm_config.cache_config
        if cache_config.kv_cache_memory_bytes is None:
            cache_config.kv_cache_memory_bytes = DEFAULT_KV_CACHE_BYTES

        # Like CUDA, OpenReg cannot be initialised again in a forked child.
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # ------------------------------------------------------------ kernels

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if attn_selector_config.use_mla or attn_selector_config.use_sparse:
            raise NotImplementedError("openreg only has a plain paged attention backend")
        return "vllm_openreg.attention.OpenRegAttentionBackend"

    @classmethod
    def import_kernels(cls) -> None:
        """vLLM's compiled kernels target CUDA; this device uses none of them."""

    @classmethod
    def register_triton_kernel_overrides(cls) -> None:
        from vllm.triton_utils.dispatcher import register_kernels

        register_kernels(
            {
                "vllm.v1.worker.block_table.ComputeSlotMappingKernel.kernel": (
                    kernels.compute_slot_mapping
                ),
            }
        )

    # ------------------------------------------------------------ device queries

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "openreg"

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        return torch.accelerator.memory_allocated(device)

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return torch.get_num_threads()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.openreg.set_device(device.index or 0)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.openreg.manual_seed_all(seed)
