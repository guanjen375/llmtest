"""Out-of-tree vLLM platform plugin for OpenReg (PyTorch's simulated PrivateUse1 device)."""

import os
import sys

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_SWITCH = "VLLM_OPENREG"


def register() -> str | None:
    """Entry point for the ``vllm.platform_plugins`` group.

    vLLM calls this in every process. Returning a class path activates the
    platform (an out-of-tree platform wins over the built-in CUDA one); returning
    None leaves vLLM's own detection alone, so the same venv can still serve
    on a GPU when the switch is off.
    """
    if os.environ.get(ENV_SWITCH) != "1":
        # Do not import torch_openreg here: once loaded, OpenReg is the
        # process-wide torch.accelerator device and vLLM's CUDA platform breaks.
        if "torch_openreg" in sys.modules:
            logger.warning(
                "torch autoloaded torch_openreg, so torch.accelerator now points "
                "at OpenReg and serving on another device will fail. Set "
                "TORCH_DEVICE_BACKEND_AUTOLOAD=0 (env.sh does), or %s=1 to serve "
                "on OpenReg.",
                ENV_SWITCH,
            )
        return None
    try:
        import torch_openreg  # noqa: F401
    except Exception as e:  # ImportError, or OSError from a build against another torch
        logger.warning("%s=1 but torch_openreg cannot be imported: %r", ENV_SWITCH, e)
        return None
    return "vllm_openreg.platform.OpenRegPlatform"
