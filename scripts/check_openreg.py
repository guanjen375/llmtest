"""Step-1 smoke test: torch_openreg is loadable and can compute on its device."""

import torch

import torch_openreg  # noqa: F401  (also autoloaded by `import torch`)

dev = torch.device("openreg")
print(f"torch {torch.__version__}, openreg devices: {torch.openreg.device_count()}")

a_cpu = torch.randn(64, 128)
b_cpu = torch.randn(128, 32)
a, b = a_cpu.to(dev), b_cpu.to(dev)
assert a.device.type == "openreg" and b.device.type == "openreg"

c = a @ b
assert c.device.type == "openreg", c.device
diff = (c.cpu() - a_cpu @ b_cpu).abs().max().item()
print(f"matmul on {c.device}: shape={tuple(c.shape)}, max|openreg - cpu| = {diff:.3e}")
assert diff == 0.0, "openreg falls back to the CPU kernels, results must be identical"

# Device memory is mprotect()ed: host code cannot read it without an explicit copy.
try:
    c.numpy()
except (TypeError, RuntimeError) as e:
    print(f"direct host access is refused as expected: {type(e).__name__}")
else:
    raise AssertionError("expected .numpy() on an openreg tensor to fail")

print("OK")
