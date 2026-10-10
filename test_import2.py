import sys
import importlib.util
import time

print("Python:", sys.version)
print("Step 1: Loading module directly...", flush=True)

# Load the main module directly from file path
spec = importlib.util.spec_from_file_location(
    "ddp_module",
    "denoising_diffusion_pytorch/denoising_diffusion_pytorch.py"
)
mod = importlib.util.module_from_spec(spec)
print("Step 2: Module created", flush=True)

try:
    spec.loader.exec_module(mod)
    print("Step 3: Module loaded!", flush=True)
    Unet = mod.Unet
    GaussianDiffusion = mod.GaussianDiffusion
    print("Step 4: Got classes", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
