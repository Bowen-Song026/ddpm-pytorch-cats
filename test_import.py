import sys
import time

print("Python version:", sys.version)
print("Start import:", time.time())
sys.stdout.flush()

try:
    from denoising_diffusion_pytorch import Unet, GaussianDiffusion
    print("Import successful!", time.time())
except Exception as e:
    print("Import error:", e)
finally:
    sys.stdout.flush()
