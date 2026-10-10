import sys
import os

pkg_dir = os.path.join(os.path.dirname(__file__), 'denoising_diffusion_pytorch')
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

import types
pkg = types.ModuleType('denoising_diffusion_pytorch')
pkg.__path__ = [pkg_dir]
sys.modules['denoising_diffusion_pytorch'] = pkg

import attend
pkg.attend = attend

import version
pkg.version = version

import importlib.util
spec = importlib.util.spec_from_file_location(
    "denoising_diffusion_pytorch.denoising_diffusion_pytorch",
    os.path.join(pkg_dir, "denoising_diffusion_pytorch.py")
)
main_mod = importlib.util.module_from_spec(spec)
sys.modules["denoising_diffusion_pytorch.denoising_diffusion_pytorch"] = main_mod
spec.loader.exec_module(main_mod)

Unet = main_mod.Unet
GaussianDiffusion = main_mod.GaussianDiffusion

import torch
from torchvision import utils
from pathlib import Path
from ema_pytorch import EMA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

model = Unet(dim=128, dim_mults=(1,2,4), flash_attn=False).to(device)
diffusion = GaussianDiffusion(model, image_size=64, timesteps=1000, sampling_timesteps=100, objective='pred_v').to(device)

model_path = './dim128_cat_results/model-final.pt'
ema_path = './dim128_cat_results/ema-final.pt'

print(f"Loading model from {model_path}", flush=True)
model.load_state_dict(torch.load(model_path, map_location=device))

print(f"Loading EMA from {ema_path}", flush=True)
ema = EMA(model, beta=0.995, update_every=10).to(device)
ema.load_state_dict(torch.load(ema_path, map_location=device))

ema.eval()
model.eval()

print("Generating images (sampling_timesteps=100)...", flush=True)
with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=True):
        for i in range(5):
            sample = diffusion.sample(batch_size=4)
            save_path = f'./dim128_cat_results/highres_{i+1}.png'
            utils.save_image((sample+1)*0.5, save_path, nrow=2)
            print(f"  Saved: {save_path}", flush=True)
        final = diffusion.sample(batch_size=8)
        save_path = './dim128_cat_results/highres_grid.png'
        utils.save_image((final+1)*0.5, save_path, nrow=4)
        print(f"  Saved: {save_path}", flush=True)

torch.cuda.empty_cache()
print("\nDone! All high-res images saved to dim128_cat_results/", flush=True)