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
pkg.denoising_diffusion_pytorch = main_mod

Unet = main_mod.Unet
GaussianDiffusion = main_mod.GaussianDiffusion

import torch
from torchvision import utils
from ema_pytorch import EMA
from pathlib import Path

def generate(
    batch_size=4,
    sampling_timesteps=50,
    ddim_eta=0.0,
    offset_noise=0.0,
    num_batches=5,
    rows=2
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    
    print(f"\n=== 生成参数 ===")
    print(f"  batch_size: {batch_size}")
    print(f"  sampling_timesteps: {sampling_timesteps}")
    print(f"  ddim_eta: {ddim_eta}")
    print(f"  offset_noise: {offset_noise}")
    print(f"  num_batches: {num_batches}")
    print(f"================", flush=True)
    
    model = Unet(dim=64, dim_mults=(1,2,4,8), flash_attn=False).to(device)
    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps,
        objective='pred_v',
        ddim_sampling_eta=ddim_eta,
        offset_noise_strength=offset_noise
    ).to(device)
    ema = EMA(model, beta=0.995).to(device)
    
    results_folder = Path('./mixed_cat_results')
    
    ema_path = results_folder / 'ema-final.pt'
    if ema_path.exists():
        print(f"Loading EMA weights from {ema_path}", flush=True)
        ema.load_state_dict(torch.load(ema_path, map_location=device))
    else:
        print(f"Loading model weights", flush=True)
        model.load_state_dict(torch.load(results_folder / 'model-final.pt', map_location=device))
        ema = model
    
    ema.eval()
    
    print("\nGenerating cat images...", flush=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            for i in range(num_batches):
                samples = diffusion.sample(batch_size=batch_size)
                utils.save_image((samples+1)*0.5, str(results_folder / f'generated_cats_{i+1}.png'), nrow=rows)
                print(f"  Saved generated_cats_{i+1}.png", flush=True)
                torch.cuda.empty_cache()
    
    print(f"\nDone! Generated {batch_size * num_batches} images saved to {results_folder}", flush=True)

if __name__ == '__main__':
    generate()
