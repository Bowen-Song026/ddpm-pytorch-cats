import sys
import os
import time
import json

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

def main():
    print("=" * 60, flush=True)
    print("Sampling Steps Test")
    print("=" * 60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = Unet(dim=64, dim_mults=(1,2,4,8), flash_attn=False).to(device)
    model_path = os.path.join(base_dir, 'mixed_cat_results', 'model-final.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    sampling_steps_list = [25, 50, 100, 250, 500]
    
    results = []
    
    for sampling_steps in sampling_steps_list:
        print(f"\nTesting sampling_steps={sampling_steps}", flush=True)
        
        diffusion = GaussianDiffusion(model, image_size=64, timesteps=1000, sampling_timesteps=sampling_steps, objective='pred_v').to(device)
        diffusion.eval()
        
        start_time = time.time()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                sample = diffusion.sample(batch_size=4)
        elapsed = time.time() - start_time
        
        results_folder = Path(f'./sampling_test_{sampling_steps}')
        results_folder.mkdir(exist_ok=True)
        utils.save_image((sample+1)*0.5, str(results_folder / 'sample.png'), nrow=2)
        
        result = {
            'sampling_steps': sampling_steps,
            'time_seconds': elapsed,
            'time_per_image': elapsed / 4,
            'image_path': str(results_folder / 'sample.png')
        }
        results.append(result)
        
        print(f"  Time: {elapsed:.2f} seconds ({elapsed/4:.3f}s per image)", flush=True)

    with open('./sampling_steps_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("Results Summary")
    print("=" * 60, flush=True)
    print(f"{'Sampling Steps':<20} {'Total Time':<15} {'Time per Image':<20}", flush=True)
    for r in results:
        print(f"{r['sampling_steps']:<20} {r['time_seconds']:<15.2f} {r['time_per_image']:<20.3f}", flush=True)

if __name__ == '__main__':
    main()