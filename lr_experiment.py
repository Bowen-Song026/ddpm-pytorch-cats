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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
from torchvision import transforms as T, utils, datasets
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
import numpy as np

class OxfordCatDataset(Dataset):
    def __init__(self, oxford_dir, image_size=64):
        self.image_size = image_size
        self.paths = []
        for f in os.listdir(oxford_dir):
            if f.endswith('.jpg') and f[0].isupper():
                self.paths.append(os.path.join(oxford_dir, f))
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

class CIFAR10CatDataset(Dataset):
    def __init__(self, root='./data', image_size=64, train=True):
        self.cifar = datasets.CIFAR10(root=root, train=train, download=True, transform=None)
        self.cat_indices = [i for i, (_, label) in enumerate(self.cifar) if label == 3]
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    def __len__(self):
        return len(self.cat_indices)
    def __getitem__(self, idx):
        img, _ = self.cifar[self.cat_indices[idx]]
        return self.transform(img.convert('RGB'))

def cycle(dl):
    while True:
        for data in dl:
            yield data

def run_experiment(lr, exp_name):
    print(f"\n{'='*60}", flush=True)
    print(f"Experiment: {exp_name}")
    print(f"Learning Rate: {lr}")
    print(f"{'='*60}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    oxford_dir = os.path.join(base_dir, 'oxford_pets_data', 'images')
    oxford_ds = OxfordCatDataset(oxford_dir, image_size=64)
    cifar_ds = CIFAR10CatDataset(root=os.path.join(base_dir, 'data'), image_size=64, train=True)
    combined_ds = ConcatDataset([oxford_ds, cifar_ds])

    batch_size = 4
    grad_accum = 4
    dl = DataLoader(combined_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    dl_iter = cycle(dl)

    model = Unet(dim=64, dim_mults=(1,2,4,8), flash_attn=False).to(device)
    diffusion = GaussianDiffusion(model, image_size=64, timesteps=1000, sampling_timesteps=50, objective='pred_v').to(device)

    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9,0.99))
    ema = EMA(model, beta=0.995, update_every=10).to(device)

    train_num_steps = 2000
    max_grad_norm = 1.0
    amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    results_folder = Path(f'./lr_experiment_{exp_name}')
    results_folder.mkdir(exist_ok=True)

    start_time = time.time()
    losses = []
    model.train()

    for step in tqdm(range(train_num_steps), desc=f'Training {exp_name}', unit='step', mininterval=0.1, maxinterval=1.0):
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(grad_accum):
            data = next(dl_iter).to(device)
            with torch.amp.autocast('cuda', enabled=amp):
                loss = diffusion(data) / grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        if step % 50 == 0:
            losses.append(total_loss)
            progress = (step + 1) / train_num_steps * 100
            elapsed = time.time() - start_time
            eta = elapsed / (step + 1) * (train_num_steps - step - 1)
            print(f"  [{progress:.1f}%] step {step}/{train_num_steps} | loss: {total_loss:.4f} | elapsed: {elapsed:.0f}s | eta: {eta:.0f}s", flush=True)

    end_time = time.time()
    total_time = end_time - start_time

    torch.save(model.state_dict(), results_folder / 'model-final.pt')
    torch.save(ema.state_dict(), results_folder / 'ema-final.pt')

    ema.eval()
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            sample = diffusion.sample(batch_size=4)
            utils.save_image((sample+1)*0.5, str(results_folder / 'sample.png'), nrow=2)

    result = {
        'experiment': exp_name,
        'lr': lr,
        'dim': 64,
        'batch_size': batch_size,
        'gradient_accumulate': grad_accum,
        'effective_batch': batch_size * grad_accum,
        'train_steps': train_num_steps,
        'total_time': total_time,
        'steps_per_sec': train_num_steps / total_time,
        'final_loss': losses[-1] if losses else None,
        'loss_mean': np.mean(losses) if losses else None,
        'loss_std': np.std(losses) if losses else None,
        'losses': losses
    }

    with open(results_folder / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nExperiment {exp_name} completed!", flush=True)
    print(f"  Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Speed: {result['steps_per_sec']:.2f} steps/sec")
    print(f"  Final Loss: {result['final_loss']:.4f}")
    print(f"  Loss STD: {result['loss_std']:.4f}", flush=True)

    return result

def main():
    print("=" * 60, flush=True)
    print("Learning Rate Experiment")
    print("=" * 60, flush=True)

    experiments = [
        {'lr': 1e-5, 'name': 'lr_1e-5'},
        {'lr': 4e-5, 'name': 'lr_4e-5'},
        {'lr': 8e-5, 'name': 'lr_8e-5'},
        {'lr': 1e-4, 'name': 'lr_1e-4'},
        {'lr': 2e-4, 'name': 'lr_2e-4'},
    ]

    all_results = []
    for exp in experiments:
        result = run_experiment(exp['lr'], exp['name'])
        all_results.append(result)

    with open('./lr_experiment_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("Experiment Summary")
    print("=" * 60, flush=True)
    print(f"{'Experiment':<20} {'LR':<15} {'Speed':<10} {'Final Loss':<12} {'Loss STD':<10}", flush=True)
    for r in all_results:
        print(f"{r['experiment']:<20} {r['lr']:<15} {r['steps_per_sec']:<10.2f} {r['final_loss']:<12.4f} {r['loss_std']:<10.4f}", flush=True)

if __name__ == '__main__':
    main()