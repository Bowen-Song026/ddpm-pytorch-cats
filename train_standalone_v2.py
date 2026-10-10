import sys
import os
import time

print("Step 1: Adding paths...", flush=True)

# Add the package directory to path so we can import modules directly
pkg_dir = os.path.join(os.path.dirname(__file__), 'denoising_diffusion_pytorch')
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

# Create a fake package module to avoid __init__.py
print("Step 2: Creating fake package...", flush=True)
import types
pkg = types.ModuleType('denoising_diffusion_pytorch')
pkg.__path__ = [pkg_dir]
sys.modules['denoising_diffusion_pytorch'] = pkg

# Now import submodules directly
print("Step 3: Importing attend...", flush=True)
import attend
pkg.attend = attend
print("Step 3a: attend OK", flush=True)

print("Step 4: Importing version...", flush=True)
import version
pkg.version = version
print("Step 4a: version OK", flush=True)

# Now import the main module file directly
print("Step 5: Importing main module...", flush=True)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "denoising_diffusion_pytorch.denoising_diffusion_pytorch",
    os.path.join(pkg_dir, "denoising_diffusion_pytorch.py")
)
main_mod = importlib.util.module_from_spec(spec)
sys.modules["denoising_diffusion_pytorch.denoising_diffusion_pytorch"] = main_mod
spec.loader.exec_module(main_mod)
pkg.denoising_diffusion_pytorch = main_mod
print("Step 5a: main module OK", flush=True)

Unet = main_mod.Unet
GaussianDiffusion = main_mod.GaussianDiffusion
print(f"Step 6: Got classes: Unet={Unet}, GaussianDiffusion={GaussianDiffusion}", flush=True)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
from torchvision import transforms as T, utils, datasets
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA

class OxfordCatDataset(Dataset):
    """Oxford-IIIT Pet: 大写开头=猫, 小写开头=狗"""
    def __init__(self, oxford_dir, image_size=64):
        self.image_size = image_size
        self.paths = []
        for f in os.listdir(oxford_dir):
            if f.endswith('.jpg') and f[0].isupper():
                self.paths.append(os.path.join(oxford_dir, f))
        print(f"   Oxford cats: {len(self.paths)} images", flush=True)
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
    """CIFAR-10: label=3 is cat"""
    def __init__(self, root='./data', image_size=64, train=True):
        self.cifar = datasets.CIFAR10(root=root, train=train, download=True, transform=None)
        self.cat_indices = [i for i, (_, label) in enumerate(self.cifar) if label == 3]
        print(f"   CIFAR-10 cats: {len(self.cat_indices)} images", flush=True)
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

def main():
    print("=" * 60, flush=True)
    print("Mixed Dataset Training (Oxford + CIFAR-10)")
    print("=" * 60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB", flush=True)

    # Dataset 1: Oxford-IIIT Pet (all cats)
    oxford_dir = './oxford_pets_data/images'
    print(f"\nLoading Oxford-IIIT Pet: {oxford_dir}", flush=True)
    oxford_ds = OxfordCatDataset(oxford_dir, image_size=64)

    # Dataset 2: CIFAR-10 cats
    print(f"\nLoading CIFAR-10 cats...", flush=True)
    cifar_ds = CIFAR10CatDataset(root='./data', image_size=64, train=True)

    # Combine
    combined_ds = ConcatDataset([oxford_ds, cifar_ds])
    print(f"\nCombined dataset: {len(combined_ds)} images", flush=True)

    dl = DataLoader(combined_ds, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    dl_iter = cycle(dl)

    print("\nCreating model...", flush=True)
    model = Unet(dim=64, dim_mults=(1,2,4,8), flash_attn=False).to(device)
    diffusion = GaussianDiffusion(model, image_size=64, timesteps=1000, sampling_timesteps=50, objective='pred_v').to(device)

    optimizer = Adam(model.parameters(), lr=8e-5, betas=(0.9,0.99))
    ema = EMA(model, beta=0.995, update_every=10).to(device)

    train_num_steps = 20000
    save_and_sample_every = 2000
    gradient_accumulate_every = 4
    max_grad_norm = 1.0
    amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    results_folder = Path('./mixed_cat_results')
    results_folder.mkdir(exist_ok=True)

    print(f"\nConfig: dim=64, 64x64, batch=4x{gradient_accumulate_every}, steps={train_num_steps}", flush=True)

    start_step = 0
    checkpoint_path = results_folder / 'model-latest.pt'
    if checkpoint_path.exists():
        print(f"Resuming from {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.load_state_dict(ckpt['ema'])
        start_step = ckpt['step']
        print(f"Resumed from step {start_step}", flush=True)

    print(f"\nTraining ({start_step} -> {train_num_steps})...\n", flush=True)
    model.train()

    for step in tqdm(range(start_step, train_num_steps), desc='Training', unit='step'):
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(gradient_accumulate_every):
            data = next(dl_iter).to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = diffusion(data) / gradient_accumulate_every
            scaler.scale(loss).backward()
            total_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        if step % 100 == 0:
            print(f"  step {step}/{train_num_steps} | loss: {total_loss:.4f}", flush=True)

        if step > 0 and step % save_and_sample_every == 0:
            print(f"\n  [SAVE] Step {step} - saving model...", flush=True)
            torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'ema': ema.state_dict()}, results_folder / 'model-latest.pt')
            torch.save(model.state_dict(), results_folder / f'model-{step//save_and_sample_every}.pt')
            torch.save(ema.state_dict(), results_folder / f'ema-{step//save_and_sample_every}.pt')
            torch.cuda.empty_cache()

    print(f"\nDone! Saving final model...", flush=True)
    torch.save({'step': train_num_steps, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'ema': ema.state_dict()}, results_folder / 'model-latest.pt')
    torch.save(model.state_dict(), results_folder / 'model-final.pt')
    torch.save(ema.state_dict(), results_folder / 'ema-final.pt')
    print(f"Results: {results_folder}", flush=True)

if __name__ == '__main__':
    main()
