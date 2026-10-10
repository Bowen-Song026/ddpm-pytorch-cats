import sys
import os
import importlib
import time

# Pre-load modules to avoid circular import deadlock
print("Pre-loading modules...", flush=True)

# First, import attend and version directly
from denoising_diffusion_pytorch import attend as _attend
from denoising_diffusion_pytorch import version as _version
print("Attend and version loaded", flush=True)

# Now import the main classes
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion
print("Unet and GaussianDiffusion loaded", flush=True)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T, utils
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA

class OxfordCatDataset(Dataset):
    def __init__(self, folder, image_size=64):
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(folder).glob('*.png')] + [p for p in Path(folder).glob('*.jpg')]
        print(f"   Found {len(self.paths)} cat images", flush=True)

        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def main():
    print("=" * 60, flush=True)
    print("Oxford-IIIT Pet Cat Training (Standalone)", flush=True)
    print("=" * 60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f}GB", flush=True)

    cat_dir = './oxford_cat_images'
    if not os.path.exists(cat_dir):
        print(f"ERROR: {cat_dir} not found", flush=True)
        return

    print(f"\nLoading dataset from: {cat_dir}", flush=True)
    ds = OxfordCatDataset(cat_dir, image_size=64)
    dl = DataLoader(ds, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    dl_iter = cycle(dl)

    print("\nCreating model...", flush=True)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=50,
        objective='pred_v'
    ).to(device)

    optimizer = Adam(model.parameters(), lr=8e-5, betas=(0.9, 0.99))
    ema = EMA(model, beta=0.995, update_every=10).to(device)

    train_num_steps = 20000
    save_and_sample_every = 2000
    gradient_accumulate_every = 4
    max_grad_norm = 1.0
    amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    results_folder = Path('./oxford_cat_results_safe')
    results_folder.mkdir(exist_ok=True)

    print(f"\nConfig:", flush=True)
    print(f"  dim=64, image=64x64, batch=4, accum={gradient_accumulate_every}", flush=True)
    print(f"  steps={train_num_steps}, save_every={save_and_sample_every}", flush=True)
    print(f"  AMP={amp}, EMA=0.995, num_workers=0", flush=True)

    start_step = 0
    checkpoint_path = results_folder / 'model-latest.pt'
    if checkpoint_path.exists():
        print(f"\nResuming from {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ema.load_state_dict(checkpoint['ema'])
        start_step = checkpoint['step']
        print(f"Resumed from step {start_step}", flush=True)

    print(f"\nStarting training ({start_step} -> {train_num_steps})...\n", flush=True)

    model.train()

    pbar = tqdm(range(start_step, train_num_steps), desc='Training', unit='step')
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(gradient_accumulate_every):
            data = next(dl_iter).to(device)

            with torch.cuda.amp.autocast(enabled=amp):
                loss = diffusion(data)
                loss = loss / gradient_accumulate_every

            scaler.scale(loss).backward()
            total_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        pbar.set_postfix(loss=f"{total_loss:.4f}")

        if step % 100 == 0:
            print(f"  step {step}/{train_num_steps} | loss: {total_loss:.4f}", flush=True)

        if step > 0 and step % save_and_sample_every == 0:
            print(f"\n  [SAVE] Step {step} - generating sample...", flush=True)
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict(),
            }, results_folder / 'model-latest.pt')
            torch.save(model.state_dict(), results_folder / f'model-{step//save_and_sample_every}.pt')

            ema.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp):
                    sample = diffusion.sample(batch_size=1)
                sample = (sample + 1) * 0.5
                utils.save_image(sample, str(results_folder / f'sample-{step//save_and_sample_every}.png'))

            model.train()
            torch.cuda.empty_cache()

    print(f"\nTraining complete! Generating final samples...", flush=True)
    ema.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            final_samples = diffusion.sample(batch_size=4)
        final_samples = (final_samples + 1) * 0.5
        utils.save_image(final_samples, str(results_folder / 'final_samples.png'), nrow=2)

    torch.save(model.state_dict(), results_folder / 'model-final.pt')
    print(f"\nResults saved to: {results_folder}", flush=True)

if __name__ == '__main__':
    main()
