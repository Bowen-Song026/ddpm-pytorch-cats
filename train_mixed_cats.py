import sys
import os
import time
import importlib.util

# === 绕过 __init__.py 循环导入 ===
print("Loading modules...", flush=True)
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
print("Modules loaded OK!", flush=True)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
from torchvision import transforms as T, utils, datasets
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
import numpy as np


# === 数据集1: Oxford-IIIT Pet (修复版，获取全部555张猫图片) ===
class OxfordCatDataset(Dataset):
    def __init__(self, oxford_images_dir, image_size=64):
        self.image_size = image_size
        self.paths = []

        # Oxford数据集规则: 大写字母开头=猫, 小写字母开头=狗
        for f in os.listdir(oxford_images_dir):
            if f.endswith('.jpg') and f[0].isupper():
                self.paths.append(os.path.join(oxford_images_dir, f))

        print(f"   Oxford-IIIT Pet cats: {len(self.paths)} images", flush=True)

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
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


# === 数据集2: CIFAR-10 猫图片 (5000张) ===
class CIFAR10CatDataset(Dataset):
    def __init__(self, root='./data', image_size=64, train=True):
        self.image_size = image_size

        # 加载CIFAR-10
        self.cifar = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=None  # 先不transform，手动处理
        )

        # 只保留猫 (label=3)
        self.cat_indices = [i for i, (_, label) in enumerate(self.cifar) if label == 3]
        print(f"   CIFAR-10 cats: {len(self.cat_indices)} images", flush=True)

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.cat_indices)

    def __getitem__(self, idx):
        real_idx = self.cat_indices[idx]
        img, _ = self.cifar[real_idx]
        img = img.convert('RGB')  # PIL Image
        return self.transform(img)


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

    # === 加载两个数据集 ===
    print("\n--- Loading Dataset 1: Oxford-IIIT Pet ---", flush=True)
    oxford_dir = './oxford_pets_data/images'
    if not os.path.exists(oxford_dir):
        print(f"ERROR: {oxford_dir} not found", flush=True)
        return
    oxford_ds = OxfordCatDataset(oxford_dir, image_size=64)

    print("\n--- Loading Dataset 2: CIFAR-10 ---", flush=True)
    cifar_ds = CIFAR10CatDataset(root='./data', image_size=64, train=True)

    # === 合并数据集 ===
    combined_ds = ConcatDataset([oxford_ds, cifar_ds])
    print(f"\n--- Combined Dataset: {len(combined_ds)} images ---", flush=True)
    print(f"   Oxford: {len(oxford_ds)} (high quality 600x400 -> 64x64)", flush=True)
    print(f"   CIFAR-10: {len(cifar_ds)} (low quality 32x32 -> 64x64)", flush=True)
    print(f"   Total: {len(combined_ds)} images", flush=True)

    # DataLoader: num_workers=0 避免Windows权限错误
    dl = DataLoader(combined_ds, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    dl_iter = cycle(dl)

    # === 创建模型 ===
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

    # === 训练配置 ===
    train_num_steps = 20000
    save_and_sample_every = 2000
    gradient_accumulate_every = 4
    max_grad_norm = 1.0
    amp = True
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    results_folder = Path('./mixed_cat_results')
    results_folder.mkdir(exist_ok=True)

    print(f"\nConfig:", flush=True)
    print(f"  Dataset: {len(combined_ds)} images (Oxford+CIFAR)", flush=True)
    print(f"  Model: dim=64, image=64x64", flush=True)
    print(f"  Batch: 4 x {gradient_accumulate_every} = 16 effective", flush=True)
    print(f"  Steps: {train_num_steps}", flush=True)
    print(f"  Save every: {save_and_sample_every}", flush=True)
    print(f"  AMP: {amp}, EMA: 0.995", flush=True)

    # === 检查点恢复 ===
    start_step = 0
    checkpoint_path = results_folder / 'model-latest.pt'
    if checkpoint_path.exists():
        print(f"\nResuming from {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.load_state_dict(ckpt['ema'])
        start_step = ckpt['step']
        print(f"Resumed from step {start_step}", flush=True)

    # === 开始训练 ===
    print(f"\nStarting training ({start_step} -> {train_num_steps})...\n", flush=True)
    model.train()

    for step in tqdm(range(start_step, train_num_steps), desc='Training', unit='step'):
        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(gradient_accumulate_every):
            data = next(dl_iter).to(device)
            with torch.amp.autocast('cuda', enabled=amp):
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

        # 保存模型和生成样本
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
                with torch.amp.autocast('cuda', enabled=amp):
                    sample = diffusion.sample(batch_size=1)
                utils.save_image((sample + 1) * 0.5, str(results_folder / f'sample-{step//save_and_sample_every}.png'))
            model.train()
            torch.cuda.empty_cache()

    # === 训练完成 ===
    print(f"\n{'='*60}", flush=True)
    print("Training complete! Generating final samples...", flush=True)
    print(f"{'='*60}", flush=True)

    ema.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=amp):
            final = diffusion.sample(batch_size=4)
        utils.save_image((final + 1) * 0.5, str(results_folder / 'final_samples.png'), nrow=2)
    torch.save(model.state_dict(), results_folder / 'model-final.pt')
    print(f"\nResults saved to: {results_folder}", flush=True)
    print(f"  - final_samples.png", flush=True)
    print(f"  - model-final.pt", flush=True)


if __name__ == '__main__':
    main()
