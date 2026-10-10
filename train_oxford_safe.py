import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T, utils
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# Windows下设置多进程策略避免权限错误
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class OxfordCatDataset(Dataset):
    def __init__(self, folder, image_size=64):
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(folder).glob('*.png')] + [p for p in Path(folder).glob('*.jpg')]
        print(f"   找到 {len(self.paths)} 张猫图片")

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
    print("=" * 60)
    print("Oxford-IIIT Pet 猫图片扩散模型训练 (稳定版)")
    print("=" * 60)

    # 1. 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        print(f"PyTorch CUDA版本: {torch.version.cuda}")
    else:
        print("WARNING: 没有检测到GPU，将使用CPU训练（会非常慢）")
        return

    # 2. 数据集路径
    cat_dir = './oxford_cat_images'
    if not os.path.exists(cat_dir):
        print(f"ERROR: 找不到猫图片目录 {cat_dir}")
        print("请先运行 train_oxford_cats.py 下载数据集")
        return

    # 3. 创建数据集 - 使用num_workers=0避免Windows权限错误
    print(f"\n加载数据集: {cat_dir}")
    ds = OxfordCatDataset(cat_dir, image_size=64)
    dl = DataLoader(ds, batch_size=4, shuffle=True, pin_memory=True, num_workers=0)
    dl_iter = cycle(dl)

    # 4. 创建模型
    print("\n创建模型...")
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

    # 5. 优化器和EMA
    optimizer = Adam(model.parameters(), lr=8e-5, betas=(0.9, 0.99))
    ema = EMA(model, beta=0.995, update_every=10).to(device)

    # 6. 训练配置
    train_num_steps = 20000
    save_and_sample_every = 2000
    gradient_accumulate_every = 4
    max_grad_norm = 1.0
    amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    results_folder = Path('./oxford_cat_results_safe')
    results_folder.mkdir(exist_ok=True)

    print(f"\n训练配置:")
    print(f"  - 模型尺寸: dim=64")
    print(f"  - 图片尺寸: 64x64")
    print(f"  - batch_size: 4, 梯度累积: {gradient_accumulate_every}")
    print(f"  - 训练步数: {train_num_steps}")
    print(f"  - 保存间隔: 每{save_and_sample_every}步")
    print(f"  - 混合精度训练(AMP): {amp}")
    print(f"  - EMA: 0.995")
    print(f"  - 数据加载: num_workers=0 (避免Windows权限错误)")

    # 7. 从之前的模型继续训练（如果存在）
    start_step = 0
    checkpoint_path = results_folder / 'model-latest.pt'
    if checkpoint_path.exists():
        print(f"\n发现检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ema.load_state_dict(checkpoint['ema'])
        start_step = checkpoint['step']
        print(f"从第 {start_step} 步继续训练")

    # 8. 开始训练
    print(f"\n{'=' * 60}")
    print(f"开始训练... (从第 {start_step} 步到第 {train_num_steps} 步)")
    print(f"{'=' * 60}\n")

    model.train()

    for step in tqdm(range(start_step, train_num_steps), desc='训练进度', unit='步'):
        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(gradient_accumulate_every):
            data = next(dl_iter).to(device)

            with torch.cuda.amp.autocast(enabled=amp):
                loss = diffusion(data)
                loss = loss / gradient_accumulate_every

            scaler.scale(loss).backward()
            total_loss += loss.item()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        # 打印损失
        if step % 100 == 0:
            print(f"  step {step}/{train_num_steps} | loss: {total_loss:.4f}")

        # 保存模型和生成样本
        if step > 0 and step % save_and_sample_every == 0:
            print(f"\n  [保存] 第 {step} 步 - 生成样本图片...")

            # 保存检查点
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict(),
            }, results_folder / 'model-latest.pt')
            torch.save(model.state_dict(), results_folder / f'model-{step//save_and_sample_every}.pt')

            # 用EMA模型生成样本
            ema.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=amp):
                    sample = diffusion.sample(batch_size=1)

                # 保存样本
                sample = (sample + 1) * 0.5  # 反归一化到[0,1]
                utils.save_image(sample, str(results_folder / f'sample-{step//save_and_sample_every}.png'))

            model.train()
            torch.cuda.empty_cache()

    # 9. 训练完成，生成最终样本
    print(f"\n{'=' * 60}")
    print("训练完成! 生成最终样本...")
    print(f"{'=' * 60}")

    ema.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            final_samples = diffusion.sample(batch_size=4)

        final_samples = (final_samples + 1) * 0.5
        utils.save_image(final_samples, str(results_folder / 'final_samples.png'), nrow=2)

    # 保存最终模型
    torch.save(model.state_dict(), results_folder / 'model-final.pt')

    print(f"\n结果保存在: {results_folder}")
    print(f"  - final_samples.png (最终生成的猫图片)")
    print(f"  - model-final.pt (最终模型权重)")

if __name__ == '__main__':
    main()
