import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os

print("=" * 60)
print("🐱 训练猫图片生成器")
print("=" * 60)

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

print("\n📥 下载CIFAR-10数据集...")
dataset = CIFAR10(root=data_dir, download=True, train=True)
print(f"✅ 数据集下载完成! 共 {len(dataset)} 张图片")

print("\n🐱 提取猫的图片...")
cat_indices = []
for i, (img, label) in enumerate(dataset):
    if label == 3:
        cat_indices.append(i)

cat_images = []
for idx in cat_indices:
    img, _ = dataset[idx]
    cat_images.append(img)

print(f"✅ 提取完成! 共 {len(cat_images)} 张猫的图片")

cat_output_dir = './cat_data'
os.makedirs(cat_output_dir, exist_ok=True)

print("\n💾 保存猫的图片...")
for i, img in enumerate(cat_images[:100]):
    img.save(f'{cat_output_dir}/cat_{i}.png')
print(f"✅ 已保存 {min(100, len(cat_images))} 张猫图片到 {cat_output_dir}/")

print("\n🚀 创建扩散模型...")
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    flash_attn=False
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=1000,
    sampling_timesteps=250
)

if torch.cuda.is_available():
    model = model.cuda()
    diffusion = diffusion.cuda()
    print("✅ 使用GPU训练")

print("\n🏋️ 开始训练...")
trainer = Trainer(
    diffusion,
    cat_output_dir,
    train_batch_size=8,
    train_lr=8e-5,
    train_num_steps=5000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=True,
    calculate_fid=False,
    save_and_sample_every=1000,
    num_samples=4
)

trainer.train()

print("\n" + "=" * 60)
print("🎉 训练完成!")
print("=" * 60)
print("📁 生成的猫图片保存在 results/ 文件夹中")