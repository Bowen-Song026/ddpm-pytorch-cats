import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torchvision import transforms as T
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import sys

def main():
    print("=" * 60)
    print("🐱 使用真实猫照片训练扩散模型")
    print("=" * 60)

    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)

    cat_output_dir = './real_cat_images'
    os.makedirs(cat_output_dir, exist_ok=True)

    print("\n📥 加载CIFAR-10数据集...")
    transform = T.Compose([
        T.ToTensor()
    ])

    dataset = CIFAR10(root=data_dir, download=False, train=True, transform=transform)
    print(f"✅ 数据集加载完成! 共 {len(dataset)} 张图片")

    print("\n🐱 提取猫的图片...")
    cat_count = 0
    for i, (img, label) in enumerate(dataset):
        if label == 3:
            save_image(img, f'{cat_output_dir}/cat_{cat_count}.png')
            cat_count += 1
            if cat_count % 200 == 0:
                print(f"   已提取 {cat_count} 张猫图片...")

    print(f"\n✅ 共提取 {cat_count} 张猫图片!")
    print(f"💾 猫图片保存在: {cat_output_dir}/")

    print("\n🚀 创建扩散模型...")
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,
        sampling_timesteps=250
    )

    print("\n🏋️ 开始训练...")
    trainer = Trainer(
        diffusion,
        cat_output_dir,
        train_batch_size=8,
        train_lr=8e-5,
        train_num_steps=10000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        calculate_fid=False,
        save_and_sample_every=1000,
        num_samples=4,
        results_folder='./real_cat_results'
    )

    trainer.train()

    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print("📁 生成的猫图片保存在 real_cat_results/ 文件夹中")

if __name__ == '__main__':
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.freeze_support()
    main()