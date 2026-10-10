import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torchvision import transforms as T
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import sys

def main():
    print("=" * 60)
    print("🐱 改进版训练：更多步数 + 数据增强")
    print("=" * 60)

    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)

    cat_output_dir = './real_cat_images_64x64'
    os.makedirs(cat_output_dir, exist_ok=True)

    print("\n📥 加载CIFAR-10数据集...")
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(64, scale=(0.8, 1.0)),
        T.ToTensor()
    ])

    dataset = CIFAR10(root=data_dir, download=False, train=True, transform=transform)
    print(f"✅ 数据集加载完成! 共 {len(dataset)} 张图片")

    print("\n🐱 提取猫的图片 (64×64)...")
    cat_count = 0
    for i, (img, label) in enumerate(dataset):
        if label == 3:
            save_image(img, f'{cat_output_dir}/cat_{cat_count}.png')
            cat_count += 1
            if cat_count % 500 == 0:
                print(f"   已提取 {cat_count} 张猫图片...")

    print(f"\n✅ 共提取 {cat_count} 张猫图片!")

    print("\n🚀 创建更大的扩散模型...")
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=50,
        objective='pred_v'
    )

    print("\n🏋️ 开始训练（改进版）...")
    print("   - 训练步数: 15000步")
    print("   - 数据增强: 随机翻转 + 随机裁剪")
    print("   - batch_size: 4")
    print("   - gradient_accumulate_every: 4 (有效batch_size=16)")

    trainer = Trainer(
        diffusion,
        cat_output_dir,
        train_batch_size=4,
        train_lr=8e-5,
        train_num_steps=15000,
        gradient_accumulate_every=4,
        ema_decay=0.995,
        amp=True,
        calculate_fid=False,
        save_and_sample_every=1500,
        num_samples=1,
        results_folder='./real_cat_results_improved'
    )

    trainer.train()

    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print("� 生成的猫图片保存在 real_cat_results_improved/ 文件夹中")

if __name__ == '__main__':
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.freeze_support()
    main()