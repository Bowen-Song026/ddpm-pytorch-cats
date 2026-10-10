import torch
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os

def main():
    print("=" * 60)
    print("🐱 使用最佳模型重新生成猫图片")
    print("=" * 60)

    output_dir = './best_cat_images'
    os.makedirs(output_dir, exist_ok=True)

    print("\n🚀 加载最佳模型...")
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=250,
    )

    model_path = './real_cat_results_64x64/model-9.pt'
    print(f"\n📥 加载模型权重: {model_path}")
    checkpoint = torch.load(model_path)
    diffusion.load_state_dict(checkpoint['model'])
    diffusion = diffusion.cuda()
    diffusion.eval()

    print("\n🎨 开始生成猫图片...")
    num_images = 8
    images_per_batch = 1
    all_images = []

    for i in range(num_images):
        print(f"   生成第 {i+1}/{num_images} 张图片...")
        with torch.no_grad():
            torch.manual_seed(i)
            image = diffusion.sample(batch_size=images_per_batch)
            all_images.append(image)

        torch.cuda.empty_cache()

    all_images = torch.cat(all_images, dim=0)

    print(f"\n✅ 生成完成! 共 {num_images} 张图片")

    grid_image = save_image(all_images, f'{output_dir}/best_cats_grid.png', nrow=4, normalize=True)
    print(f"💾 网格图片已保存: {output_dir}/best_cats_grid.png")

    for i, img in enumerate(all_images):
        save_image(img, f'{output_dir}/cat_{i+1}.png', normalize=True)
    print(f"💾 单张图片已保存: {output_dir}/cat_1.png ~ cat_{num_images}.png")

    print("\n" + "=" * 60)
    print("🎉 生成完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()