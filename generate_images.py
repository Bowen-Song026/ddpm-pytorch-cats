import torch
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os

print("=" * 60)
print("🎨 正在生成扩散模型图片...")
print("=" * 60)

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False
)

diffusion = GaussianDiffusion(
    model,
    image_size=64,
    timesteps=100,
    sampling_timesteps=25
)

if torch.cuda.is_available():
    model = model.cuda()
    diffusion = diffusion.cuda()
    print("✅ 使用GPU加速生成")
else:
    print("⚠️ 使用CPU生成，可能会比较慢")

output_dir = './generated_images'
os.makedirs(output_dir, exist_ok=True)

print("\n📸 开始采样生成图片...")
with torch.no_grad():
    sampled_images = diffusion.sample(batch_size=4)

print(f"✅ 图片生成完成! 形状: {sampled_images.shape}")

save_image(sampled_images, f'{output_dir}/generated_sample.png', nrow=2)
print(f"💾 图片已保存到: {output_dir}/generated_sample.png")

print("\n" + "=" * 60)
print("🎉 图片生成成功!")
print("=" * 60)

print("\n📋 运行结果说明:")
print("1. 生成的图片保存在 generated_images/ 文件夹中")
print("2. 图片名称: generated_sample.png")
print("3. 由于模型没有经过训练，生成的图片是随机噪声")
print("4. 要生成有意义的图片，需要使用真实数据集训练模型")