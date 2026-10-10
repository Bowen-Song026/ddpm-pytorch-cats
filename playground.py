import torch
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os

print("=" * 60)
print("🎨 扩散模型游乐场")
print("=" * 60)

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False
)

diffusion = GaussianDiffusion(
    model,
    image_size=64,
    timesteps=50,
    sampling_timesteps=25
)

if torch.cuda.is_available():
    model = model.cuda()
    diffusion = diffusion.cuda()
    print("✅ 使用GPU加速")

output_dir = './playground_output'
os.makedirs(output_dir, exist_ok=True)

print("\n🎯 玩法1：生成9张随机图片")
with torch.no_grad():
    images = diffusion.sample(batch_size=9)
save_image(images, f'{output_dir}/9_images.png', nrow=3)
print(f"💾 已保存: {output_dir}/9_images.png")

print("\n🎯 玩法2：生成128x128更大的图片")
large_model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False
)
large_diffusion = GaussianDiffusion(
    large_model,
    image_size=128,
    timesteps=50,
    sampling_timesteps=25
)
if torch.cuda.is_available():
    large_model = large_model.cuda()
    large_diffusion = large_diffusion.cuda()

with torch.no_grad():
    large_images = large_diffusion.sample(batch_size=4)
save_image(large_images, f'{output_dir}/large_images.png', nrow=2)
print(f"💾 已保存: {output_dir}/large_images.png")

print("\n🎯 玩法3：查看采样过程（生成GIF）")
with torch.no_grad():
    all_steps = diffusion.sample(batch_size=1, return_all_timesteps=True)

for i in range(0, all_steps.shape[1], 5):
    step_img = all_steps[0, i]
    save_image(step_img, f'{output_dir}/step_{i}.png')
print(f"💾 已保存采样过程图片: {output_dir}/step_*.png")

print("\n" + "=" * 60)
print("🎉 所有效果生成完毕!")
print("=" * 60)

print("\n📋 输出文件列表:")
print(f"1. {output_dir}/9_images.png      - 9张小图片")
print(f"2. {output_dir}/large_images.png  - 4张128x128大图片")
print(f"3. {output_dir}/step_*.png        - 采样过程的每一步")