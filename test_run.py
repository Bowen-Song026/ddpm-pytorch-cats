import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

print("=" * 60)
print("🚀 正在测试扩散模型项目...")
print("=" * 60)

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True
)
print("✅ 模型创建成功!")

diffusion = GaussianDiffusion(
    model,
    image_size=64,
    timesteps=100
)
print("✅ 扩散模型初始化成功!")

training_images = torch.rand(2, 3, 64, 64)
print(f"📷 生成测试图像: {training_images.shape}")

loss = diffusion(training_images)
print(f"📊 计算损失值: {loss.item():.4f}")

loss.backward()
print("✅ 反向传播成功!")

print("\n" + "=" * 60)
print("🎉 项目运行成功! 环境配置完全正确!")
print("=" * 60)