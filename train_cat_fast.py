import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from ema_pytorch import EMA
from tqdm import tqdm
import os

print("=" * 60)
print("🐱 快速训练猫图片生成器")
print("=" * 60)

class CatDataset(Dataset):
    def __init__(self):
        self.cifar = CIFAR10(root='./data', download=True, train=True)
        self.transform = T.Compose([
            T.ToTensor()
        ])
        self.cat_indices = []
        for i, (_, label) in enumerate(self.cifar):
            if label == 3:
                self.cat_indices.append(i)
        print(f"🐱 找到 {len(self.cat_indices)} 张猫的图片")

    def __len__(self):
        return len(self.cat_indices)

    def __getitem__(self, idx):
        img, _ = self.cifar[self.cat_indices[idx]]
        return self.transform(img)

print("\n📥 加载猫图片数据集...")
dataset = CatDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

print("\n🚀 创建模型...")
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    flash_attn=False
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=500,
    sampling_timesteps=50
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
diffusion = diffusion.to(device)

ema = EMA(diffusion, beta=0.995, update_every=10)
ema.to(device)

optimizer = torch.optim.Adam(diffusion.parameters(), lr=8e-5)

output_dir = './cat_results'
os.makedirs(output_dir, exist_ok=True)

print("\n🏋️ 开始训练...")
num_epochs = 5
step = 0

for epoch in range(num_epochs):
    print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
    pbar = tqdm(dataloader, desc=f"训练进度")
    
    for batch in pbar:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        loss = diffusion(batch)
        loss.backward()
        optimizer.step()
        
        ema.update()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        step += 1
        
        if step % 500 == 0:
            print(f"\n📸 生成示例图片 (step {step})")
            ema.ema_model.eval()
            with torch.no_grad():
                generated = ema.ema_model.sample(batch_size=4)
            save_image(generated, f'{output_dir}/cat_sample_step_{step}.png', nrow=2)
            print(f"💾 已保存到 {output_dir}/cat_sample_step_{step}.png")
            ema.ema_model.train()

print("\n" + "=" * 60)
print("🎉 训练完成!")
print("=" * 60)

print("\n📸 生成最终猫图片...")
ema.ema_model.eval()
with torch.no_grad():
    final_images = ema.ema_model.sample(batch_size=9)
save_image(final_images, f'{output_dir}/final_cat_images.png', nrow=3)

print(f"\n✅ 最终猫图片已保存到: {output_dir}/final_cat_images.png")