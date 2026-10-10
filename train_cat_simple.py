import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from ema_pytorch import EMA
from tqdm import tqdm
import os
import random

print("=" * 60)
print("🐱 用简单猫脸图案训练扩散模型")
print("=" * 60)

class SimpleCatDataset(Dataset):
    def __init__(self, num_images=1000, image_size=32):
        self.num_images = num_images
        self.image_size = image_size
        print(f"🎨 生成 {num_images} 张简单猫脸图片...")

    def draw_cat(self):
        img = torch.zeros(3, self.image_size, self.image_size)
        
        bg_color = torch.tensor([random.uniform(0.8, 1.0), random.uniform(0.8, 1.0), random.uniform(0.8, 1.0)])
        img[0] = bg_color[0]
        img[1] = bg_color[1]
        img[2] = bg_color[2]
        
        face_color = torch.tensor([random.uniform(0.7, 0.9), random.uniform(0.5, 0.7), random.uniform(0.3, 0.5)])
        
        cx, cy = self.image_size // 2, self.image_size // 2
        r = random.randint(10, 14)
        
        for i in range(self.image_size):
            for j in range(self.image_size):
                dist = ((i - cy) ** 2 + (j - cx) ** 2) ** 0.5
                if dist < r:
                    img[0, i, j] = face_color[0]
                    img[1, i, j] = face_color[1]
                    img[2, i, j] = face_color[2]
        
        ear_color = torch.tensor([random.uniform(0.6, 0.8), random.uniform(0.4, 0.6), random.uniform(0.2, 0.4)])
        
        ears = [
            (cx - 12, cy - 10, 6),
            (cx + 12, cy - 10, 6)
        ]
        
        for ex, ey, er in ears:
            for i in range(self.image_size):
                for j in range(self.image_size):
                    dist = ((i - ey) ** 2 + (j - ex) ** 2) ** 0.5
                    if dist < er:
                        img[0, i, j] = ear_color[0]
                        img[1, i, j] = ear_color[1]
                        img[2, i, j] = ear_color[2]
        
        eye_color = torch.tensor([0.0, 0.0, 0.0])
        
        eyes = [
            (cx - 6, cy - 2, 3),
            (cx + 6, cy - 2, 3)
        ]
        
        for ex, ey, er in eyes:
            for i in range(self.image_size):
                for j in range(self.image_size):
                    dist = ((i - ey) ** 2 + (j - ex) ** 2) ** 0.5
                    if dist < er:
                        img[0, i, j] = eye_color[0]
                        img[1, i, j] = eye_color[1]
                        img[2, i, j] = eye_color[2]
        
        nose_color = torch.tensor([1.0, random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)])
        
        for i in range(self.image_size):
            for j in range(self.image_size):
                dist = ((i - (cy + 4)) ** 2 + (j - cx) ** 2) ** 0.5
                if dist < 2:
                    img[0, i, j] = nose_color[0]
                    img[1, i, j] = nose_color[1]
                    img[2, i, j] = nose_color[2]
        
        mouth_color = torch.tensor([0.0, 0.0, 0.0])
        
        for j in range(cx - 4, cx + 5):
            img[0, cy + 8, j] = mouth_color[0]
            img[1, cy + 8, j] = mouth_color[1]
            img[2, cy + 8, j] = mouth_color[2]
        
        return img

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.draw_cat()

print("\n📥 创建猫脸数据集...")
dataset = SimpleCatDataset(num_images=500)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print("\n🖼️ 保存一些示例猫脸...")
os.makedirs('./cat_faces', exist_ok=True)
sample_images = torch.stack([dataset[i] for i in range(9)])
save_image(sample_images, './cat_faces/sample_cats.png', nrow=3)
print("✅ 示例猫脸已保存到 ./cat_faces/sample_cats.png")

print("\n🚀 创建模型...")
model = Unet(
    dim=32,
    dim_mults=(1, 2, 4),
    flash_attn=False
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=100,
    sampling_timesteps=25
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
diffusion = diffusion.to(device)

ema = EMA(diffusion, beta=0.995, update_every=5)
ema.to(device)

optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

output_dir = './cat_results_simple'
os.makedirs(output_dir, exist_ok=True)

print("\n🏋️ 开始训练...")
num_epochs = 20

for epoch in range(num_epochs):
    print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="训练进度")
    
    for batch in pbar:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        loss = diffusion(batch)
        loss.backward()
        optimizer.step()
        
        ema.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    print(f"📊 平均损失: {avg_loss:.4f}")
    
    if (epoch + 1) % 5 == 0:
        print(f"\n📸 生成示例图片 (epoch {epoch+1})")
        ema.ema_model.eval()
        with torch.no_grad():
            generated = ema.ema_model.sample(batch_size=9)
        save_image(generated, f'{output_dir}/cat_epoch_{epoch+1}.png', nrow=3)
        print(f"💾 已保存到 {output_dir}/cat_epoch_{epoch+1}.png")
        ema.ema_model.train()

print("\n" + "=" * 60)
print("🎉 训练完成!")
print("=" * 60)

print("\n📸 生成最终猫图片...")
ema.ema_model.eval()
with torch.no_grad():
    final_images = ema.ema_model.sample(batch_size=16)
save_image(final_images, f'{output_dir}/final_cats.png', nrow=4)

print(f"\n✅ 最终猫图片已保存到: {output_dir}/final_cats.png")
print("\n📋 所有文件:")
print(f"1. ./cat_faces/sample_cats.png - 训练用的示例猫脸")
print(f"2. {output_dir}/cat_epoch_*.png - 各阶段生成的猫图片")
print(f"3. {output_dir}/final_cats.png - 最终生成的猫图片")