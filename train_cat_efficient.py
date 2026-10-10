import torch
from torchvision.utils import save_image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os
import glob
from PIL import Image

class CatDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = glob.glob(os.path.join(folder_path, '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def main():
    print("=" * 60)
    print("🐱 高效版训练：使用已提取的图片")
    print("=" * 60)

    cat_dir = './real_cat_images_64x64'
    output_dir = './real_cat_results_efficient'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📥 加载猫图片...")
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(64, scale=(0.8, 1.0)),
        T.ToTensor()
    ])

    dataset = CatDataset(cat_dir, transform=transform)
    print(f"✅ 数据集加载完成! 共 {len(dataset)} 张图片")

    print("\n🚀 创建扩散模型...")
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=50,
        objective='pred_v'
    ).cuda()

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=8e-5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print("\n🏋️ 开始训练...")
    print(f"   - batch_size: 4")
    print(f"   - 训练步数: 10000")

    num_steps = 10000
    step = 0
    best_loss = float('inf')

    for epoch in range(100):
        for imgs in dataloader:
            if step >= num_steps:
                break

            imgs = imgs.cuda()

            with torch.cuda.amp.autocast():
                loss = diffusion(imgs)

            loss.backward()

            if (step + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f"step: {step}, loss: {loss.item():.4f}")

            if step % 1000 == 0 and step > 0:
                diffusion.eval()
                with torch.no_grad():
                    torch.manual_seed(42)
                    generated = diffusion.sample(batch_size=1)
                    save_image(generated, f'{output_dir}/sample_{step}.png', normalize=True)
                diffusion.train()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save({'model': diffusion.state_dict()}, f'{output_dir}/best_model.pt')

            step += 1

        if step >= num_steps:
            break

    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print(f"📁 生成的猫图片保存在 {output_dir}/ 文件夹中")

if __name__ == '__main__':
    main()