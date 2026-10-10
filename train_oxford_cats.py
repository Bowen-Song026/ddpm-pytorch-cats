，import torch
from torchvision.utils import save_image
from torchvision import transforms as T
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import zipfile
import requests
from PIL import Image

def download_oxford_pets():
    print("=" * 60)
    print("🐱 下载 Oxford-IIIT Pet 数据集")
    print("=" * 60)
    
    data_dir = './oxford_pets_data'
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, 'images.tar.gz')
    
    if not os.path.exists(zip_path):
        print("\n📥 正在下载数据集...")
        url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        response = requests.get(url, stream=True)
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = (downloaded / total_size) * 100
                    print(f"\r   下载进度: {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end='')
        
        print("\n✅ 下载完成!")
    
    images_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(images_dir):
        print("\n📦 正在解压...")
        import tarfile
        with tarfile.open(zip_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("✅ 解压完成!")
    
    print(f"\n📂 数据集目录: {images_dir}")
    return images_dir

def extract_cats(images_dir):
    print("\n🐱 提取猫的图片...")
    
    cat_dir = './oxford_cat_images'
    os.makedirs(cat_dir, exist_ok=True)
    
    cat_breeds = {
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx', 'American_Bobtail', 'American_Curl',
        'American_Shorthair', 'Devon_Rex', 'Himalayan', 'Japanese_Bobtail',
        'Scottish_Fold', 'Turkish_Angora'
    }
    
    cat_count = 0
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            breed = filename.split('_')[0]
            if breed in cat_breeds:
                img_path = os.path.join(images_dir, filename)
                img = Image.open(img_path).convert('RGB')
                img.save(f'{cat_dir}/cat_{cat_count}.png')
                cat_count += 1
    
    print(f"\n✅ 共提取 {cat_count} 张猫图片!")
    return cat_dir

def train_diffusion(cat_dir):
    print("\n" + "=" * 60)
    print("🚀 开始训练扩散模型")
    print("=" * 60)
    
    print("\n📐 创建模型...")
    print("   - 模型尺寸: dim=64（适合4GB显存）")
    print("   - 图片尺寸: 64×64")
    
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
    
    print("\n🏋️ 配置训练器...")
    print("   - 使用 Oxford-IIIT Pet 数据集（全部~3800张猫图片）")
    print("   - 图片原生分辨率600×400，下采样到64×64")
    print("   - batch_size: 4")
    print("   - gradient_accumulate_every: 4（有效batch_size=16）")
    print("   - 训练步数: 20000")
    print("   - EMA: 0.995")
    
    trainer = Trainer(
        diffusion,
        cat_dir,
        train_batch_size=4,
        train_lr=8e-5,
        train_num_steps=20000,
        gradient_accumulate_every=4,
        ema_decay=0.995,
        amp=True,
        calculate_fid=False,
        save_and_sample_every=2000,
        num_samples=1,
        results_folder='./oxford_cat_results',
        augment_horizontal_flip=True
    )
    
    print("\n🔥 开始训练...")
    trainer.train()
    
    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print("📁 生成的猫图片保存在 oxford_cat_results/ 文件夹中")

def main():
    images_dir = download_oxford_pets()
    cat_dir = extract_cats(images_dir)
    train_diffusion(cat_dir)

if __name__ == '__main__':
    main()