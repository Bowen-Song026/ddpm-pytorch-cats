import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torchvision import transforms as T
from torchvision.datasets.utils import download_url
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import tarfile
import os

print("=" * 60)
print("🐱 使用真实猫照片训练扩散模型（国内镜像加速）")
print("=" * 60)

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

cat_output_dir = './real_cat_images'
os.makedirs(cat_output_dir, exist_ok=True)

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
local_url = 'https://mirrors.tuna.tsinghua.edu.cn/torchvision/datasets/cifar-10-python.tar.gz'
filename = 'cifar-10-python.tar.gz'
tgz_md5 = 'c58f30108f718f92721af3b95e74349a'

file_path = os.path.join(data_dir, filename)

print("\n📥 使用清华镜像下载CIFAR-10数据集...")
try:
    download_url(local_url, data_dir, filename=filename, md5=tgz_md5)
    print("✅ 使用清华镜像下载成功!")
except Exception as e:
    print(f"⚠️ 清华镜像下载失败: {e}")
    print("📥 尝试使用官方源下载...")
    download_url(url, data_dir, filename=filename, md5=tgz_md5)
    print("✅ 官方源下载成功!")

print("\n📦 解压数据集...")
with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall(path=data_dir)
print("✅ 数据集解压完成!")

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

print("\n🖼️ 查看一些猫图片...")
sample_images = []
for i in range(min(9, cat_count)):
    img_path = f'{cat_output_dir}/cat_{i}.png'
    from PIL import Image
    img = Image.open(img_path)
    sample_images.append(T.ToTensor()(img))
save_image(torch.stack(sample_images), './real_cat_images/sample_cats.png', nrow=3)
print("✅ 示例猫图片已保存到 ./real_cat_images/sample_cats.png")

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