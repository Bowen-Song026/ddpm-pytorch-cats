import requests
import os
import sys

print("=" * 60)
print("📥 使用requests下载CIFAR-10数据集")
print("=" * 60)

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

filename = 'cifar-10-python.tar.gz'
file_path = os.path.join(data_dir, filename)

urls = [
    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    'https://mirrors.tuna.tsinghua.edu.cn/torchvision/datasets/cifar-10-python.tar.gz',
    'https://mirrors.aliyun.com/torchvision/datasets/cifar-10-python.tar.gz',
]

for url in urls:
    print(f"\n尝试从: {url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = (downloaded_size / total_size) * 100
                    sys.stdout.write(f"\r下载进度: {progress:.1f}% ({downloaded_size / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")
                    sys.stdout.flush()
        
        print(f"\n✅ 下载成功! 文件保存在: {file_path}")
        break
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        continue
else:
    print("\n❌ 所有镜像都下载失败了!")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 下载完成!")
print("=" * 60)