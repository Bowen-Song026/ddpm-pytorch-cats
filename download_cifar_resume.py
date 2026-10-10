import requests
import os
import sys

print("=" * 60)
print("📥 断点续传下载CIFAR-10数据集")
print("=" * 60)

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

filename = 'cifar-10-python.tar.gz'
file_path = os.path.join(data_dir, filename)

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

if os.path.exists(file_path):
    existing_size = os.path.getsize(file_path)
    print(f"📁 已有文件大小: {existing_size / 1024 / 1024:.1f}MB")
else:
    existing_size = 0

try:
    headers = {'Range': f'bytes={existing_size}-'}
    response = requests.get(url, stream=True, headers=headers, timeout=120)
    response.raise_for_status()
    
    content_range = response.headers.get('content-range')
    if content_range:
        total_size = int(content_range.split('/')[1])
    else:
        total_size = existing_size + int(response.headers.get('content-length', 0))
    
    downloaded_size = existing_size
    
    mode = 'ab' if existing_size > 0 else 'wb'
    with open(file_path, mode) as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                progress = (downloaded_size / total_size) * 100
                sys.stdout.write(f"\r下载进度: {progress:.1f}% ({downloaded_size / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")
                sys.stdout.flush()
    
    print(f"\n✅ 下载成功! 文件保存在: {file_path}")
    print(f"📦 文件大小: {downloaded_size / 1024 / 1024:.1f}MB")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 下载完成!")
print("=" * 60)