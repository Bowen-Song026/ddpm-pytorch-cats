import shutil
import os

source = r'd:\denoising-diffusion-pytorch-main\denoising-diffusion-pytorch-main\扩散模型核心代码详解.docx'
destination = os.path.join(os.path.expanduser('~'), 'Desktop', '扩散模型核心代码详解.docx')

if os.path.exists(source):
    shutil.copy(source, destination)
    print(f'已复制到桌面: {destination}')
else:
    print(f'源文件不存在: {source}')