import subprocess
import os

doc1 = r'd:\denoising-diffusion-pytorch-main\denoising-diffusion-pytorch-main\扩散模型核心代码详解.docx'
doc2 = r'd:\denoising-diffusion-pytorch-main\denoising-diffusion-pytorch-main\扩散模型核心代码.docx'

print(f'检查文件1: {doc1}')
print(f'存在: {os.path.exists(doc1)}')
print(f'大小: {os.path.getsize(doc1) if os.path.exists(doc1) else 0} bytes')

print(f'\n检查文件2: {doc2}')
print(f'存在: {os.path.exists(doc2)}')
print(f'大小: {os.path.getsize(doc2) if os.path.exists(doc2) else 0} bytes')

if os.path.exists(doc1):
    subprocess.Popen(['start', '', doc1], shell=True)
    print(f'\n已尝试打开: {doc1}')

if os.path.exists(doc2):
    subprocess.Popen(['start', '', doc2], shell=True)
    print(f'已尝试打开: {doc2}')

print('\n如果Word窗口仍未显示，请手动双击上述文件路径')