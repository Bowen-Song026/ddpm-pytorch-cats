import os

doc1 = r'd:\denoising-diffusion-pytorch-main\denoising-diffusion-pytorch-main\扩散模型核心代码详解.docx'
doc2 = r'd:\denoising-diffusion-pytorch-main\denoising-diffusion-pytorch-main\扩散模型核心代码.docx'

if os.path.exists(doc1):
    os.startfile(doc1)
    print(f'已打开: {doc1}')
else:
    print(f'文件不存在: {doc1}')

if os.path.exists(doc2):
    os.startfile(doc2)
    print(f'已打开: {doc2}')
else:
    print(f'文件不存在: {doc2}')