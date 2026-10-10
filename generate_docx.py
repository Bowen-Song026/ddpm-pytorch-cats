from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn

doc = Document()

style = doc.styles['Normal']
style.font.name = '微软雅黑'
style.font.size = Pt(11)
style.paragraph_format.line_spacing = 1.5

sections = doc.sections
for section in sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)

title = doc.add_heading('扩散模型（DDPM）核心代码详解', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('基于 PyTorch 的完整实现')
run.font.size = Pt(14)

doc.add_paragraph()
intro = doc.add_paragraph()
intro.add_run('本文档对扩散模型的核心代码进行逐行详解，帮助初学者理解扩散模型的实现原理。代码包含两个核心类：')
doc.add_paragraph()
intro_list = doc.add_paragraph(style='List Bullet')
intro_list.add_run('UNet：用于预测噪声的神经网络架构')
intro_list = doc.add_paragraph(style='List Bullet')
intro_list.add_run('GaussianDiffusion：实现扩散过程的数学逻辑')
doc.add_paragraph()
intro.add_run('让我们从最基础的导入和辅助函数开始。')

doc.add_heading('第一部分：导入和辅助函数', level=1)

doc.add_heading('1.1 导入依赖', level=2)
doc.add_paragraph('首先，我们需要导入所有需要的库和模块：')

code_lines = [
    'import math',
    'import types',
    'from functools import partial',
    'from collections import namedtuple',
    '',
    'import torch',
    'from torch import nn, einsum',
    'import torch.nn.functional as F',
    'from torch.nn import Module, ModuleList',
    '',
    'from einops import rearrange, reduce, repeat',
    'from einops.layers.torch import Rearrange',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0, 102, 204)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('torch：PyTorch深度学习框架')
doc.add_paragraph(style='List Bullet').add_run('nn：神经网络模块')
doc.add_paragraph(style='List Bullet').add_run('einops：用于张量维度重排的工具库，让代码更清晰')

doc.add_heading('1.2 辅助函数', level=2)
doc.add_paragraph('这些小函数虽然简单，但在整个代码中被大量使用：')

code_section = [
    ('exists(x)', '检查变量是否为None'),
    ('default(val, d)', '如果val存在就返回val，否则返回d（d可以是函数）'),
    ('cast_tuple(t, length)', '把非元组变成元组，方便统一处理'),
    ('divisible_by(numer, denom)', '检查是否能整除'),
    ('identity(t)', '恒等函数，直接返回输入'),
    ('cycle(dl)', '创建一个无限循环的数据加载器'),
]

for func_name, desc in code_section:
    p = doc.add_paragraph()
    run = p.add_run(f'def {func_name}:')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.bold = True
    p.add_run(f'  {desc}')

doc.add_paragraph()
doc.add_paragraph('特别重要的两个函数：')

code_lines = [
    'def normalize_to_neg_one_to_one(img):',
    '    return img * 2 - 1',
    '',
    'def unnormalize_to_zero_to_one(t):',
    '    return (t + 1) * 0.5',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(102, 102, 102)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('normalize_to_neg_one_to_one：把图片像素从 [0, 1] 范围转换到 [-1, 1] 范围，这是扩散模型常用的标准化方式')
doc.add_paragraph(style='List Bullet').add_run('unnormalize_to_zero_to_one：反向操作，用于生成图片后恢复到正常范围')

doc.add_heading('第二部分：基础组件', level=1)

doc.add_heading('2.1 RMSNorm 归一化层', level=2)
doc.add_paragraph('RMSNorm 是一种简化版的归一化，比 BatchNorm 更适合扩散模型：')

code_lines = [
    'class RMSNorm(Module):',
    '    def __init__(self, dim):',
    '        super().__init__()',
    '        self.scale = dim ** 0.5',
    '        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))',
    '',
    '    def forward(self, x):',
    '        return F.normalize(x, dim=1) * self.g * self.scale',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0, 102, 0)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('dim：输入通道数')
doc.add_paragraph(style='List Bullet').add_run('self.g：可学习的缩放参数，初始值为1')
doc.add_paragraph(style='List Bullet').add_run('forward：先标准化，再乘以缩放参数')

doc.add_heading('2.2 SinusoidalPosEmb 时间步嵌入', level=2)
doc.add_paragraph('扩散模型需要知道当前是第几步，时间步嵌入就是把步数变成向量：')

code_lines = [
    'class SinusoidalPosEmb(Module):',
    '    def __init__(self, dim, theta=10000):',
    '        super().__init__()',
    '        self.dim = dim',
    '        self.theta = theta',
    '',
    '    def forward(self, x):',
    '        device = x.device',
    '        half_dim = self.dim // 2',
    '        emb = math.log(self.theta) / (half_dim - 1)',
    '        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)',
    '        emb = x[:, None] * emb[None, :]',
    '        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)',
    '        return emb',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(153, 0, 153)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('dim：嵌入向量的维度')
doc.add_paragraph(style='List Bullet').add_run('theta：频率基数，默认10000')
doc.add_paragraph(style='List Bullet').add_run('forward：把时间步x转换成正弦和余弦的组合，这样不同时间步的向量距离合理')

doc.add_heading('2.3 Block 基础卷积块', level=2)

code_lines = [
    'class Block(Module):',
    '    def __init__(self, dim, dim_out, dropout=0.):',
    '        super().__init__()',
    '        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)',
    '        self.norm = RMSNorm(dim_out)',
    '        self.act = nn.SiLU()',
    '        self.dropout = nn.Dropout(dropout)',
    '',
    '    def forward(self, x, scale_shift=None):',
    '        x = self.proj(x)      # 卷积变换',
    '        x = self.norm(x)      # 归一化',
    '        if exists(scale_shift):',
    '            scale, shift = scale_shift',
    '            x = x * (scale + 1) + shift  # 条件归一化',
    '        x = self.act(x)       # 激活函数',
    '        return self.dropout(x)',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(204, 102, 0)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('SiLU：一种平滑的激活函数，比ReLU更好')
doc.add_paragraph(style='List Bullet').add_run('scale_shift：来自时间步的条件信息，可以动态调整特征')

doc.add_heading('2.4 ResnetBlock 残差块', level=2)
doc.add_paragraph('残差块是深度学习的核心，让网络可以更深：')

code_lines = [
    'class ResnetBlock(Module):',
    '    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):',
    '        super().__init__()',
    '        self.mlp = nn.Sequential(',
    '            nn.SiLU(),',
    '            nn.Linear(time_emb_dim, dim_out * 2)',
    '        ) if exists(time_emb_dim) else None',
    '',
    '        self.block1 = Block(dim, dim_out, dropout=dropout)',
    '        self.block2 = Block(dim_out, dim_out)',
    '        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()',
    '',
    '    def forward(self, x, time_emb=None):',
    '        scale_shift = None',
    '        if exists(self.mlp) and exists(time_emb):',
    '            time_emb = self.mlp(time_emb)            # 时间步变换',
    '            time_emb = rearrange(time_emb, \'b c -> b c 1 1\')  # 扩展维度',
    '            scale_shift = time_emb.chunk(2, dim=1)   # 分成scale和shift',
    '',
    '        h = self.block1(x, scale_shift=scale_shift)',
    '        h = self.block2(h)',
    '        return h + self.res_conv(x)  # 残差连接',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0, 102, 204)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('mlp：把时间步嵌入转换成scale和shift')
doc.add_paragraph(style='List Bullet').add_run('res_conv：当输入输出维度不同时，用1x1卷积调整')
doc.add_paragraph(style='List Bullet').add_run('残差连接：h + res_conv(x)，这样梯度可以直接传递，避免梯度消失')

doc.add_heading('第三部分：注意力机制', level=1)

doc.add_heading('3.1 LinearAttention 线性注意力', level=2)
doc.add_paragraph('线性注意力是一种轻量级的注意力，计算更快：')

code_lines = [
    'class LinearAttention(Module):',
    '    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):',
    '        super().__init__()',
    '        self.heads = heads',
    '        hidden_dim = dim_head * heads',
    '',
    '        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)',
    '        self.to_out = nn.Conv2d(hidden_dim, dim, 1)',
    '',
    '    def forward(self, x):',
    '        qkv = self.to_qkv(x).chunk(3, dim=1)',
    '        q, k, v = map(lambda t: rearrange(t, \'b (h c) x y -> b h c (x y)\', h=self.heads), qkv)',
    '',
    '        q = q.softmax(dim=-2)',
    '        k = k.softmax(dim=-1)',
    '',
    '        context = torch.einsum(\'b h d n, b h e n -> b h d e\', k, v)',
    '        out = torch.einsum(\'b h d e, b h d n -> b h e n\', context, q)',
    '',
    '        out = rearrange(out, \'b h c (x y) -> b (h c) x y\', h=self.heads, x=h, y=w)',
    '        return self.to_out(out)',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(102, 102, 0)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('多头注意力：把特征分成多个头并行处理')
doc.add_paragraph(style='List Bullet').add_run('qkv：查询、键、值，是注意力的三个核心概念')
doc.add_paragraph(style='List Bullet').add_run('einsum：爱因斯坦求和，用字符串描述张量运算')

doc.add_heading('3.2 Attention 完整注意力', level=2)

code_lines = [
    'class Attention(Module):',
    '    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):',
    '        super().__init__()',
    '        self.flash = flash  # 是否使用Flash Attention加速',
    '',
    '    def forward(self, x):',
    '        qkv = self.to_qkv(x).chunk(3, dim=1)',
    '        q, k, v = map(lambda t: rearrange(t, \'b (h c) x y -> b h (x y) c\', h=self.heads), qkv)',
    '',
    '        if self.flash:',
    '            out = F.scaled_dot_product_attention(q, k, v)  # PyTorch内置加速',
    '        else:',
    '            scale = q.shape[-1] ** -0.5',
    '            sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale',
    '            attn = sim.softmax(dim=-1)',
    '            out = einsum(f"b h i j, b h j d -> b h i d", attn, v)',
    '',
    '        return self.to_out(out)',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(153, 0, 153)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('Flash Attention：PyTorch 2.0+的特性，大幅加速注意力计算')
doc.add_paragraph(style='List Bullet').add_run('sim：相似度矩阵，计算q和k的匹配程度')
doc.add_paragraph(style='List Bullet').add_run('softmax：把相似度变成权重')

doc.add_heading('第四部分：采样模块', level=1)

doc.add_heading('4.1 Upsample 上采样', level=2)

code_lines = [
    'def Upsample(dim, dim_out=None):',
    '    return nn.Sequential(',
    '        nn.Upsample(scale_factor=2, mode=\'nearest\'),',
    '        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)',
    '    )',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(204, 102, 0)

doc.add_paragraph()
doc.add_paragraph('解释：先把图像放大2倍，再用卷积调整特征')

doc.add_heading('4.2 Downsample 下采样', level=2)

code_lines = [
    'def Downsample(dim, dim_out=None):',
    '    return nn.Sequential(',
    '        Rearrange(\'b c (h p1) (w p2) -> b (c p1 p2) h w\', p1=2, p2=2),',
    '        nn.Conv2d(dim * 4, default(dim_out, dim), 1)',
    '    )',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0, 102, 0)

doc.add_paragraph()
doc.add_paragraph('解释：先把2x2的像素合并成一个特征，再用1x1卷积降维')

doc.add_heading('第五部分：UNet 核心架构', level=1)

doc.add_paragraph('UNet是扩散模型的核心网络，它的结构像一个U字：')
doc.add_paragraph()
doc.add_paragraph('┌─────────────────────────────────┐')
doc.add_paragraph('│         输入图像 (64x64)        │')
doc.add_paragraph('└─────────────────┬───────────────┘')
doc.add_paragraph('                  │')
doc.add_paragraph('┌─────────────────▼───────────────┐')
doc.add_paragraph('│  编码器（下采样）                │')
doc.add_paragraph('│  ResnetBlock → Attention → Down│')
doc.add_paragraph('│  ResnetBlock → Attention → Down│')
doc.add_paragraph('│  ResnetBlock → Attention → Down│')
doc.add_paragraph('└─────────────────┬───────────────┘')
doc.add_paragraph('                  │')
doc.add_paragraph('┌─────────────────▼───────────────┐')
doc.add_paragraph('│  中间层                          │')
doc.add_paragraph('│  ResnetBlock → Attention → Resnet│')
doc.add_paragraph('└─────────────────┬───────────────┘')
doc.add_paragraph('                  │')
doc.add_paragraph('┌─────────────────▼───────────────┐')
doc.add_paragraph('│  解码器（上采样）                │')
doc.add_paragraph('│  ResnetBlock → Attention → Up  │')
doc.add_paragraph('│  ResnetBlock → Attention → Up  │')
doc.add_paragraph('│  ResnetBlock → Attention → Up  │')
doc.add_paragraph('└─────────────────┬───────────────┘')
doc.add_paragraph('                  │')
doc.add_paragraph('┌─────────────────▼───────────────┐')
doc.add_paragraph('│         输出预测噪声              │')
doc.add_paragraph('└─────────────────────────────────┘')

doc.add_heading('5.1 UNet 初始化', level=2)

code_lines = [
    'class Unet(Module):',
    '    def __init__(',
    '        self,',
    '        dim,           # 基础通道数',
    '        dim_mults=(1, 2, 4, 8),  # 各层通道倍数',
    '        channels=3,    # 输入图像通道数（RGB=3）',
    '        flash_attn=False',
    '    ):',
    '',
    '        # 计算各层维度',
    '        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]',
    '        # dims = [64, 64, 128, 256, 512]',
    '',
    '        # 时间步嵌入MLP',
    '        self.time_mlp = nn.Sequential(',
    '            SinusoidalPosEmb(dim),',
    '            nn.Linear(dim, dim * 4),',
    '            nn.GELU(),',
    '            nn.Linear(dim * 4, dim * 4)',
    '        )',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0, 102, 204)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('dim=64：最底层有64个通道')
doc.add_paragraph(style='List Bullet').add_run('dim_mults=(1,2,4,8)：各层通道数依次是64, 128, 256, 512')
doc.add_paragraph(style='List Bullet').add_run('time_mlp：把时间步变成一个4倍dim的向量')

doc.add_heading('5.2 UNet 前向传播', level=2)

code_lines = [
    'def forward(self, x, time, x_self_cond=None):',
    '    x = self.init_conv(x)  # 初始卷积',
    '    r = x.clone()          # 保存初始特征',
    '',
    '    t = self.time_mlp(time)  # 时间步嵌入',
    '',
    '    h = []  # 保存跳连接特征',
    '',
    '    # 编码器：下采样',
    '    for block1, block2, attn, downsample in self.downs:',
    '        x = block1(x, t)',
    '        h.append(x)',
    '        x = block2(x, t)',
    '        x = attn(x) + x',
    '        h.append(x)',
    '        x = downsample(x)',
    '',
    '    # 中间层',
    '    x = self.mid_block1(x, t)',
    '    x = self.mid_attn(x) + x',
    '    x = self.mid_block2(x, t)',
    '',
    '    # 解码器：上采样',
    '    for block1, block2, attn, upsample in self.ups:',
    '        x = torch.cat((x, h.pop()), dim=1)  # 跳连接',
    '        x = block1(x, t)',
    '        x = torch.cat((x, h.pop()), dim=1)',
    '        x = block2(x, t)',
    '        x = attn(x) + x',
    '        x = upsample(x)',
    '',
    '    x = torch.cat((x, r), dim=1)',
    '    x = self.final_res_block(x, t)',
    '    return self.final_conv(x)',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(153, 0, 153)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('跳连接（Skip Connection）：编码器的特征直接传给解码器，保留细节信息')
doc.add_paragraph(style='List Bullet').add_run('h.pop()：后进先出，保证特征匹配')
doc.add_paragraph(style='List Bullet').add_run('torch.cat：在通道维度拼接特征')

doc.add_heading('第六部分：GaussianDiffusion 扩散过程', level=1)

doc.add_paragraph('扩散过程分为两个阶段：')
doc.add_paragraph()
doc.add_paragraph('📈 前向扩散（训练时）：')
doc.add_paragraph('  从清晰图像 x0 开始，一步步加噪声，直到变成纯噪声 xT')
doc.add_paragraph('  x_t = sqrt(alpha_cumprod[t]) * x0 + sqrt(1-alpha_cumprod[t]) * noise')
doc.add_paragraph()
doc.add_paragraph('📉 反向去噪（生成时）：')
doc.add_paragraph('  从纯噪声 xT 开始，一步步去噪，直到恢复清晰图像 x0')
doc.add_paragraph('  x_{t-1} = model(x_t, t) 预测噪声，然后减去噪声')

doc.add_heading('6.1 Beta 调度', level=2)

code_lines = [
    'def linear_beta_schedule(timesteps):',
    '    beta_start = 0.0001',
    '    beta_end = 0.02',
    '    return torch.linspace(beta_start, beta_end, timesteps)',
    '',
    'def cosine_beta_schedule(timesteps):',
    '    t = torch.linspace(0, timesteps, steps=timesteps+1) / timesteps',
    '    alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2',
    '    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])',
    '    return torch.clip(betas, 0, 0.999)',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(204, 102, 0)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('beta：每步加多少噪声')
doc.add_paragraph(style='List Bullet').add_run('linear：线性增长，早期少噪声，后期多噪声')
doc.add_paragraph(style='List Bullet').add_run('cosine：余弦调度，噪声分布更均匀')

doc.add_heading('6.2 GaussianDiffusion 初始化', level=2)

code_lines = [
    'class GaussianDiffusion(Module):',
    '    def __init__(',
    '        self,',
    '        model,           # UNet模型',
    '        image_size=64,   # 图像尺寸',
    '        timesteps=1000,  # 总步数',
    '        sampling_timesteps=50,  # 采样步数（DDIM加速）',
    '        objective=\'pred_v\',  # 预测目标',
    '        beta_schedule=\'sigmoid\'',
    '    ):',
    '',
    '        # 计算alpha相关参数',
    '        betas = beta_schedule_fn(timesteps)',
    '        alphas = 1. - betas',
    '        alphas_cumprod = torch.cumprod(alphas, dim=0)',
    '',
    '        # 注册缓冲区（不参与训练的参数）',
    '        register_buffer(\'betas\', betas)',
    '        register_buffer(\'alphas_cumprod\', alphas_cumprod)',
    '        register_buffer(\'sqrt_alphas_cumprod\', torch.sqrt(alphas_cumprod))',
    '        register_buffer(\'sqrt_one_minus_alphas_cumprod\', torch.sqrt(1. - alphas_cumprod))',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0, 102, 0)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('alphas_cumprod：累积乘积，代表保留多少原始信息')
doc.add_paragraph(style='List Bullet').add_run('sqrt_alphas_cumprod：前向扩散时乘以原始图像')
doc.add_paragraph(style='List Bullet').add_run('sqrt_one_minus_alphas_cumprod：前向扩散时乘以噪声')

doc.add_heading('6.3 三种预测目标', level=2)

code_lines = [
    'if objective == \'pred_noise\':',
    '    # 直接预测噪声',
    '    target = noise',
    '',
    'elif objective == \'pred_x0\':',
    '    # 预测原始图像',
    '    target = x_start',
    '',
    'elif objective == \'pred_v\':',
    '    # v参数化（推荐）',
    '    target = self.predict_v(x_start, t, noise)',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0, 102, 204)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('pred_noise：最简单，直接预测加进去的噪声')
doc.add_paragraph(style='List Bullet').add_run('pred_x0：直接预测原始图像，训练不稳定')
doc.add_paragraph(style='List Bullet').add_run('pred_v：v参数化，是pred_noise和pred_x0的组合，训练更稳定')

doc.add_heading('6.4 采样过程', level=2)

code_lines = [
    '@torch.inference_mode()',
    'def ddim_sample(self, shape, return_all_timesteps=False):',
    '    img = torch.randn(shape, device=device)  # 从纯噪声开始',
    '',
    '    for time, time_next in tqdm(time_pairs):',
    '        # 1. 预测噪声',
    '        pred_noise, x_start, *_ = self.model_predictions(img, time_cond)',
    '',
    '        # 2. 计算下一个状态',
    '        alpha = self.alphas_cumprod[time]',
    '        alpha_next = self.alphas_cumprod[time_next]',
    '',
    '        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()',
    '        c = (1 - alpha_next - sigma ** 2).sqrt()',
    '',
    '        img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise',
    '',
    '    return self.unnormalize(img)  # 转换回[0,1]范围',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(153, 0, 153)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('DDIM采样：可以跳过很多步，加速生成')
doc.add_paragraph(style='List Bullet').add_run('sampling_timesteps=50：只用50步就完成1000步的去噪')
doc.add_paragraph(style='List Bullet').add_run('eta=0：确定性采样，每次生成相同图片')

doc.add_heading('6.5 损失计算', level=2)

code_lines = [
    'def p_losses(self, x_start, t, noise=None):',
    '    # 1. 生成噪声',
    '    noise = default(noise, lambda: torch.randn_like(x_start))',
    '',
    '    # 2. 前向扩散：加噪声',
    '    x = self.q_sample(x_start=x_start, t=t, noise=noise)',
    '',
    '    # 3. 模型预测',
    '    model_out = self.model(x, t, x_self_cond)',
    '',
    '    # 4. 计算损失',
    '    loss = F.mse_loss(model_out, target, reduction=\'none\')',
    '    loss = loss.mean(dim=(1, 2, 3))',
    '    loss = loss * extract(self.loss_weight, t, loss.shape)',
    '',
    '    return loss.mean()',
]

for line in code_lines:
    p = doc.add_paragraph()
    run = p.add_run(f'{line}')
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(204, 102, 0)

doc.add_paragraph()
doc.add_paragraph('解释：')
doc.add_paragraph(style='List Bullet').add_run('q_sample：前向扩散函数')
doc.add_paragraph(style='List Bullet').add_run('loss_weight：不同时间步的权重，改善训练稳定性')

doc.add_heading('总结', level=1)

doc.add_paragraph('扩散模型的核心流程：')
doc.add_paragraph()
doc.add_paragraph('🔄 训练阶段：')
doc.add_paragraph('  1. 随机选一个时间步 t')
doc.add_paragraph('  2. 给清晰图像加噪声得到 x_t')
doc.add_paragraph('  3. 用UNet预测噪声')
doc.add_paragraph('  4. 计算MSE损失并反向传播')
doc.add_paragraph()
doc.add_paragraph('🎨 生成阶段：')
doc.add_paragraph('  1. 从纯噪声 x_T 开始')
doc.add_paragraph('  2. 用UNet预测噪声')
doc.add_paragraph('  3. 减去噪声得到 x_{t-1}')
doc.add_paragraph('  4. 重复直到得到清晰图像')
doc.add_paragraph()
doc.add_paragraph('关键组件：')
doc.add_paragraph(style='List Bullet').add_run('UNet：提取特征并预测噪声')
doc.add_paragraph(style='List Bullet').add_run('ResnetBlock：带时间条件的残差块')
doc.add_paragraph(style='List Bullet').add_run('Attention：捕捉全局依赖')
doc.add_paragraph(style='List Bullet').add_run('DDIM：加速采样')

import os
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', '扩散模型核心代码详解.docx')
doc.save(desktop_path)
print(f"文档已保存到桌面: {desktop_path}")