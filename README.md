# ddpm-pytorch-cats：基于 DDPM 的猫图生成实验工程

> A portfolio of DDPM (Denoising Diffusion Probabilional Model) training and ablation experiments for cat face image generation, adapted from `lucidrains/denoising-diffusion-pytorch` to run entirely on a consumer-grade 4GB GPU (NVIDIA RTX 3050 Laptop).

本项目基于 [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) 二次开发，围绕 DDPM（去噪扩散概率模型）原理学习与猫图生成展开。在保留上游核心实现（`denoising_diffusion_pytorch` 包）的前提下，围绕消费级 4GB 显存 GPU 完成了完整训练、多组消融实验与报告生成的工程实践。上游原版英文 README 保留为 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。

![Python](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5OCIgaGVpZ2h0PSIyMCI+PHJlY3Qgd2lkdGg9Ijk4IiBoZWlnaHQ9IjIwIiByeD0iMyIgZmlsbD0iIzM3NzZBQiIvPjx0ZXh0IHg9IjQ5IiB5PSIxNCIgZm9udC1mYW1pbHk9IlZlcmRhbmEsR2VuZXZhLERlamFWdSBTYW5zLHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZpbGw9IiNmZmYiIHRleHQtYW5jaG9yPSJtaWRkbGUiPlB5dGhvbiAzLjgrPC90ZXh0Pjwvc3ZnPg==)
![PyTorch](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMTAiIGhlaWdodD0iMjAiPjxyZWN0IHdpZHRoPSIxMTAiIGhlaWdodD0iMjAiIHJ4PSIzIiBmaWxsPSIjRUU0QzJDIi8+PHRleHQgeD0iNTUiIHk9IjE0IiBmb250LWZhbWlseT0iVmVyZGFuYSxHZW5ldmEsRGVqYVZ1IFNhbnMsc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSI+UHlUb3JjaCAyLjArPC90ZXh0Pjwvc3ZnPg==)
![License](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5MiIgaGVpZ2h0PSIyMCI+PHJlY3Qgd2lkdGg9IjkyIiBoZWlnaHQ9IjIwIiByeD0iMyIgZmlsbD0iIzAwN0VDNiIvPjx0ZXh0IHg9IjQ2IiB5PSIxNCIgZm9udC1mYW1pbHk9IlZlcmRhbmEsR2VuZXZhLERlamFWdSBTYW5zLHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZpbGw9IiNmZmYiIHRleHQtYW5jaG9yPSJtaWRkbGUiPkxpY2Vuc2UgTUlUPC90ZXh0Pjwvc3ZnPg==)

---

## 目录

1. [项目背景与动机](#项目背景与动机)
2. [硬件与环境](#硬件与环境)
3. [主要工作](#主要工作)
4. [关键发现](#关键发现)
5. [目录结构](#目录结构)
6. [快速开始](#快速开始)
7. [与上游项目差异](#与上游项目差异)
8. [关键文件速查表](#关键文件速查表)
9. [License](#license)
10. [致谢](#致谢)

---

## 项目背景与动机

本项目是一个学习 DDPM 原理并完成完整工程实践的工程作品集。出发点有两点：

1. **学习扩散模型原理**：在 PyTorch 上从零跟踪 DDPM 的训练与采样流程，理解去噪目标、时间嵌入、U-Net 结构、EMA、DDIM 采样等核心组件，而不只是调用高阶 API。
2. **在 4GB 消费级 GPU 上做完整训练 + 消融实验**：把一个面向高配 GPU / TPU 的工业级实现（lucidrains 版），改造为在 NVIDIA RTX 3050 Laptop GPU（4GB VRAM）上能真正跑通、能保存权重、能批量采样、能产出实验报告的完整工程流水线。

围绕上述目标，本项目对模型规模、训练超参、目标函数、采样步数做了系统消融，并把结果整理成 Excel/Word 报告，方便复盘与展示。

---

## 硬件与环境

| 项目 | 配置 / 版本 |
| --- | --- |
| GPU | NVIDIA RTX 3050 Laptop GPU（4GB VRAM） |
| 操作系统 | Windows |
| Python | 3.8+ |
| PyTorch | 2.0+ |
| 核心依赖 | `accelerate`、`einops`、`ema-pytorch>=0.4.2`、`numpy`、`pillow`、`pytorch-fid`、`scipy`、`torchvision`、`tqdm` |
| 模型规模 | `dim=32` / `dim=64` / `dim=128` 三档 |
| 输入分辨率 | `32×32` 与 `64×64` 两档 |
| 数据集 | Oxford-IIIT Pet Dataset 的猫图子集（`600×400` 下采样到 `64×64`） |

关键显存预算：

- `dim=64` + `64×64` 图 → 约 2GB VRAM，4GB GPU 可行。
- `dim=128` + `dim_mults=(1,2,4)` + `batch_size=2` + `gradient_accumulate_every=8` + 混合精度（`amp=True`）→ 峰值约 2.5GB VRAM，4GB GPU 可跑。

---

## 主要工作

主要工作围绕四类消融实验、多个版本的训练/生成脚本，以及报告生成流水线展开。

### 1. 四类消融实验

| 消融类型 | 脚本 | 目的 |
| --- | --- | --- |
| `batch_size` 消融 | `batch_size_experiment.py` | 在显存约束下对比不同 `batch_size` × `gradient_accumulate_every` 组合的收敛与质量 |
| 学习率消融 | `lr_experiment.py` | 对比不同 `train_lr` 对收敛速度与采样质量的影响 |
| 目标函数消融 | `objective_experiment.py` | 对比 `pred_x0` / `pred_noise` / `pred_v` 等不同目标函数 |
| 采样步数消融 | `sampling_steps_test.py` | 对比不同 `sampling_timesteps`（DDIM 步数）下的采样质量与耗时 |

### 2. 训练脚本矩阵

为覆盖不同模型规模、不同数据来源、不同显存策略，本项目保留了多版本训练脚本：

- 基础与改进：`train_cat.py`、`train_cat_simple.py`、`train_cat_improved.py`、`train_cat_efficient.py`、`train_cat_fast.py`
- 分辨率/规模专项：`train_cat_64x64.py`、`train_dim32.py`、`train_dim128.py`
- Oxford-IIIT 数据集：`train_oxford_cats.py`、`train_oxford_safe.py`、`train_oxford_standalone.py`
- 真实猫图与混合：`train_real_cats.py`、`train_real_cats_fast.py`、`train_mixed_cats.py`、`train_standalone_v2.py`

### 3. 生成脚本

对应不同模型与采样步数的生成脚本：

`generate_cats.py`、`generate_better_cats.py`、`generate_best_cats.py`、`generate_dim128.py`、`generate_dim128_highres.py`、`generate_dim128_250steps.py`、`generate_dim32_500steps.py`、`generate_images.py`

### 4. 实验汇总与报告生成

- 实验汇总：`experiment_runner.py`
- 报告/表格生成：`generate_excel.py`、`generate_experiment_report.py`、`generate_report.py`、`generate_code_docx.py`、`generate_docx.py`
- 结果回填：`update_excel_with_image_analysis.py`、`update_excel_with_lr.py`、`update_excel_with_objective.py`、`update_excel_with_sampling.py`

### 5. 模型权重保存

训练过程中每 2000 步保存一次检查点：`model-1.pt` ~ `model-N.pt` + `model-latest.pt`，并同步保存 EMA 权重 `ema-X.pt`，便于回溯与采样复现。

---

## 关键发现

以下九条为本项目在 4GB 显存约束下做实验得出的关键工程经验：

1. **简单几何图形数据集生成质量受限**：在简单几何图形数据集上训练得到的生成图像是低质量卡通画，与真实物体差距大，难以体现 DDPM 的真实生成能力。
2. **小规模 + 小分辨率不够**：`dim=32` + `20 epochs` + `32×32` 图的生成质量明显受限，仅适合做原理验证，不适合做真实图像生成。
3. **`64×64` + `dim=64` 是 4GB GPU 的甜点配置**：该组合约用 2GB VRAM，4GB GPU 可稳定训练，是性价比最高的可行基线。
4. **训练中采样会瞬时拉满显存导致 OOM**：训练循环里调用 `.sample()` 会在原训练显存之上叠加采样显存，4GB 卡极易 OOM，必须跳过训练中采样（skip in-training sampling）。
5. **Oxford-IIIT Pet Dataset 猫图特征比 CIFAR-10 好**：把 `600×400` 的 Oxford 猫图下采样到 `64×64`，特征比 CIFAR-10 更聚焦、更适合做猫图生成实验。
6. **Oxford 数据集品种名下划线 bug**：Oxford 数据集中部分品种名含下划线（如 `British_Shorthair`）会被截断，导致约 198 张图被错误丢失，需要专门处理。
7. **Windows 上 `DataLoader` 必须用 `num_workers=0`**：Windows 下多进程 DataLoader 会触发权限/路径错误，必须设 `num_workers=0` 才能稳定训练。
8. **后台僵尸 Python 进程会导致训练中断**：后台残留的僵尸 Python 进程会占用显存或文件锁，导致训练莫名中断，需定期清理。
9. **`dim=128` + `dim_mults=(1,2,4)` + `batch_size=2` + `gradient_accumulate_every=8` + 混合精度 → 峰值约 2.5GB VRAM**：这是 4GB GPU 上能跑得动的较大配置，配合梯度累积与 `amp=True` 可控显存。

---

## 目录结构

```text
ddpm-pytorch-cats/
├── denoising_diffusion_pytorch/      # 上游核心实现包（U-Net / GaussianDiffusion / Trainer），保留
├── data/                              # 训练数据集根目录（不入库，体积过大）
├── oxford_pets_data/                 # Oxford-IIIT Pet 数据集（不入库）
├── cat_faces/                        # 处理后的猫图数据（不入库）
├── cat_faces_improved/               # 改进后的猫图数据（不入库）
├── real_cat_images/                  # 真实猫图原始数据（不入库）
├── real_cat_images_64x64/            # 真实猫图 64×64 版本（不入库）
├── oxford_cat_images/                # Oxford 猫图子集（不入库）
│
├── *_results*/                       # 各训练/生成结果目录（不入库，体积过大）
│   ├── real_cat_results/             #   真实猫图训练结果
│   ├── real_cat_results_64x64/       #   64×64 训练结果
│   ├── real_cat_results_efficient/   #   高效训练结果
│   ├── real_cat_results_improved/    #   改进训练结果
│   ├── mixed_cat_results/            #   混合猫图结果
│   ├── dim128_cat_results/           #   dim=128 结果
│   ├── dim32_cat_results/            #   dim=32 结果
│   ├── oxford_cat_results/           #   Oxford 结果
│   ├── oxford_cat_results_safe/      #   Oxford 安全模式结果
│   ├── batch_experiment_*/           #   batch 消融各组合结果
│   ├── best_cat_images/             #   精选生成图
│   ├── optimized_cat_images/        #   优化生成图
│   ├── cat_results_simple/          #   简化训练结果
│   ├── cat_results_improved/        #   改进训练结果
│   ├── generated_images/            #   通用生成图
│   └── playground_output/            #   临时实验输出
│
├── *.pt                              # 模型权重（不入库，可由训练重新生成）
│   ├── model-1.pt ~ model-N.pt       #   每 2000 步一存
│   ├── model-latest.pt              #   最新权重
│   └── ema-X.pt                      #   EMA 权重
│
├── batch_size_experiment.py          # batch_size 消融
├── lr_experiment.py                  # 学习率消融
├── objective_experiment.py          # 目标函数消融
├── sampling_steps_test.py           # 采样步数消融
├── experiment_runner.py              # 实验汇总运行器
│
├── train_cat*.py                    # 训练脚本（多版本）
├── train_dim32.py / train_dim128.py  # 不同模型规模训练
├── train_oxford_*.py                # Oxford 数据集训练
├── train_real_cats*.py               # 真实猫图训练
├── train_mixed_cats.py               # 混合猫图训练
├── train_standalone_v2.py            # 独立训练脚本 v2
│
├── generate_cats*.py                # 猫图生成脚本
├── generate_dim*.py                  # dim 系列生成脚本
├── generate_images.py                # 通用生成脚本
│
├── generate_excel.py                 # 生成 Excel 报告
├── generate_experiment_report.py     # 生成实验报告
├── generate_report.py                # 生成报告
├── generate_code_docx.py             # 生成核心代码 docx
├── generate_docx.py                  # 生成 docx
├── update_excel_with_*.py            # 结果回填到 Excel
│
├── 参数对比表.xlsx                    # 参数对比表
├── 实验结果对比表.xlsx                 # 实验结果对比表
├── 扩散模型核心代码.docx               # 核心代码文档
├── 扩散模型核心代码详解.docx           # 核心代码详解文档
│
├── download_cifar.py                 # CIFAR 下载脚本
├── download_cifar_resume.py          # CIFAR 续传下载
├── run_mixed.py                      # 混合训练运行入口
├── run_train.bat                     # Windows 训练批处理
├── setup.py                          # 包安装配置
├── LICENSE                           # MIT License
├── UPSTREAM_README.md                # 上游 lucidrains 原版英文 README
└── README.md                         # 本文件
```

---

## 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/Bowen-Song026/ddpm-pytorch-cats.git
cd ddpm-pytorch-cats

# 以可编辑模式安装上游核心包（提供 denoising_diffusion_pytorch 模块）
pip install -e .
```

依赖见 `setup.py` 的 `install_requires`（`torch>=2.0`、`torchvision`、`accelerate`、`einops`、`ema-pytorch>=0.4.2`、`numpy`、`pillow`、`pytorch-fid`、`scipy`、`tqdm`）。

### 2. 数据准备

准备 Oxford-IIIT Pet Dataset 的猫图子集，放置到 `oxford_pets_data/`，处理后下采样到 `64×64` 放入 `cat_faces/` 或 `real_cat_images_64x64/`。

> 注意：Oxford 数据集存在品种名下划线 bug（如 `British_Shorthair` 会被截断，约 198 张图丢失），处理脚本需做相应修复。

### 3. 训练

以 `dim=64` + `64×64` 图为例（4GB GPU 的甜点配置）：

```bash
python train_cat_64x64.py
```

其它规模：

```bash
# dim=32（原理验证）
python train_dim32.py

# dim=128（4GB GPU 极限配置，batch_size=2 + gradient_accumulate_every=8 + 混合精度）
python train_dim128.py

# Oxford 数据集
python train_oxford_cats.py
```

> Windows 注意：`DataLoader` 必须设 `num_workers=0`，否则会触发权限错误。训练中采样会瞬时拉满 4GB 显存导致 OOM，相关训练脚本已跳过 in-training sampling。

### 4. 生成

```bash
# dim=128，250 步 DDIM 采样
python generate_dim128_250steps.py

# dim=32，500 步采样
python generate_dim32_500steps.py

# 高分辨率
python generate_dim128_highres.py

# 精选生成图
python generate_best_cats.py
```

### 5. 消融实验

```bash
# batch_size 消融
python batch_size_experiment.py

# 学习率消融
python lr_experiment.py

# 目标函数消融
python objective_experiment.py

# 采样步数消融
python sampling_steps_test.py

# 一键汇总运行
python experiment_runner.py
```

### 6. 报告生成

```bash
# 生成 Excel 实验结果对比表
python generate_excel.py

# 生成完整实验报告
python generate_experiment_report.py

# 生成核心代码 docx
python generate_code_docx.py

# 把各类消融结果回填到 Excel
python update_excel_with_image_analysis.py
python update_excel_with_lr.py
python update_excel_with_objective.py
python update_excel_with_sampling.py
```

---

## 与上游项目差异

本项目在上游 `lucidrains/denoising-diffusion-pytorch` 基础上做的改动：

1. **保留上游核心实现**：`denoising_diffusion_pytorch` 包（U-Net / GaussianDiffusion / Trainer / 1D 变体等）原样保留，作为底层引擎。
2. **面向 4GB GPU 的训练适配**：新增多版本训练脚本（`train_cat*.py`、`train_dim*.py`、`train_oxford_*.py`、`train_real_cats*.py` 等），针对小显存做了 `batch_size` + `gradient_accumulate_every` + 混合精度的组合调优，并禁用训练中采样以避开 OOM。
3. **数据集替换为猫图**：训练数据由示例随机张量/CIFAR 切换为 Oxford-IIIT Pet Dataset 的猫图子集（`600×400` 下采样到 `64×64`），并处理了 Oxford 品种名下划线 bug。
4. **系统消融实验**：新增 `batch_size`、学习率、目标函数、采样步数四类消融脚本，并配 `experiment_runner.py` 汇总。
5. **报告生成流水线**：新增 Excel/Word 报告生成与结果回填脚本，把训练、采样、消融结果整理为 `参数对比表.xlsx`、`实验结果对比表.xlsx`、`扩散模型核心代码.docx` 等可交付文档。
6. **原版 README 保留**：上游英文 README 改名为 `UPSTREAM_README.md`，本 README 为面向本仓库实验内容的中文说明。
7. **`.gitignore` 收紧**：训练数据、结果目录、模型权重、Office 临时锁文件、上游 workflow 文件等均不入库。

---

## 关键文件速查表

| 文件名 | 作用 | 主要依赖 |
| --- | --- | --- |
| `denoising_diffusion_pytorch/` | 上游核心实现包（U-Net / Diffusion / Trainer） | `torch`、`einops`、`ema-pytorch`、`accelerate` |
| `batch_size_experiment.py` | `batch_size` × `gradient_accumulate_every` 消融 | `denoising_diffusion_pytorch`、`torch` |
| `lr_experiment.py` | 学习率消融 | `denoising_diffusion_pytorch`、`torch` |
| `objective_experiment.py` | 目标函数（`pred_x0` / `pred_noise` / `pred_v`）消融 | `denoising_diffusion_pytorch`、`torch` |
| `sampling_steps_test.py` | 采样步数（DDIM）消融 | `denoising_diffusion_pytorch`、`torch` |
| `experiment_runner.py` | 汇总运行各类消融 | 上述消融脚本 |
| `train_cat.py` | 猫图训练基础版 | `denoising_diffusion_pytorch`、`torch`、`torchvision` |
| `train_cat_simple.py` | 简化训练入口 | 同上 |
| `train_cat_improved.py` | 改进训练（含调优） | 同上 |
| `train_cat_efficient.py` | 高效训练（显存优化） | 同上 |
| `train_cat_fast.py` | 快速训练 | 同上 |
| `train_cat_64x64.py` | `64×64` 分辨率训练（甜点配置） | 同上 |
| `train_dim32.py` | `dim=32` 模型训练 | 同上 |
| `train_dim128.py` | `dim=128` 模型训练（4GB 极限） | 同上 |
| `train_oxford_cats.py` | Oxford 数据集训练 | 同上 |
| `train_oxford_safe.py` | Oxford 安全模式训练 | 同上 |
| `train_oxford_standalone.py` | Oxford 独立训练 | 同上 |
| `train_real_cats.py` | 真实猫图训练 | 同上 |
| `train_real_cats_fast.py` | 真实猫图快速训练 | 同上 |
| `train_mixed_cats.py` | 混合猫图训练 | 同上 |
| `train_standalone_v2.py` | 独立训练脚本 v2 | 同上 |
| `generate_cats.py` | 基础猫图生成 | `denoising_diffusion_pytorch`、`torch` |
| `generate_better_cats.py` | 改进猫图生成 | 同上 |
| `generate_best_cats.py` | 精选猫图生成 | 同上 |
| `generate_dim128.py` | `dim=128` 生成 | 同上 |
| `generate_dim128_highres.py` | `dim=128` 高分辨率生成 | 同上 |
| `generate_dim128_250steps.py` | `dim=128` + 250 步生成 | 同上 |
| `generate_dim32_500steps.py` | `dim=32` + 500 步生成 | 同上 |
| `generate_images.py` | 通用生成脚本 | 同上 |
| `generate_excel.py` | 生成 Excel 报告 | `openpyxl` / `pandas` |
| `generate_experiment_report.py` | 生成实验报告 | Python 文档/表格库 |
| `generate_report.py` | 生成报告 | 同上 |
| `generate_code_docx.py` | 生成核心代码 docx | `python-docx` |
| `generate_docx.py` | 生成 docx | `python-docx` |
| `update_excel_with_image_analysis.py` | 图像分析结果回填 Excel | `openpyxl` / `pandas` |
| `update_excel_with_lr.py` | 学习率结果回填 Excel | 同上 |
| `update_excel_with_objective.py` | 目标函数结果回填 Excel | 同上 |
| `update_excel_with_sampling.py` | 采样步数结果回填 Excel | 同上 |
| `download_cifar.py` | CIFAR 下载 | `torchvision` |
| `download_cifar_resume.py` | CIFAR 续传下载 | 同上 |
| `run_mixed.py` | 混合训练运行入口 | 训练脚本 |
| `run_train.bat` | Windows 训练批处理 | 训练脚本 |
| `setup.py` | 包安装配置 | `setuptools` |
| `LICENSE` | MIT License | - |
| `UPSTREAM_README.md` | 上游 lucidrains 原版英文 README | - |

---

## License

本项目沿用上游 [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) 的 **MIT License**，版权声明见 [`LICENSE`](./LICENSE)（Copyright (c) 2020 Phil Wang）。本仓库新增的训练、消融、生成、报告脚本同样在 MIT 许可下发布。

---

## 致谢

本项目基于 [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) 二次开发，感谢原作者 **lucidrains (Phil Wang)** 提供高质量、可读性极强的 PyTorch DDPM 参考实现，以及 DDPM / DDIM / Improved DDPM / Elucidating 等系列论文作者的理论贡献。上游原版 README 保留为 [`UPSTREAM_README.md`](./UPSTREAM_README.md)。数据集使用 Oxford-IIIT Pet Dataset。
