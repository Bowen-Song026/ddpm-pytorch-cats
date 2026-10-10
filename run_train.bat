@echo off
chcp 65001
set PYTHONUTF8=1
cd /d "d:\denoising-diffusion-pytorch-main\denoising-diffusion-pytorch-main"
python train_oxford_safe.py
pause
