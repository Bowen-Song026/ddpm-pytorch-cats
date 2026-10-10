import subprocess
import sys
import os

os.environ['PYTHONUTF8'] = '1'
result = subprocess.run(
    [sys.executable, '-u', 'train_mixed_cats.py'],
    capture_output=True,
    text=True,
    cwd='d:\\denoising-diffusion-pytorch-main\\denoising-diffusion-pytorch-main',
    timeout=120
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
