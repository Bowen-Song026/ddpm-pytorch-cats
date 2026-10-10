import sys
import os
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
from torchvision import transforms as T, datasets
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA

pkg_dir = os.path.join(os.path.dirname(__file__), 'denoising_diffusion_pytorch')
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

import types
pkg = types.ModuleType('denoising_diffusion_pytorch')
pkg.__path__ = [pkg_dir]
sys.modules['denoising_diffusion_pytorch'] = pkg

import attend
pkg.attend = attend
import version
pkg.version = version

import importlib.util
spec = importlib.util.spec_from_file_location(
    "denoising-diffusion-pytorch.denoising_diffusion_pytorch",
    os.path.join(pkg_dir, "denoising_diffusion_pytorch.py")
)
main_mod = importlib.util.module_from_spec(spec)
sys.modules["denoising-diffusion-pytorch.denoising_diffusion_pytorch"] = main_mod
spec.loader.exec_module(main_mod)
pkg.denoising_diffusion_pytorch = main_mod

Unet = main_mod.Unet
GaussianDiffusion = main_mod.GaussianDiffusion

class OxfordCatDataset(Dataset):
    def __init__(self, oxford_dir, image_size=64):
        self.image_size = image_size
        self.paths = []
        for f in os.listdir(oxford_dir):
            if f.endswith('.jpg') and f[0].isupper():
                self.paths.append(os.path.join(oxford_dir, f))
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

class CIFAR10CatDataset(Dataset):
    def __init__(self, root='./data', image_size=64, train=True):
        self.cifar = datasets.CIFAR10(root=root, train=train, download=True, transform=None)
        self.cat_indices = [i for i, (_, label) in enumerate(self.cifar) if label == 3]
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    def __len__(self):
        return len(self.cat_indices)
    def __getitem__(self, idx):
        img, _ = self.cifar[self.cat_indices[idx]]
        return self.transform(img.convert('RGB'))

def cycle(dl):
    while True:
        for data in dl:
            yield data

BASE_CONFIG = {
    'dim': 64,
    'dim_mults': (1, 2, 4, 8),
    'flash_attn': False,
    'image_size': 64,
    'timesteps': 1000,
    'sampling_timesteps': 50,
    'objective': 'pred_v',
    'batch_size': 4,
    'gradient_accumulate_every': 4,
    'lr': 8e-5,
    'ema_decay': 0.995,
    'train_num_steps': 10000,
    'amp': True,
}

EXPERIMENTS = [
    {
        'name': 'dim_study',
        'param_name': 'dim',
        'values': [32, 64],
        'fixed_params': BASE_CONFIG.copy(),
    },
    {
        'name': 'objective_study',
        'param_name': 'objective',
        'values': ['pred_noise', 'pred_x0', 'pred_v'],
        'fixed_params': BASE_CONFIG.copy(),
    },
    {
        'name': 'lr_study',
        'param_name': 'lr',
        'values': [5e-5, 8e-5, 1e-4],
        'fixed_params': BASE_CONFIG.copy(),
    },
    {
        'name': 'sampling_timesteps_study',
        'param_name': 'sampling_timesteps',
        'values': [25, 50, 100],
        'fixed_params': BASE_CONFIG.copy(),
    },
    {
        'name': 'image_size_study',
        'param_name': 'image_size',
        'values': [32, 64],
        'fixed_params': BASE_CONFIG.copy(),
    },
    {
        'name': 'batch_size_study',
        'param_name': 'batch_size',
        'values': [2, 4, 8],
        'fixed_params': BASE_CONFIG.copy(),
    },
    {
        'name': 'ema_decay_study',
        'param_name': 'ema_decay',
        'values': [0.99, 0.995, 0.999],
        'fixed_params': BASE_CONFIG.copy(),
    },
    {
        'name': 'timesteps_study',
        'param_name': 'timesteps',
        'values': [500, 1000, 2000],
        'fixed_params': BASE_CONFIG.copy(),
    },
]

def run_single_experiment(exp_name, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"实验: {exp_name}")
    print(f"参数: {params}")
    print(f"设备: {device}")
    print(f"{'='*60}")
    
    image_size = params['image_size']
    
    try:
        oxford_dir = './oxford_pets_data/images'
        oxford_ds = OxfordCatDataset(oxford_dir, image_size=image_size)
    except:
        oxford_ds = None
    
    cifar_ds = CIFAR10CatDataset(root='./data', image_size=image_size, train=True)
    
    if oxford_ds is not None and len(oxford_ds) > 0:
        combined_ds = ConcatDataset([oxford_ds, cifar_ds])
    else:
        combined_ds = cifar_ds
    
    dl = DataLoader(combined_ds, batch_size=params['batch_size'], shuffle=True, pin_memory=True, num_workers=0)
    dl_iter = cycle(dl)
    
    model = Unet(
        dim=params['dim'],
        dim_mults=params['dim_mults'],
        flash_attn=params['flash_attn']
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model,
        image_size=params['image_size'],
        timesteps=params['timesteps'],
        sampling_timesteps=params['sampling_timesteps'],
        objective=params['objective']
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.99))
    ema = EMA(model, beta=params['ema_decay'], update_every=10).to(device)
    
    train_num_steps = params['train_num_steps']
    gradient_accumulate_every = params['gradient_accumulate_every']
    amp = params['amp']
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    results = {
        'experiment_name': exp_name,
        'params': params,
        'loss_history': [],
        'time_history': [],
        'final_loss': None,
        'total_time': None,
        'peak_memory': 0,
        'status': 'running',
    }
    
    start_time = time.time()
    model.train()
    
    try:
        for step in tqdm(range(train_num_steps), desc=f'{exp_name}训练', unit='step'):
            optimizer.zero_grad()
            total_loss = 0.0
            
            for _ in range(gradient_accumulate_every):
                data = next(dl_iter).to(device)
                with torch.cuda.amp.autocast(enabled=amp):
                    loss = diffusion(data) / gradient_accumulate_every
                scaler.scale(loss).backward()
                total_loss += loss.item()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            
            if step % 100 == 0:
                results['loss_history'].append({'step': step, 'loss': total_loss})
                results['time_history'].append({'step': step, 'elapsed': time.time() - start_time})
                
                if device.type == 'cuda':
                    mem_used = torch.cuda.memory_allocated(0) / 1024**2
                    if mem_used > results['peak_memory']:
                        results['peak_memory'] = mem_used
                
                print(f"  step {step}/{train_num_steps} | loss: {total_loss:.4f} | mem: {mem_used:.0f}MB")
        
        results['final_loss'] = total_loss
        results['total_time'] = time.time() - start_time
        results['status'] = 'completed'
        
        print(f"\n✅ 实验完成!")
        print(f"   最终损失: {total_loss:.4f}")
        print(f"   总时间: {results['total_time']/60:.1f}分钟")
        print(f"   峰值显存: {results['peak_memory']:.0f}MB")
        
    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        print(f"\n❌ 实验失败: {e}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return results

def main():
    all_results = []
    
    for experiment in EXPERIMENTS:
        exp_name = experiment['name']
        param_name = experiment['param_name']
        values = experiment['values']
        fixed_params = experiment['fixed_params']
        
        print(f"\n\n{'#'*80}")
        print(f"开始参数研究: {exp_name}")
        print(f"研究参数: {param_name}")
        print(f"取值范围: {values}")
        print(f"{'#'*80}")
        
        for value in values:
            params = fixed_params.copy()
            params[param_name] = value
            
            result = run_single_experiment(f"{exp_name}_{param_name}_{value}", params)
            all_results.append(result)
            
            results_dir = Path('./experiment_results')
            results_dir.mkdir(exist_ok=True)
            
            with open(results_dir / 'all_results.json', 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n📊 结果已保存到 {results_dir / 'all_results.json'}")
    
    print(f"\n\n{'#'*80}")
    print(f"所有实验完成! 共 {len(all_results)} 组实验")
    print(f"{'#'*80}")

if __name__ == '__main__':
    main()