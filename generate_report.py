import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def generate_report():
    results_path = Path('./experiment_results/all_results.json')
    
    if not results_path.exists():
        print(f"❌ 未找到实验结果文件: {results_path}")
        return
    
    with open(results_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    print(f"✅ 加载了 {len(all_results)} 组实验结果")
    
    results_dir = Path('./experiment_results')
    results_dir.mkdir(exist_ok=True)
    
    df_data = []
    for result in all_results:
        df_data.append({
            '实验名称': result['experiment_name'],
            '状态': result['status'],
            '最终损失': result['final_loss'],
            '总时间(分钟)': round(result['total_time']/60, 1) if result['total_time'] else None,
            '峰值显存(MB)': round(result['peak_memory'], 0) if result['peak_memory'] else None,
            'dim': result['params']['dim'],
            'objective': result['params']['objective'],
            'lr': result['params']['lr'],
            'image_size': result['params']['image_size'],
            'batch_size': result['params']['batch_size'],
            'sampling_timesteps': result['params']['sampling_timesteps'],
            'timesteps': result['params']['timesteps'],
            'ema_decay': result['params']['ema_decay'],
        })
    
    df = pd.DataFrame(df_data)
    excel_path = results_dir / '实验结果汇总.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"📊 表格已保存到 {excel_path}")
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    dim_results = [r for r in all_results if r['experiment_name'].startswith('dim_study')]
    if dim_results:
        dims = [r['params']['dim'] for r in dim_results]
        losses = [r['final_loss'] for r in dim_results]
        
        plt.figure(figsize=(8, 5))
        plt.bar([str(d) for d in dims], losses, color=['blue', 'green', 'orange'])
        plt.title('模型尺寸(dim)对最终损失的影响')
        plt.xlabel('dim')
        plt.ylabel('最终损失')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(results_dir / 'dim_study.png', dpi=150)
        plt.close()
        print(f"📈 dim研究图表已保存")
    
    objective_results = [r for r in all_results if r['experiment_name'].startswith('objective_study')]
    if objective_results:
        objectives = [r['params']['objective'] for r in objective_results]
        losses = [r['final_loss'] for r in objective_results]
        
        plt.figure(figsize=(8, 5))
        plt.bar(objectives, losses, color=['red', 'green', 'blue'])
        plt.title('预测目标(objective)对最终损失的影响')
        plt.xlabel('objective')
        plt.ylabel('最终损失')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(results_dir / 'objective_study.png', dpi=150)
        plt.close()
        print(f"📈 objective研究图表已保存")
    
    lr_results = [r for r in all_results if r['experiment_name'].startswith('lr_study')]
    if lr_results:
        lrs = [f"{r['params']['lr']:.1e}" for r in lr_results]
        losses = [r['final_loss'] for r in lr_results]
        
        plt.figure(figsize=(8, 5))
        plt.plot(lrs, losses, marker='o', linewidth=2, color='purple')
        plt.title('学习率(lr)对最终损失的影响')
        plt.xlabel('lr')
        plt.ylabel('最终损失')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(results_dir / 'lr_study.png', dpi=150)
        plt.close()
        print(f"📈 lr研究图表已保存")
    
    image_size_results = [r for r in all_results if r['experiment_name'].startswith('image_size_study')]
    if image_size_results:
        sizes = [f"{r['params']['image_size']}x{r['params']['image_size']}" for r in image_size_results]
        losses = [r['final_loss'] for r in image_size_results]
        mems = [r['peak_memory'] for r in image_size_results]
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        
        ax1.bar(sizes, losses, color='blue', alpha=0.7, label='最终损失')
        ax2.plot(sizes, mems, marker='o', color='red', label='峰值显存')
        
        ax1.set_title('图像分辨率对损失和显存的影响')
        ax1.set_xlabel('分辨率')
        ax1.set_ylabel('最终损失')
        ax2.set_ylabel('峰值显存(MB)')
        
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.tight_layout()
        plt.savefig(results_dir / 'image_size_study.png', dpi=150)
        plt.close()
        print(f"📈 image_size研究图表已保存")
    
    batch_results = [r for r in all_results if r['experiment_name'].startswith('batch_size_study')]
    if batch_results:
        batches = [r['params']['batch_size'] for r in batch_results]
        losses = [r['final_loss'] for r in batch_results]
        times = [r['total_time']/60 for r in batch_results]
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        
        ax1.bar([str(b) for b in batches], losses, color='green', alpha=0.7, label='最终损失')
        ax2.plot([str(b) for b in batches], times, marker='o', color='orange', label='训练时间')
        
        ax1.set_title('batch_size对损失和训练时间的影响')
        ax1.set_xlabel('batch_size')
        ax1.set_ylabel('最终损失')
        ax2.set_ylabel('训练时间(分钟)')
        
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.tight_layout()
        plt.savefig(results_dir / 'batch_size_study.png', dpi=150)
        plt.close()
        print(f"📈 batch_size研究图表已保存")
    
    ema_results = [r for r in all_results if r['experiment_name'].startswith('ema_decay_study')]
    if ema_results:
        decays = [r['params']['ema_decay'] for r in ema_results]
        losses = [r['final_loss'] for r in ema_results]
        
        plt.figure(figsize=(8, 5))
        plt.plot(decays, losses, marker='o', linewidth=2, color='cyan')
        plt.title('EMA衰减率(ema_decay)对最终损失的影响')
        plt.xlabel('ema_decay')
        plt.ylabel('最终损失')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(results_dir / 'ema_decay_study.png', dpi=150)
        plt.close()
        print(f"📈 ema_decay研究图表已保存")
    
    timesteps_results = [r for r in all_results if r['experiment_name'].startswith('timesteps_study')]
    if timesteps_results:
        timesteps = [r['params']['timesteps'] for r in timesteps_results]
        losses = [r['final_loss'] for r in timesteps_results]
        times = [r['total_time']/60 for r in timesteps_results]
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        
        ax1.bar([str(t) for t in timesteps], losses, color='magenta', alpha=0.7, label='最终损失')
        ax2.plot([str(t) for t in timesteps], times, marker='o', color='blue', label='训练时间')
        
        ax1.set_title('扩散时间步(timesteps)对损失和训练时间的影响')
        ax1.set_xlabel('timesteps')
        ax1.set_ylabel('最终损失')
        ax2.set_ylabel('训练时间(分钟)')
        
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.tight_layout()
        plt.savefig(results_dir / 'timesteps_study.png', dpi=150)
        plt.close()
        print(f"📈 timesteps研究图表已保存")
    
    sampling_results = [r for r in all_results if r['experiment_name'].startswith('sampling_timesteps_study')]
    if sampling_results:
        sampling_steps = [r['params']['sampling_timesteps'] for r in sampling_results]
        losses = [r['final_loss'] for r in sampling_results]
        
        plt.figure(figsize=(8, 5))
        plt.plot(sampling_steps, losses, marker='o', linewidth=2, color='brown')
        plt.title('采样步数(sampling_timesteps)对最终损失的影响')
        plt.xlabel('sampling_timesteps')
        plt.ylabel('最终损失')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(results_dir / 'sampling_timesteps_study.png', dpi=150)
        plt.close()
        print(f"📈 sampling_timesteps研究图表已保存")
    
    print(f"\n🎉 实验报告生成完成!")
    print(f"📁 报告目录: {results_dir}")

if __name__ == '__main__':
    generate_report()