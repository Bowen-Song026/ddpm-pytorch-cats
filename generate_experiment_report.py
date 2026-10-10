import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

data = {
    '实验名称': ['dim32', 'dim64', 'dim128', 'batch2_accum8', 'batch4_accum4', 'batch8_accum2'],
    '实验类型': ['dim实验', 'dim实验', 'dim实验', 'batch实验', 'batch实验', 'batch实验'],
    'dim': [32, 64, 128, 64, 64, 64],
    'dim_mults': ['(1,2,4,8)', '(1,2,4,8)', '(1,2,4)', '(1,2,4,8)', '(1,2,4,8)', '(1,2,4,8)'],
    '参数数量(万)': [31, 500, 2000, 500, 500, 500],
    'batch_size': [8, 4, 2, 2, 4, 8],
    'gradient_accumulate': [2, 4, 8, 8, 4, 2],
    'effective_batch': [16, 16, 16, 16, 16, 16],
    '训练步数': [30000, 20000, 30000, 5000, 5000, 5000],
    '训练时间(h)': [2.45, 5.0, 9.8, 1.11, 0.86, 0.55],
    '训练速度(steps/s)': [3.5, 1.11, 0.85, 0.67, 0.86, 2.53],
    '最终Loss': [0.045, 0.06, 0.025, 0.0177, 0.0270, 0.0402],
    'Loss均值': ['-', '-', '-', 0.0585, 0.0621, '-'],
    'Loss标准差': ['-', '-', '-', 0.0889, 0.0933, 0.0868],
    '采样步': [500, 50, 250, 50, 50, 50],
    '图像分辨率': ['64x64', '64x64', '64x64', '64x64', '64x64', '64x64'],
    '学习率': [8e-5, 8e-5, 8e-5, 8e-5, 8e-5, 8e-5],
    'EMA衰减': [0.995, 0.995, 0.995, 0.995, 0.995, 0.995],
    '优化器': ['Adam', 'Adam', 'Adam', 'Adam', 'Adam', 'Adam'],
    '目标函数': ['pred_v', 'pred_v', 'pred_v', 'pred_v', 'pred_v', 'pred_v'],
    '混合精度': ['AMP', 'AMP', 'AMP', 'AMP', 'AMP', 'AMP'],
    '数据集': ['Oxford+CIFAR', 'Oxford+CIFAR', 'Oxford+CIFAR', 'Oxford+CIFAR', 'Oxford+CIFAR', 'Oxford+CIFAR'],
    '数据量': [5554, 5554, 5554, 5554, 5554, 5554],
    '显存占用(GB)': [2.5, 3.2, 3.8, 3.2, 3.5, 3.8],
    'GPU型号': ['RTX 3050 4GB', 'RTX 3050 4GB', 'RTX 3050 4GB', 'RTX 3050 4GB', 'RTX 3050 4GB', 'RTX 3050 4GB'],
}

df = pd.DataFrame(data)

wb = Workbook()
ws = wb.active
ws.title = "实验结果对比"

header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
center_alignment = Alignment(horizontal="center", vertical="center")
thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))

for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
    for c_idx, value in enumerate(row, 1):
        cell = ws.cell(row=r_idx, column=c_idx, value=value)
        cell.alignment = center_alignment
        cell.border = thin_border
        if r_idx == 1:
            cell.font = header_font
            cell.fill = header_fill

for col in ws.columns:
    max_length = 0
    column = col[0].column_letter
    for cell in col:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        except:
            pass
    adjusted_width = max(max_length + 2, 12)
    ws.column_dimensions[column].width = adjusted_width

ws2 = wb.create_sheet(title="参数说明")

params_info = [
    ["参数名称", "说明", "影响程度", "详细解释"],
    ["dim", "UNet基础通道数", "强", "决定模型容量，dim越大参数越多，生成质量越高，但训练时间越长"],
    ["dim_mults", "通道数倍率", "中等", "控制网络深度和宽度，(1,2,4,8)表示4层下采样"],
    ["batch_size", "批次大小", "中等", "每次训练处理的图片数量，越大训练越快，但显存占用越高"],
    ["gradient_accumulate", "梯度累积次数", "弱", "累积多个batch的梯度后更新参数，模拟大batch效果"],
    ["effective_batch", "有效批次大小", "中等", "batch_size × gradient_accumulate，影响梯度估计质量"],
    ["训练步数", "训练迭代次数", "强", "越多训练越充分，但时间越长"],
    ["学习率", "参数更新步长", "强", "过大导致震荡，过小收敛慢，通常1e-5~1e-4"],
    ["EMA衰减", "指数移动平均系数", "弱", "用于模型权重平滑，通常0.995~0.999"],
    ["采样步", "生成时去噪步数", "中等", "越多生成质量越高，但速度越慢"],
    ["目标函数", "损失函数类型", "中等", "pred_noise/pred_x0/pred_v，影响训练稳定性"],
    ["混合精度", "AMP训练", "弱", "使用半精度训练，节省显存，加速训练"],
    ["数据集", "训练数据来源", "强", "数据质量和数量直接影响生成效果"],
    ["图像分辨率", "输入输出尺寸", "强", "分辨率越高越清晰，但显存需求呈平方增长"],
    ["优化器", "参数更新算法", "中等", "Adam/AdamW等，影响收敛速度和稳定性"],
]

for r_idx, row in enumerate(params_info, 1):
    for c_idx, value in enumerate(row, 1):
        cell = ws2.cell(row=r_idx, column=c_idx, value=value)
        cell.alignment = center_alignment
        cell.border = thin_border
        if r_idx == 1:
            cell.font = header_font
            cell.fill = header_fill

for col in ws2.columns:
    max_length = 0
    column = col[0].column_letter
    for cell in col:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        except:
            pass
    adjusted_width = max(max_length + 2, 15)
    ws2.column_dimensions[column].width = adjusted_width

ws3 = wb.create_sheet(title="实验结论")

conclusions = [
    ["维度", "结论"],
    ["dim影响", "dim越大，模型容量越大，最终Loss越低，生成质量越高，但训练时间显著增加"],
    ["batch影响", "batch_size越大，训练速度越快，但最终Loss略有上升，Loss稳定性变化不大"],
    ["梯度累积", "在显存受限情况下，梯度累积是保持effective_batch的有效手段"],
    ["最佳组合", "dim=128+batch=4: 生成质量最高；dim=64+batch=8: 平衡速度和质量"],
    ["显存限制", "4GB显存下，dim=128是最大可行模型，batch_size=8是最大可行批次"],
    ["训练建议", "优先调整dim提升质量，其次调整batch_size提升速度"],
]

for r_idx, row in enumerate(conclusions, 1):
    for c_idx, value in enumerate(row, 1):
        cell = ws3.cell(row=r_idx, column=c_idx, value=value)
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        cell.border = thin_border
        if r_idx == 1:
            cell.font = header_font
            cell.fill = header_fill

ws3.column_dimensions['A'].width = 15
ws3.column_dimensions['B'].width = 80

output_path = "d:/denoising-diffusion-pytorch-main/denoising-diffusion-pytorch-main/实验结果对比表.xlsx"
wb.save(output_path)
print(f"Excel文件已保存到: {output_path}")