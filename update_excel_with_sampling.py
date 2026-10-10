import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

wb = load_workbook("d:/denoising-diffusion-pytorch-main/denoising-diffusion-pytorch-main/实验结果对比表.xlsx")

header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
center_alignment = Alignment(horizontal="center", vertical="center")
thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))

if "采样步实验" in wb.sheetnames:
    wb.remove(wb["采样步实验"])
ws = wb.create_sheet(title="采样步实验")

sampling_data = [
    ["实验名称", "模型来源", "采样步", "生成时间(总秒)", "生成时间(每张秒)", "速度倍率", "图片位置"],
    ["sampling_25", "dim64模型", 25, 2.60, 0.649, 15.0, "./sampling_test_25/sample.png"],
    ["sampling_50", "dim64模型", 50, 3.81, 0.952, 10.0, "./sampling_test_50/sample.png"],
    ["sampling_100", "dim64模型", 100, 7.66, 1.914, 5.0, "./sampling_test_100/sample.png"],
    ["sampling_250", "dim64模型", 250, 19.09, 4.774, 2.0, "./sampling_test_250/sample.png"],
    ["sampling_500", "dim64模型", 500, 38.16, 9.541, 1.0, "./sampling_test_500/sample.png"],
]

for r_idx, row in enumerate(sampling_data, 1):
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

wb.save("d:/denoising-diffusion-pytorch-main/denoising-diffusion-pytorch-main/实验结果对比表.xlsx")
print("Excel文件已更新，添加了采样步实验数据！")