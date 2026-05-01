import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
import os
import re

def read_and_sort_xlsx(file_path):
    """
    读取xlsx文件并按mask、image和lambda升序排序
    :param file_path: xlsx文件的路径
    :return: 排序后的字典列表
    """
    df = pd.read_excel(file_path)
    data_dict = df.to_dict(orient='records')
    sorted_data = sorted(data_dict, key=lambda x: (x['mask'], x['image'], x['lambda']))
    return sorted_data

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def plot_psnr(sorted_data):
    output_dir = 'C:/Users/GGBond/Desktop/test/plot_psnr/'
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

    results = []  # 用于存储结果

    for (mask, image), group in groupby(sorted_data, key=lambda x: (x['mask'], x['image'])):
        group = list(group)
        lambda_values = [item['lambda'] for item in group]
        psnr_values = [item['psnr'] for item in group]
        plt.figure(figsize=(16, 9), dpi=300)
        plt.plot(lambda_values, psnr_values, marker='o', label='PSNR', linestyle='-', markersize=2, linewidth=1)

        max_psnr = max(psnr_values)
        max_index = psnr_values.index(max_psnr)
        max_lambda = lambda_values[max_index]

        # 记录最大PSNR和对应的lambda值
        results.append({'image': image, 'mask': mask, 'max_psnr': max_psnr, 'max_lambda': max_lambda})

        plt.plot(max_lambda, max_psnr, 'r*')  # 用红色星标注最大点
        plt.text(max_lambda, max_psnr, f'({max_lambda:.2f}, {max_psnr:.2f})', fontsize=6, ha='center', va='bottom')
        plt.title(f'PSNR vs Lambda (Image: {image}, Mask: {mask})')
        plt.xlabel('Lambda')
        plt.ylabel('PSNR')
        plt.grid()

        filename = f'{sanitize_filename(image)}_{sanitize_filename(mask)}.png'
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()

    # 将结果写入Excel文件
    results_df = pd.DataFrame(results)
    results_df.to_excel('C:/Users/GGBond/Desktop/test/max_psnr_results.xlsx', index=False)

file_path = 'C:/Users/GGBond/Desktop/denoise_test.xlsx'  # 替换为你的文件路径
sorted_result = read_and_sort_xlsx(file_path)
plot_psnr(sorted_result)