import os
import pandas as pd
import re

# 假设日志文件的路径
log_file_path = './logs/denoise_test.log'  # 请替换为您的日志文件路径
output_file_path = './result/' + log_file_path[7:-4] + '.xlsx'  # 输出的 Excel 文件路径

# 创建一个空的列表来存储所有提取的数据
data_list = []

# 打开日志文件并逐行读取
with open(log_file_path, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 逐行解析数据
for i in range(0, len(lines), 5):  # 每4行是一组数据
    if i + 4 < len(lines):  # 确保有完整的4行
        image_match = re.search(r'Image: \s*(\S+),', lines[i + 1])
        mask_match = re.search(r'mask: \s*(\S+), ', lines[i + 1])
        lam_match = re.search(r'lambda\s*=\s*([\d.]+)', lines[i + 1])
        psnr_d_match = re.search(r'psnr_destroyed=(\S+),', lines[i + 2])
        psnr_i_match = re.search(r'psnr_inpainting=(\S+)', lines[i + 2])
        ssim_d_match = re.search(r'ssim_destroyed=(\S+),', lines[i + 3])
        ssim_i_match = re.search(r'ssim_inpainting=(\S+)', lines[i + 3])
        loss_d_match = re.search(r'loss_destroyed=(\S+),', lines[i + 4])
        loss_i_match = re.search(r'loss_inpainting=(\S+)', lines[i + 4])

        # 提取数据
        if image_match and mask_match :
            image = image_match.group(1)
            mask = mask_match.group(1)
            lam = lam_match.group(1) if lam_match else None
            psnr_destroyed = psnr_d_match.group(1) if psnr_d_match else None
            psnr_inpainting = psnr_i_match.group(1) if psnr_i_match else None
            ssim_destroyed = ssim_d_match.group(1) if ssim_d_match else None
            ssim_inpainting = ssim_i_match.group(1) if ssim_i_match else None
            loss_destroyed = loss_d_match.group(1) if loss_d_match else None
            loss_inpainting = loss_i_match.group(1) if loss_i_match else None

            # 将提取的数据添加到列表中
            data_list.append({
                'image': image,
                'mask': mask,
                'method': 'Fracial Laplace',
                'lambda': float(lam),
                'psnr': float(psnr_inpainting),
                'ssim': float(ssim_inpainting),
                'loss': float(loss_inpainting),
                'psnr destroyed' : float(psnr_destroyed),
                'ssim destroyed' : float(ssim_destroyed),
                'loss destroyed' : float(loss_destroyed)
            })

# 创建 DataFrame
df = pd.DataFrame(data_list)

# 写入 Excel 文件
df.to_excel(output_file_path, index=False)

print(f"数据已成功写入 {output_file_path}")