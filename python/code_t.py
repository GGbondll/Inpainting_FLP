from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
import os


# PDF 文件路径
pdf_path = 'C:/Users/GGBond/Desktop/24color.pdf'

# 转换为 PNG
def pdf2image(pdf_path):
    images = convert_from_path(pdf_path)

    # 保存为 PNG
    for i, image in enumerate(images):
        image.save(f'output_{i + 1}.png', 'PNG')

image_names = ['鹦鹉', '猫', '海滩', '花', '山', '芦苇']
mask_names = ['mark', 'random', 'scratch', 'waterprint']
denoise_name = []
for mask_name in mask_names:
    denoise_name.append('denoise_'+mask_name)
for mask_name in mask_names:
    for n_i in ['n1', 'n2','n3']:
        denoise_name.append(n_i+'_'+mask_name)
names = []
for image_name in image_names:
    for mask_name in mask_names:
        names.append(image_name+'_'+mask_name)
names += denoise_name
def split_pdf(input_pdf, output_folder):
    # 读取 PDF 文件
    reader = PdfReader(input_pdf)

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历 PDF 的每一页，并保存为单独的 PDF 文件
    for page_num in range(len(reader.pages)):
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num])

        # 生成输出 PDF 文件的路径
        output_pdf = os.path.join(output_folder, f"{names[page_num]}.pdf")

        # 将每一页写入新的 PDF 文件
        with open(output_pdf, 'wb') as output_file:
            writer.write(output_file)

        print(f"保存了 {output_pdf}")  # 输出已保存文件的路径

# 示例用法
input_pdf = "C:/Users/GGBond/Desktop/test_cn.pdf"  # 输入 PDF 文件路径
output_folder = "C:/Users/GGBond/Desktop/论文材料/论文LaTex/pdf_result"  # 输出文件夹路径

split_pdf(input_pdf, output_folder)
