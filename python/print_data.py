import pandas as pd

def read_excel_to_dict(file_path, sheet_name, columns):
    """
    从Excel文件的指定工作表中读取多列并返回为字典。
    :param file_path: Excel文件的路径
    :param sheet_name: 工作表名称
    :param columns: 要读取的列名列表
    :return: 字典形式的数据
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # 将指定列的数据转换为字典
        data_dict = df[columns].to_dict(orient='records')
        return data_dict
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None

def sort_dict_list(data_dict_list, sort_keys):
    """
    根据指定的键对字典列表进行排序。
    :param data_dict_list: 字典列表
    :param sort_keys: 排序的键列表
    :return: 排序后的字典列表
    """
    return sorted(data_dict_list, key=lambda x: tuple(x[k] for k in sort_keys))

def print_list_in_chunks(data_list):
    """
    每chunk_size行打印一次列表中的数据。
    :param data_list: 要打印的列表
    :param chunk_size: 每次打印的行数
    """
    for i in range(0, len(data_list)):
        chunk = data_list[i]
        line = []
        if i%2==0:
            for key in columns[4:]:
                line.append(f'{chunk[key]:.2f}')
        else:
            for key in columns[4:]:
                line.append(f'{chunk[key]:.4f}')
        print(chunk['image'],chunk['mask'],chunk['quality_assess'],'\n')
        print(' & '.join(line))
        print()  # 添加换行以分隔块

# 使用示例
file_path = 'C:/Users/GGBond/Desktop/new_result/quality_assess.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'          # 替换为你的工作表名称
columns = ['image',	'mask',	'quality_assess', 'masked',	'TV', 'TGV', 'ftTV', 'AMSI', 'GLCIC', 'R.I-MEDFE', 'AOT_GAN', 'FcF', 'FLP']  # 包括'denoise'列
data_dict = read_excel_to_dict(file_path, sheet_name, columns)

if data_dict is not None:
    sorted_data = sort_dict_list(data_dict, ['image', 'mask', 'quality_assess'])  # 根据'sigma', 'mask', 'quality_assess'列排序
    print_list_in_chunks(sorted_data)