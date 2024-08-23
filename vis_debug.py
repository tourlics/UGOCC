import os
import numpy as np


def load_and_print_npy_files(directory):
    """
    从指定目录中读取所有 .npy 文件，并打印每个文件的名称和大小。

    参数:
    directory (str): 包含 .npy 文件的目录路径。

    返回:
    dict: 包含每个文件内容的字典，键为文件名（不含扩展名），值为对应的 numpy 数组。
    """
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    npy_data = {}

    for file_name in npy_files:
        file_path = os.path.join(directory, file_name)
        data = np.load(file_path)
        file_size = os.path.getsize(file_path)
        print(f'File: {file_name}, Size: {file_size / 1024:.2f} KB, Shape: {data.shape}')

        # 去掉文件扩展名作为字典键
        key = os.path.splitext(file_name)[0]
        npy_data[key] = data

    return npy_data


# 示例目录
directory = 'outputs_debug/6401/'

# 调用函数
npy_data = load_and_print_npy_files(directory)