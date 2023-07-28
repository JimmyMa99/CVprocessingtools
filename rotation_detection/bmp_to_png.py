import os
import concurrent.futures
from PIL import Image
from tqdm import tqdm
import shutil

def convert_image(filename, source_dir, target_dir):
    if filename.endswith('.bmp'):
        img = Image.open(os.path.join(source_dir, filename))
        base = os.path.splitext(filename)[0]
        img.save(os.path.join(target_dir, base + '.png'), 'PNG')

def convert_bmp_to_png(source_dir, target_dir):

    # 创建目标文件夹和子文件夹
    os.makedirs(os.path.join(target_dir), exist_ok=True)

    # 检测目标文件夹是否已经存在文件
    if os.path.exists(target_dir):
        # 目标文件夹已经存在文件
        user_input = input("目标文件夹已经存在文件。是否要清空原有内容并覆盖目标文件夹？(y/n): ")
        if user_input.lower() != 'y':
            print("操作已取消。")
            return

        # 删除目标文件夹中的所有文件
        shutil.rmtree(target_dir)

        # 重新创建目标文件夹和子文件夹
        os.makedirs(target_dir)

    filenames = os.listdir(source_dir)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(convert_image, filenames, [source_dir]*len(filenames), [target_dir]*len(filenames)), total=len(filenames)))


source_dir = '' # 这里填写源文件夹路径
target_dir = '' # 这里填写目标文件夹路径

convert_bmp_to_png(source_dir, target_dir)
