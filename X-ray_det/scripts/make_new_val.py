import os
import random
import shutil

def copy_images_and_labels(source_image_folder, source_label_folder, destination_folder, num_samples):
    # 创建目标文件夹和子文件夹
    os.makedirs(os.path.join(destination_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, 'labels'), exist_ok=True)

    # 检测目标文件夹是否已经存在文件
    destination_image_folder = os.path.join(destination_folder, 'images')
    destination_label_folder = os.path.join(destination_folder, 'labels')

    if os.path.exists(destination_image_folder) or os.path.exists(destination_label_folder):
        # 目标文件夹已经存在文件
        user_input = input("目标文件夹已经存在文件。是否要清空原有内容并覆盖目标文件夹？(y/n): ")
        if user_input.lower() != 'y':
            print("操作已取消。")
            return

        # 删除目标文件夹中的所有文件
        shutil.rmtree(destination_image_folder)
        shutil.rmtree(destination_label_folder)

        # 重新创建目标文件夹和子文件夹
        os.makedirs(destination_image_folder)
        os.makedirs(destination_label_folder)

    # 获取图片文件夹中的所有文件名
    image_files = os.listdir(source_image_folder)

    # 随机选择指定数量的样本
    selected_files = random.sample(image_files, num_samples)

    for file_name in selected_files:
        # 复制图片文件
        source_image_path = os.path.join(source_image_folder, file_name)
        destination_image_path = os.path.join(destination_image_folder, file_name)
        shutil.copy(source_image_path, destination_image_path)

        # 复制对应的标签文件（假设标签文件和图片文件有相同的前缀）
        label_file_name = os.path.splitext(file_name)[0] + ".txt"
        source_label_path = os.path.join(source_label_folder, label_file_name)
        destination_label_path = os.path.join(destination_label_folder, label_file_name)
        shutil.copy(source_label_path, destination_label_path)

    print("复制完成！")

# 设置源图片文件夹路径、源标签文件夹路径和目标文件夹路径
source_image_folder = "/media/ders/mazhiming/mm/mmrotate/data/processed_data/train_upload/imagespng" # 图片
source_label_folder = "/media/ders/mazhiming/mm/mmrotate/data/processed_data/train_upload/dota_format" # 标签
destination_folder = "/media/ders/mazhiming/mm/mmrotate/data/few_shot" # 目标文件夹

# 指定需要复制的样本数量
num_samples = 1000

# 调用函数进行复制
copy_images_and_labels(source_image_folder, source_label_folder, destination_folder, num_samples)
