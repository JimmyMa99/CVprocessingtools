import os
import glob

# 设定目录
image_dir = "/media/ders/mazhiming/mm/mmrotate/data/data/train_upload/images"
label_path = "/media/ders/mazhiming/mm/mmrotate/data/data/train_upload/labelTxt_new"
new_label_path = "/media/ders/mazhiming/mm/mmrotate/data/data/dota_format"


if not os.path.exists(new_label_path):
    os.makedirs(new_label_path)

for file in os.listdir(label_path):
    if file.endswith('.txt'):
        with open(os.path.join(label_path, file), 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            coords, category, difficult = line.rsplit(',', 2)
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, coords.split(','))

            # 将类别名称中的空格和'-'替换为下划线
            category = category.strip()
            category = category.replace(' ', '_')

            difficult = difficult.strip()

            # 根据你的数据来调整坐标的顺序
            new_coords = ' '.join(map(str, [x1, y1, x2, y2, x3, y3, x4, y4]))
            new_line = new_coords + ' ' + category + ' ' + difficult

            new_lines.append(new_line)

        with open(os.path.join(new_label_path, file), 'w') as f:
            f.write('\n'.join(new_lines))
