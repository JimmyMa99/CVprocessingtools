import json

def merge_coco_datasets(dataset1_path, dataset2_path, output_path):
    # 读取第一个数据集的JSON文件
    with open(dataset1_path, 'r') as f:
        dataset1 = json.load(f)

    # 读取第二个数据集的JSON文件
    with open(dataset2_path, 'r') as f:
        dataset2 = json.load(f)

    # 获取第一个数据集中最大的ID
    max_image_id = max(image['id'] for image in dataset1['images'])
    max_ann_id = max(annotation['id'] for annotation in dataset1['annotations'])

    # 将第二个数据集中的ID逐一增加最大ID的值
    for image in dataset2['images']:
        image['id'] += max_image_id

    for annotation in dataset2['annotations']:
        annotation['image_id'] += max_image_id
        annotation['id'] += max_ann_id

    # 合并两个数据集
    # breakpoint()
    merged_dataset = {
        'type': dataset1['type'],
        'categories': dataset1['categories'],
        'images': dataset1['images'] + dataset2['images'],
        'annotations': dataset1['annotations'] + dataset2['annotations']
    }

    # 写入合并后的JSON文件
    with open(output_path, 'w') as f:
        json.dump(merged_dataset, f)

# 示例用法
dataset1_path = '/root/autodl-tmp/mmyolo/data/kdxf/output.json'
dataset2_path = '/root/autodl-tmp/mmyolo/data/kdxf_aug/augmented_dataset.json'
output_path = 'merged_dataset.json'

merge_coco_datasets(dataset1_path, dataset2_path, output_path)
