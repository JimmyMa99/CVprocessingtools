import os
import shutil
import random
import json
from PIL import Image

def flip_image(image_path, output_path):
    with Image.open(image_path) as img:
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_img.save(output_path)

def flip_annotations(annotations, width):
    for annotation in annotations['annotations']:
        bbox = annotation['bbox']
        x_min = width - bbox[0] - bbox[2]
        annotation['bbox'][0] = x_min

def augment_dataset(image_dir, annotation_file, output_dir):
    class_counts = {
        'glassbottle': 3069,
        'knife': 722,
        'laptop': 366,
        'lighter': 429,
        'metalcup': 251,
        'presure': 240,
        'scissor': 441,
        'tongs': 361,
        'umbrella': 1265
    }

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    augmented_images = []
    augmented_annotations = []

    for class_name, count in class_counts.items():
        # Randomly select existing samples to augment
        samples_to_augment = random.sample(range(len(annotations['images'])), count)

        for idx in samples_to_augment:
            image_info = annotations['images'][idx]
            image_file = image_info['file_name']
            image_path = os.path.join(image_dir, image_file)

            output_image_file = f"{os.path.splitext(image_file)[0]}_aug.jpg"
            output_image_path = os.path.join(output_dir, output_image_file)

            flip_image(image_path, output_image_path)
            flip_annotations(annotations, width=image_info['width'])

            # Update the annotations for the augmented image
            augmented_image_info = {
                'id': len(augmented_images) + 1,
                'file_name': output_image_file,
                'width': image_info['width'],
                'height': image_info['height']
            }
            augmented_images.append(augmented_image_info)

            for annotation in annotations['annotations']:
                if annotation['image_id'] == image_info['id']:
                    augmented_annotation = annotation.copy()
                    augmented_annotation['id'] = len(augmented_annotations) + 1
                    augmented_annotation['image_id'] = augmented_image_info['id']
                    augmented_annotations.append(augmented_annotation)

    # Save the augmented annotations to a JSON file
    augmented_data = {
        'images': augmented_images,
        'annotations': augmented_annotations,
        'categories': annotations['categories']
    }

    output_annotation_file = 'augmented_dataset.json'
    output_annotation_path = os.path.join(output_dir, output_annotation_file)

    with open(output_annotation_path, 'w') as f:
        json.dump(augmented_data, f)

    print("Data augmentation completed!")


# 脚本输入
data_image_dir = 'data/kdxf'
data_annotation_dir = 'data/kdxf/output.json'
output_dir = 'data/kdxf_aug'

# 数据增强
augment_dataset(data_image_dir, data_annotation_dir, output_dir)
