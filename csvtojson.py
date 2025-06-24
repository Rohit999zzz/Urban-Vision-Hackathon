import os
import json
import pandas as pd
from tqdm import tqdm
import yaml


# Define categories with supercategory
CATEGORIES = [
    {"id": 1, "name": "Hatchback", "supercategory": "Defect"},
    {"id": 2, "name": "Sedan", "supercategory": "Defect"},
    {"id": 3, "name": "SUV", "supercategory": "Defect"},
    {"id": 4, "name": "MUV", "supercategory": "Defect"},
    {"id": 5, "name": "Bus", "supercategory": "Defect"},
    {"id": 6, "name": "Truck", "supercategory": "Defect"},
    {"id": 7, "name": "Three-wheeler", "supercategory": "Defect"},
    {"id": 8, "name": "Two-wheeler", "supercategory": "Defect"},
    {"id": 9, "name": "LCV", "supercategory": "Defect"},
    {"id": 10, "name": "Mini-bus", "supercategory": "Defect"},
    {"id": 11, "name": "Mini-truck", "supercategory": "Defect"},
    {"id": 12, "name": "tempo-traveller", "supercategory": "Defect"},
    {"id": 13, "name": "bicycle", "supercategory": "Defect"},
    {"id": 14, "name": "Van", "supercategory": "Defect"},
    {"id": 15, "name": "Others", "supercategory": "Defect"}
]
CATEGORY_NAME_TO_ID = {cat['name']: cat['id'] for cat in CATEGORIES}
CATEGORY_NAMES = [cat['name'] for cat in CATEGORIES]


def create_yolo_yaml(output_path, train_path, class_names, val_path=None):
    data = {
        'train': train_path,
        'nc': len(class_names),
        'names': class_names
    }
    if val_path:
        data['val'] = val_path
        
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"YOLO YAML config created at: {output_path}")

def process_dataframe_to_coco(df, image_list):
    coco = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    image_id_map = {}
    ann_id = 1
    img_id = 1

    for image_name in image_list:
        image_info = {
            "id": img_id,
            "file_name": image_name,
            "width": 1920,
            "height": 1080
        }
        coco['images'].append(image_info)
        image_id_map[image_name] = img_id
        img_id += 1
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row['image_name']
        cat_name = row['class_name'].strip()

        if cat_name not in CATEGORY_NAME_TO_ID:
            continue
        
        width = row['x_max'] - row['x_min']
        height = row['y_max'] - row['y_min']

        coco['annotations'].append({
            "id": ann_id,
            "image_id": image_id_map[image_name],
            "category_id": CATEGORY_NAME_TO_ID[cat_name],
            "bbox": [row['x_min'], row['y_min'], width, height],
            "area": width * height,
            "iscrowd": 0,
            "segmentation": []
        })
        ann_id += 1
        
    return coco

def csv_to_coco(csv_path, output_json):
    df = pd.read_csv(csv_path)
    images = list(df['image_name'].unique())
    coco = process_dataframe_to_coco(df, images)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"{output_json} created.")

if __name__ == "__main__":
    csv_to_coco('data/train_annotations.csv', 'data/annotations/train.json')
    create_yolo_yaml(
        output_path='data/dataset.yaml',
        train_path='../images/train',
        class_names=CATEGORY_NAMES
    )