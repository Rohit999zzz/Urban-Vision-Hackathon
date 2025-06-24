import os
import json
from collections import defaultdict

def coco_to_yolo(coco_json_path, labels_dir):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Map image_id to file_name and image size
    image_id_to_info = {img['id']: img for img in coco['images']}
    # Map category_id to 0-based index for YOLO
    categories = sorted(coco['categories'], key=lambda x: x['id'])
    cat_id_to_yolo = {cat['id']: i for i, cat in enumerate(categories)}

    # Prepare output directory
    os.makedirs(labels_dir, exist_ok=True)

    # Group annotations by image
    image_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        image_to_anns[ann['image_id']].append(ann)

    for image_id, image_info in image_id_to_info.items():
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        label_lines = []
        for ann in image_to_anns.get(image_id, []):
            cat_id = ann['category_id']
            yolo_cat = cat_id_to_yolo[cat_id]
            x, y, w, h = ann['bbox']
            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            label_lines.append(f"{yolo_cat} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Sanitize filename to be valid on Windows by replacing colons
        sanitized_file_name = file_name.replace(':', '_')

        # Write label file
        txt_name = os.path.splitext(sanitized_file_name)[0] + '.txt'
        with open(os.path.join(labels_dir, txt_name), 'w') as f:
            f.write('\n'.join(label_lines))

    print(f"YOLO labels written to {labels_dir}")

if __name__ == "__main__":
    # For train set:
    coco_to_yolo(
        coco_json_path='data/annotations/train.json',
        labels_dir='data/labels/train'
    ) 