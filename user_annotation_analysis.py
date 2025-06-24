import pandas as pd

def export_high_agreement_to_coco(annotations_df, images_df, output_json, agreement_matrix, user_ids, agreement_threshold=0.8):
    """
    Export only high-agreement annotations to COCO format for training.
    Args:
        annotations_df (DataFrame): All user annotations
        images_df (DataFrame): Image metadata
        output_json (str): Path to output COCO JSON
        agreement_matrix (DataFrame): Matrix of user agreement per bbox
        user_ids (list): List of user IDs in the matrix
        agreement_threshold (float): Minimum agreement rate to keep a bbox
    """
    import json, os
    # Find bbox columns with agreement above threshold
    high_agreement_bboxes = []
    for col in agreement_matrix.columns:
        non_missing = agreement_matrix[col][agreement_matrix[col] != -1]
        if len(non_missing) > 1:
            agreement = len(set(non_missing)) == 1
            if agreement:
                high_agreement_bboxes.append(col)
    # Filter annotations_df to only those bbox_ids
    filtered_annots = []
    for bbox_col in high_agreement_bboxes:
        bbox_id = bbox_col.split('=')[1]
        # Find all rows in annotations_df with this baseline_annotation_id or new id
        if bbox_id.startswith('new_'):
            record_id = int(bbox_id.replace('new_', ''))
            filtered_annots.append(annotations_df[annotations_df['id'] == record_id])
        else:
            filtered_annots.append(annotations_df[annotations_df['baseline_annotation_id'] == int(bbox_id)])
    if filtered_annots:
        filtered_df = pd.concat(filtered_annots)
    else:
        filtered_df = pd.DataFrame(columns=annotations_df.columns)
    # Use your previous COCO export logic here
    images = list(filtered_df['image_name'].unique())
    def process_dataframe_to_coco(df, image_list):
        # (You can copy your previous process_dataframe_to_coco function here)
        coco = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i+1, "name": name} for i, name in enumerate([
                    "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", 
                    "Two-wheeler", "LCV", "Mini-bus", "Mini-truck", "tempo-traveller", 
                    "bicycle", "people", "white-swift-dzire"
                ])
            ]
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
        for _, row in df.iterrows():
            image_name = row['image_name']
            cat_id = row['user_submitted_category_id']
            if pd.isna(cat_id):
                continue
            x = row['x']
            y = row['y']
            width = row['width']
            height = row['height']
            coco['annotations'].append({
                "id": ann_id,
                "image_id": image_id_map[image_name],
                "category_id": int(cat_id),
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1
        return coco
    coco = process_dataframe_to_coco(filtered_df, images)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"High-agreement COCO JSON created at {output_json}")

# Example usage after your analysis:
if __name__ == "__main__":
    # ... after your analysis code ...
    # Suppose you have run process_image_annotations for a specific image_id:
    # matrix, user_ids = process_image_annotations(image_id, data, logger)
    # To export high-agreement bboxes for this image:
    # export_high_agreement_to_coco(data['annotations_df'], data['images_df'], 'output/high_agreement.json', matrix, user_ids, agreement_threshold=0.8)
    pass