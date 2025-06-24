import os
import json
import argparse
from ultralytics import YOLO
from PIL import Image

def create_coco_annotations(image_dir: str, output_dir: str) -> None:
    """
    Create and save COCO format annotations from object detection results.
    
    Args:
        image_dir: Directory containing the images
        output_dir: Directory to save the JSON output
    """
    # Initialize COCO format dictionary
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Load YOLO model only once
    if image_id == 0:
        MODEL_PATH = 'vehicle_yolov10/yolov10m2/weights/best.pt'
        model = YOLO(MODEL_PATH)
        # COCO categories (id starts from 1)
        CATEGORIES = [
            {"id": 1, "name": "Hatchback", "supercategory": "Defect"},
            {"id": 2, "name": "Sedan", "supercategory": "Defect"},
            {"id": 3, "name": "SUV", "supercategory": "Defect"},
            {"id": 4, "name": "MUV", "supercategory": "Defect"},
            {"id": 5, "name": "Bus", "supercategory": "Defect"},
            {"id": 6, "name": "Truck", "supercategory": "Defect"},
            {"id": 7, "name": "Three Wheeler", "supercategory": "Defect"},
            {"id": 8, "name": "Two Wheeler", "supercategory": "Defect"},
            {"id": 9, "name": "LCV", "supercategory": "Defect"},
            {"id": 10, "name": "Mini Bus", "supercategory": "Defect"},
            {"id": 11, "name": "Mini-truck", "supercategory": "Defect"},
            {"id": 12, "name": "Tempo-Traveller", "supercategory": "Defect"},
            {"id": 13, "name": "Bicycle", "supercategory": "Defect"},
            {"id": 14, "name": "Vans", "supercategory": "Defect"},
            {"id": 15, "name": "Others", "supercategory": "Defect"}
        ]
        YOLO_IDX_TO_CAT_ID = {i: cat["id"] for i, cat in enumerate(CATEGORIES)}

    # Define your categories here
    coco_format["categories"] = CATEGORIES
    
    image_id = 0
    annotation_id = 0
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": image_file,
            "width": 1920,
            "height": 1080
        }
        coco_format["images"].append(image_info)
        
        # Get image size
        with Image.open(image_path) as im:
            width, height = im.size
        # Run inference
        results = model(image_path)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                area = w * h
                class_idx = int(box.cls[0].item())
                category_id = YOLO_IDX_TO_CAT_ID.get(class_idx, 15)  # fallback to 'Others'
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1
        
        image_id += 1
    
    # Save annotations to JSON file
    output_file = os.path.join(output_dir, 'output.json')
    try:
        with open(output_file, 'w') as f:
            json.dump(coco_format, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")


# NOTE: Please do not change the code below!
def main():
    parser = argparse.ArgumentParser(description='Create COCO format annotations from object detection results')
    parser.add_argument('--image_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save the COCO format JSON file')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate and save COCO format annotations
    create_coco_annotations(args.image_dir, args.output_dir)

if __name__ == "__main__":
    main() 