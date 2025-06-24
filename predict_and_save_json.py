import os
import json
from ultralytics import YOLO
from PIL import Image

# Model and data paths
MODEL_PATH = 'vehicle_yolov10/yolov10m2/weights/best.pt'
IMAGE_DIR = r"C:\Users\Rohit\Videos\testing\val\images\val-images-300"
OUTPUT_JSON = 'output.json'
NUM_IMAGES = 300

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

# Map YOLO class index to COCO category id (YOLO: 0-based, COCO: 1-based)
YOLO_IDX_TO_CAT_ID = {i: cat["id"] for i, cat in enumerate(CATEGORIES)}

# Get first 300 image filenames (sorted)
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])[:NUM_IMAGES]

# Load YOLO model
model = YOLO(MODEL_PATH)

# Prepare COCO output structure
coco = {
    "images": [],
    "categories": CATEGORIES,
    "annotations": []
}

annotation_id = 1
for img_id, img_name in enumerate(image_files):
    img_path = os.path.join(IMAGE_DIR, img_name)
    # Get image size
    with Image.open(img_path) as im:
        width, height = im.size
    # Add image info
    coco["images"].append({
        "file_name": img_name,
        "height": height,
        "width": width,
        "id": img_id
    })
    # Run inference
    results = model(img_path)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            area = w * h
            class_idx = int(box.cls[0].item())
            category_id = YOLO_IDX_TO_CAT_ID.get(class_idx, 15)  # fallback to 'Others'
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "bbox": [x1, y1, w, h],
                "area": area,
                "category_id": category_id
            })
            annotation_id += 1

# Save to JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco, f, indent=2)

print(f"Saved results to {OUTPUT_JSON}")
