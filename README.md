# Urban Vision Hackathon

Yeh project ek computer vision based solution hai jo urban environments mein vehicle aur object detection ke liye bana hai.

## Project Structure

- `main.py` : Main entry point
- `train.py` : Model training script
- `predict_and_save_json.py` : Prediction aur JSON output
- `coco2yolo.py` : COCO se YOLO format conversion
- `csvtojson.py` : CSV se JSON conversion
- `process_directory.py` : Directory processing tools
- `user_annotation_analysis.py` : User annotation analysis
- `data/` : Data, images, labels, configs
- `vehicle_yolov10/` : YOLOv10 models aur weights
- `slurm/` : Slurm job scripts

## Requirements

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Training:
```
python train.py
```

Prediction:
```
python predict_and_save_json.py
```

## License

MIT 