from ultralytics import YOLO

# Define hyperparameters and paths
DATASET_YAML = r"C:\Users\Rohit\Videos\testing\data\dataset.yaml"
MODEL = 'yolov10m.pt'
EPOCHS = 1
IMG_SIZE = 640
BATCH_SIZE = 8  # Increased batch size to utilize the A6000's memory
PROJECT = 'vehicle_yolov10'
NAME = 'yolov10m'
CHECKPOINT_INTERVAL = 1  # Save every epoch

if __name__ == '__main__':
    # Initialize the YOLO model
    model = YOLO(MODEL)

    # Train the model
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=NAME,
        device=None,  # Automatically detect and use GPU if available
        save_period=CHECKPOINT_INTERVAL,
        val=False  # Disable validation during training
    )

