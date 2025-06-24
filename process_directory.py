def process_directory(model_path, input_dir, output_dir, conf_threshold=0.5):
    """
    Process all images in a directory and save JSON results

    Args:
        model_path: Path to the trained YOLOv8 model
        input_dir: Directory containing images to process
        output_dir: Where to save JSON results
        conf_threshold: Confidence threshold for predictions
    """
    from pathlib import Path
    import os
    from predict_and_save_json import predict_and_save_json

    # Check if input directory exists
    if not Path(input_dir).is_dir():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [f for f in Path(input_dir).iterdir() if f.suffix.lower() in image_extensions]

    # Process each image
    for img_path in image_paths:
        output_json_path = Path(output_dir) / f"{img_path.stem}_predictions.json"
        predict_and_save_json(model_path, str(img_path), str(output_json_path), conf_threshold)

# ...existing code...

if __name__ == "__main__":
    model_path = "best.pt"  # Update with your model path
    input_dir = "input_images"  # Update with your images folder
    output_dir = "output_jsons"  # Update with your desired output folder
    conf_threshold = 0.5  # Adjust if needed

    process_directory(model_path, input_dir, output_dir, conf_threshold)
