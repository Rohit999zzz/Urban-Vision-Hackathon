import pandas as pd
from sklearn.model_selection import train_test_split
import os

def process_data():
    # Define paths relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    # === Step 1: Load CSVs ===
    print("Loading data...")
    annotations = pd.read_csv(os.path.join(data_dir, "phase_2_user_annotation.csv"))
    user_image = pd.read_csv(os.path.join(data_dir, "phase_2_user_image_user_annotation.csv"))
    image_meta = pd.read_csv(os.path.join(data_dir, "phase_2_image.csv")).rename(columns={"id": "image_id"})
    user_scores = pd.read_csv(os.path.join(data_dir, "phase_2_user_progression_score.csv"))

    # === Step 2: Initial annotation cleaning ===
    print("Cleaning annotations...")
    clean_annotations = annotations[
        (annotations["is_confirmed"] == True) &
        (annotations["is_deleted"] == False) &
        (annotations["is_label_changed"] == False)
    ]

    # === Step 2.5: Filter users by progression score ===
    print("Filtering users by score...")
    SCORE_THRESHOLD = 0.8
    latest_scores = user_scores.sort_values("created_at", ascending=False).drop_duplicates("user_id")
    users_to_keep = latest_scores[latest_scores["ax_percentage_score"] >= SCORE_THRESHOLD]["user_id"]
    annotations_before = len(clean_annotations)
    clean_annotations = clean_annotations[clean_annotations["user_id"].isin(users_to_keep)]
    annotations_after = len(clean_annotations)

    print(f"Kept {len(users_to_keep.unique())} high-quality users.")
    print(f"Annotations before: {annotations_before}, after: {annotations_after}")

    # === Step 3: Filter for submitted images only ===
    print("Filtering for submitted images...")
    submitted_pairs = user_image[user_image["is_submitted"] == True][["user_id", "image_id"]]
    clean_annotations = clean_annotations.merge(submitted_pairs, on=["user_id", "image_id"], how="inner")

    # === Step 4: Merge with image metadata ===
    print("Merging with image metadata...")
    image_meta_renamed = image_meta.rename(columns={"image_name": "original_image_name"})
    merged_df = clean_annotations.merge(
        image_meta_renamed[["image_id", "original_image_name", "height", "width"]],
        on="image_id",
        how="left"
    )

    # === Step 5: Convert bounding boxes ===
    print("Converting bounding boxes...")
    merged_df["x_min"] = merged_df["x"]
    merged_df["y_min"] = merged_df["y"]
    merged_df["x_max"] = merged_df["x"] + merged_df["width_x"]
    merged_df["y_max"] = merged_df["y"] + merged_df["height_x"]

    # === Step 6: Map category IDs ===
    print("Mapping class names...")
    category_map = {
        1: "Hatchback", 2: "Sedan", 3: "SUV", 4: "MUV", 5: "Bus",
        6: "Truck", 7: "Three-wheeler", 8: "Two-wheeler", 9: "LCV",
        10: "Mini-bus", 11: "Mini-truck", 12: "tempo-traveller",
        13: "bicycle", 14: "Van", 15: "Others"
    }
    merged_df["class_name"] = merged_df["baseline_category_id"].map(category_map)

    # === Step 7: Final clean format ===
    print("Creating final dataframe...")
    final_df = merged_df[[
        "original_image_name", "x_min", "y_min", "x_max", "y_max",
        "baseline_category_id", "class_name"
    ]].rename(columns={"original_image_name": "image_name"})

    # === Step 8: Save to CSV ===
    print("Saving annotations for training...")
    final_df.to_csv(os.path.join(data_dir, "train_annotations.csv"), index=False)

    print("✅ Done! Training annotations saved.")

if __name__ == "__main__":
    process_data()