import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# Define label mapping (modify based on your dataset)
LABEL_MAPPING = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
    "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8, "terrain": 9,
    "sky": 10, "person": 11, "rider": 12, "car": 13, "truck": 14, "bus": 15,
    "train": 16, "motorcycle": 17, "bicycle": 18, "void": 255  # Ignore class
}

def convert_json_to_png(json_path, output_path):
    """Convert a JSON annotation file to a segmentation mask PNG."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_width = data['imgWidth']
    img_height = data['imgHeight']
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for obj in data['objects']:
        label = obj['label']
        label_id = LABEL_MAPPING.get(label, LABEL_MAPPING["void"])
        polygon = np.array(obj['polygon'], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], int(label_id))

    cv2.imwrite(output_path, mask)

def process_val_dataset(dataset_root):
    """Process only JSON files in folders inside the val directory."""
    val_root = os.path.join(dataset_root, "gtFine", "val")

    if not os.path.exists(val_root):
        print(f"Error: {val_root} does not exist!")
        return

    for city in os.listdir(val_root):
        city_path = os.path.join(val_root, city)
        if not os.path.isdir(city_path):
            continue  

        json_files = [f for f in os.listdir(city_path) if f.endswith("_polygons.json")]

        if not json_files:
            print(f"No JSON files found in {city_path}, skipping...")
            continue

        for json_file in tqdm(json_files, desc=f"Processing val/{city}"):
            json_path = os.path.join(city_path, json_file)
            png_filename = json_file.replace("_polygons.json", "_gtFine_labelIds.png")
            png_path = os.path.join(city_path, png_filename)

            if os.path.exists(png_path):
                print(f"Skipping {png_path}, already exists.")
                continue

            convert_json_to_png(json_path, png_path)

if __name__ == "__main__":
    dataset_root = "IDD_Segmentation"  # Change this to your dataset path
    process_val_dataset(dataset_root)  # Process only the 'val' split
