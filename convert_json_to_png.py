import json
import numpy as np
import cv2
import os

# Set paths (CHANGE THIS TO YOUR JSON FILE)
json_file = r"C:\Users\Shanmukha Gautam\Desktop\Lost_In_Space_2.O\IDD_Segmentation\gtFine\train\0\005506_gtFine_polygons.json"
output_file = json_file.replace("_polygons.json", "_labelIds.png")

# Load JSON
with open(json_file, "r") as f:
    data = json.load(f)

# Create an empty mask (assuming resolution 1280x720)
mask = np.zeros((720, 1280), dtype=np.uint8)

# Fill mask with label IDs (may need mapping)
for obj in data["objects"]:
    
    # Map class names to level 3 IDs (Modify according to dataset)
    label_mapping = {
        "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
        "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8, 
        "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13, 
        "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
    }

    label_name = obj["label"]  # Get label name
    label_id = label_mapping.get(label_name, 26)  # Default to 26 (miscellaneous)

    polygon = np.array(obj["polygon"], np.int32)
    cv2.fillPoly(mask, [polygon], int(label_id))  # Ensure label_id is an integer

      # This may need adjustment
    polygon = np.array(obj["polygon"], np.int32)
    cv2.fillPoly(mask, [polygon], label_id)

# Save as PNG
cv2.imwrite(output_file, mask)

print(f"✅ Saved mask: {output_file}")
