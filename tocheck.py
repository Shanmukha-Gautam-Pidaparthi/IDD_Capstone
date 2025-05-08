import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to a sample segmentation mask
mask_path = "IDD_Segmentation/gtFine/train/0/005506_gtFine_labelIds.png"

# Load the mask as a grayscale image (preserves class IDs)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# Check unique values in the mask (class IDs)
unique_classes = np.unique(mask)
print("Unique Class IDs in Mask:", unique_classes)

# Display the mask using a color map
plt.figure(figsize=(8, 6))
plt.imshow(mask, cmap="jet")  # Use 'jet' colormap for better visualization
plt.colorbar(label="Class Labels")  # Adds a legend for class IDs
plt.title("Segmentation Mask Visualization")
plt.show()
