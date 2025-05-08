'''To Load image for training purpose'''


# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image

# class IDDSegmentationDataset(Dataset):
#     def __init__(self, root_dir, split="train", transform=None):
#         self.root_dir = root_dir
#         self.split = split
#         self.transform = transform

#         # Paths to images and masks
#         self.image_dir = os.path.join(root_dir, "leftImg8bit", split)
#         self.mask_dir = os.path.join(root_dir, "gtFine", split)

#         # Get list of image files
#         self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".png")])
#         self.mask_filenames = sorted([f for f in os.listdir(self.mask_dir) if f.endswith("_labelIds.png")])

#         if len(self.image_filenames) == 0 or len(self.mask_filenames) == 0:
#             raise ValueError("No images or masks found in the specified dataset directories.")

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, idx):
#         img_filename = self.image_filenames[idx]
#         img_path = os.path.join(self.image_dir, img_filename)

#         # Construct the corresponding mask filename
#         mask_filename = img_filename.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
#         mask_path = os.path.join(self.mask_dir, mask_filename)

#         # Check if mask exists, otherwise raise an error
#         if not os.path.exists(mask_path):
#             raise FileNotFoundError(f"Mask not found: {mask_path}")

#         # Load Image and Mask
#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")  # Load as grayscale

#         # Resize to 1280x720 as required by the challenge
#         image = image.resize((1280, 720), Image.BILINEAR)
#         mask = mask.resize((1280, 720), Image.NEAREST)

#         if self.transform:
#             image = self.transform(image)

#         mask = torch.tensor(np.array(mask), dtype=torch.long)

#         return image, mask

# # Define Data Transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Testing
# if __name__ == "__main__":
#     dataset = IDDSegmentationDataset(root_dir="IDD_Segmentation", split="train", transform=transform)
#     print(f"Dataset Size: {len(dataset)}")
#     image, mask = dataset[0]
#     print(f"Image Shape: {image.shape}, Mask Shape: {mask.shape}")

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class IDDSegmentationDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Select a single image for training
        self.image_filename = "005506_leftImg8bit.png"  # Change this to your image name
        self.mask_filename = "005506_gtFine_labelIds.png"

        self.image_path = os.path.join(root_dir, "leftImg8bit", split, "0", self.image_filename)
        self.mask_path = os.path.join(root_dir, "gtFine", split, "0", self.mask_filename)

    def __len__(self):
        return 1  # Only one image

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")
        mask = Image.open(self.mask_path).convert("L")  # Grayscale mask

        # Resize to match training input size
        image = image.resize((1280, 720), Image.BILINEAR)
        mask = mask.resize((1280, 720), Image.NEAREST)

        # Transformations
        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

# Define data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test dataset
if __name__ == "__main__":
    dataset = IDDSegmentationDataset(root_dir="IDD_Segmentation", split="train", transform=transform)
    image, mask = dataset[0]
    print(f"Image Shape: {image.shape}, Mask Shape: {mask.shape}")
