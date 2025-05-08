import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet101

class IDDSegmentationDataset(Dataset):
    def __init__(self, txt_file, transform=None, img_size=(512, 1024)):
        self.transform = transform
        self.img_size = img_size

        # Read image-mask paths from text file
        with open(txt_file, "r") as f:
            self.data_pairs = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.data_pairs[idx]

        # Load Image
        image = cv2.imread(img_path)  # Read image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, self.img_size)  # Resize

        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Load as grayscale
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)  # Resize mask
        mask = torch.tensor(mask, dtype=torch.long)  # Convert to tensor

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, mask

# Define transformations for images
transform = T.Compose([
    T.ToTensor(),  # Convert to tensor
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
])

# Path to the text file containing image-mask pairs
train_txt = r"C:\Users\Shanmukha Gautam\Desktop\Lost_In_Space_2.O\IDD_Segmentation\train_list.txt"
val_txt = r"C:\Users\Shanmukha Gautam\Desktop\Lost_In_Space_2.O\IDD_Segmentation\val_list.txt"

# Load datasets
train_dataset = IDDSegmentationDataset(train_txt, transform=transform)
val_dataset = IDDSegmentationDataset(val_txt, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet101(pretrained=True, num_classes=19).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train_one_epoch(epoch):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Training Loss = {total_loss / len(train_loader)}")

# Validation function
def validate():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(val_loader)}")

# Train and Validate
num_epochs = 10
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    train_one_epoch(epoch)
    validate()

    # Save model checkpoint
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

print("Full training complete! ✅")
