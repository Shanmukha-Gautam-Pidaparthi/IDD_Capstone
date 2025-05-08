import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet101

class IDDSegmentationDataset(Dataset):
    def __init__(self, txt_file, transform=None, img_size=(256, 128)):
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

        # Set ignore index (255) to -1
        # mask[mask == 255] = 255  # Keep 255 as ignore index
        mask[mask > 18] = 255 

        # Clip values to be within 0-18 (avoid unexpected values)
        mask = np.clip(mask, 0, 18)

        # Convert to tensor (important: dtype must be long for CrossEntropyLoss)
        mask = torch.tensor(mask, dtype=torch.long)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, mask

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for images
transform = T.Compose([
    T.ToTensor(),  # Convert to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Path to the text file containing image-mask pairs
txt_file = r"C:\Users\Shanmukha Gautam\Desktop\Lost_In_Space_2.O\IDD_Segmentation\train_list.txt"

# Create dataset and dataloader
dataset = IDDSegmentationDataset(txt_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

# Load the pretrained model
model = deeplabv3_resnet101(weights="DEFAULT")

# Modify the classifier for IDD dataset (19 classes: 18 foreground + 1 background)
model.classifier[4] = nn.Conv2d(256, 19, kernel_size=(1, 1), stride=(1, 1))
model = model.to(device)


# Load the checkpoint
checkpoint_path ="checkpoint_epoch8_batch320.pth"  # Path to your checkpoint file
checkpoint = torch.load(checkpoint_path, map_location=device)

# Extract epoch and batch index
start_epoch = checkpoint['epoch']
start_batch = checkpoint['batch_idx'] + 1  # Resume from next batch

# Load model and optimizer state
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer = optim.Adam(model.parameters(), lr=0.1e-5)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Resuming training from Epoch {start_epoch+1}, Batch {start_batch}")

# Loss function
criterion = nn.CrossEntropyLoss(ignore_index=255)

print("Begin Training")

# Training loop (for 3 epochs)
for epoch in range(start_epoch, 20):
    print(f"Beginning epoch {epoch+1}")
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        # Skip batches that were already processed in the current epoch
        if epoch == start_epoch and batch_idx < start_batch:
            continue
            
        batch_count += 1
        images, masks = images.to(device), masks.to(device)

        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)["out"]  # Shape: (B, 19, H, W)

        # Ensure masks have correct shape: (B, H, W)
        if masks.dim() == 4:  # If mask has an extra channel dimension, remove it
            masks = masks.squeeze(1)

        masks = masks.long()  # Ensure dtype is long
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        
        # Print current batch status
        print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Save a checkpoint every 10 batches
        if batch_idx % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, f'checkpoint_epoch{epoch+1}_batch{batch_idx}.pth')
            print(f"Checkpoint saved at Epoch {epoch+1}, Batch {batch_idx}")
    
    # Print epoch summary
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1} completed: Average Loss = {avg_loss:.4f}")

print("Training complete! ✅")

# Optional: Save final model
torch.save(model.state_dict(), 'final_model.pth')
print("Final model saved!")