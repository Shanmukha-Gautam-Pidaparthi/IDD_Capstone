# import torch
# torch.cuda.empty_cache()

# import os

# # # dataset_path = "IDD_Segmentation/gtFine/train"
# # # total_images = sum(len(files) for _, _, files in os.walk(dataset_path))

# # # print("Total images in dataset:", total_images)




# # for images, masks in dataloader:
# #         print("loss:",total_loss)
# #         images, masks = images.to(device), masks.to(device)

# #         optimizer.zero_grad()
# #         outputs = model(images)["out"]  # Shape: (B, 19, H, W)

# #         # Ensure masks have correct shape: (B, H, W)
# #         if masks.dim() == 4:  # If mask has an extra channel dimension, remove it
# #             masks = masks.squeeze(1)

# #         # Ensure dtype is long
# #         masks = masks.long()
# #         loss = criterion(outputs, masks)
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()

# # Load checkpoint



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# # Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define model (replace with your actual model)
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.fc = nn.Linear(32 * 32, 369)  # Example model for 32x32 images

#     def forward(self, x):
#         return self.fc(x.view(x.size(0), -1))

# model = MyModel().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Load dataset and DataLoader (replace with your actual dataset)
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = datasets.FakeData(transform=transform)  # Replace with real dataset
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Load checkpoint if available
# checkpoint_path = r"C:\Users\Shanmukha Gautam\Desktop\Lost_In_Space_2.O\checkpoint_epoch1_batch170.pth"
# try:
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch']
#     start_batch = checkpoint['batch']
#     print(f"Resuming training from Epoch {start_epoch}, Batch {start_batch}")
# except FileNotFoundError:
#     print("No checkpoint found, starting fresh.")
#     start_epoch, start_batch = 0, 0

# # Training loop
# num_epochs = 50
# for epoch in range(start_epoch, num_epochs):  
#     model.train()
    
#     for batch_idx, (images, labels) in enumerate(train_loader):
#         if epoch == start_epoch and batch_idx < start_batch:
#             continue  # Skip batches before batch 17 in epoch 1

#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
        
#         # Save checkpoint every 10 batches
#         if batch_idx % 10 == 0:
#             torch.save({
#                 'epoch': epoch,
#                 'batch': batch_idx,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss
#             }, f'checkpoint_epoch{epoch}_batch{batch_idx}.pth')

#     # Reset start_batch after first resumed epoch
#     start_batch = 0


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

import torch
torch.cuda.empty_cache()

import os

pred = torch.argmax(output.squeeze(), dim=0)
print("Unique predicted classes:", np.unique(pred.numpy()))

