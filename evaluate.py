'''Once the model is trained mnam validation set ki ichi evaluate chestham'''

# import torch
# import torchvision.models.segmentation as models
# from dataset import IDDSegmentationDataset, transform
# from torch.utils.data import DataLoader
# import os

# # Load Model
# model = models.deeplabv3_resnet101(pretrained=False)
# model.classifier[4] = torch.nn.Conv2d(256, 27, kernel_size=1)
# model.load_state_dict(torch.load("checkpoints/deeplabv3_epoch10.pth"))
# model = model.cuda()
# model.eval()

# # Load Dataset
# val_dataset = IDDSegmentationDataset("IDD_Segmentation", split="val", transform=transform)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# # Evaluation Loop
# correct_pixels = 0
# total_pixels = 0

# with torch.no_grad():
#     for images, masks in val_loader:
#         images, masks = images.cuda(), masks.cuda()
#         outputs = model(images)["out"]
#         preds = torch.argmax(outputs, dim=1)

#         correct_pixels += (preds == masks).sum().item()
#         total_pixels += masks.numel()

# accuracy = correct_pixels / total_pixels
# print(f"Validation Accuracy: {accuracy:.4f}")

# import torch
# import numpy as np
# import cv2
# from dataset import IDDSegmentationDataset, transform
# from train import DeepLabV3Plus  # Import trained model
# from torchvision import transforms
# from PIL import Image

# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DeepLabV3Plus(num_classes=19).to(device)
# model.load_state_dict(torch.load("models/deeplabv3plus.pth"))
# model.eval()

# # Load a single image for testing
# image_path = "IDD_Segmentation\gtFine\train\0\005506_gtFine_labelIds.png"
# image = Image.open(image_path).convert("RGB")

# # Transform image
# image = image.resize((1280, 720), Image.BILINEAR)
# image = transform(image).unsqueeze(0).to(device)

# # Inference
# with torch.no_grad():
#     output = model(image)
#     prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# # Save the predicted mask
# cv2.imwrite("sample_prediction.png", prediction)
# print("Inference completed! Prediction saved as sample_prediction.png")




















import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import IDDSegmentationDataset, transform
from torchvision.models.segmentation import deeplabv3_resnet101

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet101(pretrained=False, num_classes=19)
model.load_state_dict(torch.load("checkpoint_epoch3_batch690.pth"))
model.to(device)
model.eval()

# Load Dataset (Single Image for Evaluation)
dataset = IDDSegmentationDataset(root_dir="IDD_Segmentation", split="train", transform=transform)
image, _ = dataset[0]
image = image.unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)['out']
    predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

# Visualize Prediction
plt.imshow(predicted_mask, cmap="jet")
plt.colorbar()
plt.title("Predicted Segmentation Mask")
plt.show()

































# import torch
# import numpy as np
# from tqdm import tqdm
# from dataset import IDDSegmentationDataset, transform
# from torchvision.models.segmentation import deeplabv3_resnet101
# from sklearn.metrics import confusion_matrix

# # --- IoU Computation ---
# def compute_mIoU(gt, pred, num_classes):
#     cm = confusion_matrix(gt.flatten(), pred.flatten(), labels=list(range(num_classes)))
#     intersection = np.diag(cm)
#     union = cm.sum(1) + cm.sum(0) - intersection
#     iou = intersection / np.maximum(union, 1)
#     return np.nanmean(iou), iou

# # --- Setup ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = 27  # 0-25 are level3 classes, 26 is 'misc'

# # --- Load Model ---
# model = deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
# model.load_state_dict(torch.load("checkpoint_epoch3_batch690.pth"),strict = False)
# model.to(device)
# model.eval()

# # --- Load Validation Set ---
# val_dataset = IDDSegmentationDataset(root_dir="IDD_Segmentation", split="val", transform=transform)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# all_preds = []
# all_gts = []

# # --- Inference ---
# with torch.no_grad():
#     for images, masks in tqdm(val_loader):
#         images = images.to(device)
#         outputs = model(images)['out']
#         preds = torch.argmax(outputs, dim=1).cpu().numpy()
#         gts = masks.numpy()

#         all_preds.append(preds)
#         all_gts.append(gts)

# # --- Flatten and Compute mIoU ---
# all_preds = np.concatenate(all_preds).astype(np.uint8)
# all_gts = np.concatenate(all_gts).astype(np.uint8)

# mean_iou, class_iou = compute_mIoU(all_gts, all_preds, num_classes)

# # --- Output ---
# print(f"Validation mIoU: {mean_iou:.4f}")
# for i, iou in enumerate(class_iou):
#     print(f"Class {i} IoU: {iou:.4f}")
