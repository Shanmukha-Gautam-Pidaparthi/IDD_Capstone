import torch
import torchvision.transforms as T
from PIL import Image
import os
import torchvision.models.segmentation as models
import matplotlib.pyplot as plt

# -------- MODEL DEFINITION --------
def deeplabv3_resnet101(num_classes=19):
    model = models.deeplabv3_resnet101(weights=None, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model

# -------- CONFIG --------
IMAGE_DIR = "test_images"
OUTPUT_DIR = "predicted_masks"
COLORED_DIR = "predicted_masks_colored"
CHECKPOINT_PATH = "checkpoint_epoch7_batch540.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (128, 256)  # (H, W)

# -------- TRANSFORM --------
transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=Image.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# -------- LOAD MODEL --------
model = deeplabv3_resnet101(num_classes=19).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

for k in list(state_dict.keys()):
    if 'aux_classifier.4' in k:
        del state_dict[k]

model.load_state_dict(state_dict, strict=False)
model.eval()

# -------- PREDICT & SAVE --------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COLORED_DIR, exist_ok=True)

image_names = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png") or f.endswith(".jpg")]
cmap = plt.get_cmap('tab20', 19)  # 19 classes

with torch.no_grad():
    for img_name in image_names:
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        output = model(input_tensor)['out']  # [1, 19, H, W]
        pred = torch.argmax(output.squeeze(), dim=0).byte().cpu()  # [H, W]

        # Save grayscale mask
        out_path = os.path.join(OUTPUT_DIR, img_name.replace('.jpg', '.png'))
        Image.fromarray(pred.numpy()).save(out_path)
        print(f"Saved grayscale mask: {out_path}")

        # Save color mask
        plt.imshow(pred, cmap=cmap, vmin=0, vmax=18)
        plt.axis('off')
        plt.savefig(os.path.join(COLORED_DIR, img_name.replace('.jpg', '.png')), bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved color mask: {os.path.join(COLORED_DIR, img_name.replace('.jpg', '.png'))}")



# from PIL import Image
# import numpy as np

# pred = np.array(Image.open("predicted_masks/frame2410_leftImg8bit.png"))
# print("Unique class values in prediction:", np.unique(pred))
