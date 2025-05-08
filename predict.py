import torch
import torchvision.models.segmentation as models
import cv2
import numpy as np
import os
from dataset import transform

# Load Model
model = models.deeplabv3_resnet101(pretrained=False)
model.classifier[4] = torch.nn.Conv2d(256, 27, kernel_size=1)
model.load_state_dict(torch.load("checkpoints/deeplabv3_epoch10.pth"))
model = model.cuda()
model.eval()

# Load and Predict
test_path = "IDD_Segmentation/leftImg8bit/test"
output_path = "outputs"

os.makedirs(output_path, exist_ok=True)

for img_name in os.listdir(test_path):
    img_path = os.path.join(test_path, img_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1280, 720))
    
    tensor_image = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(tensor_image)["out"]
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]

    output_img = os.path.join(output_path, img_name.replace(".png", "_pred.png"))
    cv2.imwrite(output_img, pred)

print("Predictions saved in outputs/")
