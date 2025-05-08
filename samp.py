import os

tar_path = r"C:\Users\Shanmukha Gautam\.paddleseg\pretrained_model\resnet101_vd_ssld_v2\resnet101_vd_ssld_v2.tar.gz"

if os.path.exists(tar_path):
    print("File exists:", tar_path)
    print("File size:", os.path.getsize(tar_path), "bytes")
else:
    print("File does not exist.")
