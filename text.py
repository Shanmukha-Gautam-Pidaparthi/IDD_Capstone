import os
import glob

def create_file_list(image_dir, mask_dir, output_txt):
    images = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
    masks = sorted(glob.glob(os.path.join(mask_dir, "**", "*.png"), recursive=True))

    with open(output_txt, "w") as f:
        for img, mask in zip(images, masks):
            f.write(f"{img} {mask}\n")

# Generate train_list.txt
create_file_list("IDD_Segmentation/leftImg8bit/train", "IDD_Segmentation/gtFine/train", "IDD_Segmentation/train_list.txt")
