**Overview**

Road perception models are typically trained on structured environments found in Europe or North America.
The Indian Driving Dataset (IDD) presents a more complex and realistic challenge due to its unstructured traffic, mixed road usage, inconsistent lane markings, and varying lighting and weather conditions.

This project explores semantic segmentation using deep learning to classify every pixel in a road scene into predefined classes such as road, vehicles, pedestrians, and background.
The repository contains the full pipeline from data preparation to model training, evaluation, and inference.

**Objectives**

- Convert IDD annotations into mask images suitable for training
- Train a segmentation model capable of understanding Indian road conditions
- Resume partially completed training runs using checkpoints
- Evaluate model performance on the test dataset
- Generate predicted segmentation masks for unseen images


**Repository Structure**

- predicted_masks/                Output grayscale masks from inference
- predicted_masks_colored/        Output color-annotated masks for visualization
- convert_json_to_png.py          Converts JSON annotations into pixel masks
- createLabels.py                 Processes and formats annotation classes
- dataset.py                      Custom PyTorch dataloader
- train.py                        Main training pipeline
- resume_train.py                 Resume training from saved checkpoints
- evaluate.py                     Computes key performance metrics
- predict_save_test.py            Runs inference on test images


**Dataset**

- Name: Indian Driving Dataset (IDD)
- Task: Semantic segmentation
- Description: A collection of images from Indian roads featuring heterogeneous traffic patterns and diverse road structures
- URL: https://idd.insaan.iiit.ac.in/
- This dataset includes multiple locations, weather conditions, and urban layouts, making it suitable for real-world perception challenges.

**Technology Stack**

- Python
- PyTorch and Torchvision
- OpenCV
- NumPy
- Matplotlib (for visualization)
- Model architecture (deeplabv3_resnet101, DeepLab).

**Running the Project**

1. Install dependencies
pip install -r requirements.txt

2. Convert annotation JSON files into segmentation mask images
python convert_json_to_png.py

3. Train the segmentation model
python train.py

4. Resume training from a checkpoint if required
python resume_train.py

5. Generate predictions and save output masks
python predict_save_test.py


Prediction results are saved automatically to:
- predicted_masks/
- predicted_masks_colored/


**Key Learning Outcomes**

1. Through this project the following concepts were implemented and reinforced:
2. Handling real-world annotated datasets
3. Creating data processing pipelines from raw label formats
4. Training, monitoring, and resuming deep learning models with the help of checkpoints.
5. Evaluating pixel-level classification models on challenging data
6. Understanding the practical complexity of computer vision on non-ideal road environments.
