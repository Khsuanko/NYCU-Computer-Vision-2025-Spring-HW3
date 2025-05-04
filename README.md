# NYCU-Computer-Vision-2025-Spring-HW3
StudentID: 110550122  
Name: 柯凱軒

## Introduction
The task is instance segmentation on cell microscopy images, where the goal is to detect and segment individual cells across four predefined classes (class1–class4). The final predictions need to include bounding boxes, segmentation masks, class labels, and confidence scores, formatted for evaluation on a hidden test set.  
Our approach is based on a Mask R-CNN architecture, which has proven effective for simultaneous object detection and segmentation tasks. The core idea is to leverage a pre-trained backbone (ResNet-50) combined with a Feature Pyramid Network (FPN) to handle multi-scale feature representation, followed by dedicated heads for predicting masks, bounding boxes, and class labels.



## How to install
1. Install Dependencies  
```python
pip install torch torchvision torchaudio numpy opencv-python scikit-image tqdm matplotlib pycocotools
```
2. Ensure you have the dataset structured as follows:
```python
./data/
    ├── train/
    ├── test/
    ├── test_image_name_to_ids.json
```
3. Run the code
```python
python train.py
python test.py
```
## Performance snapshot
![performance](https://github.com/Khsuanko/NYCU-Computer-Vision-2025-Spring-HW3/blob/main/performance.png)
