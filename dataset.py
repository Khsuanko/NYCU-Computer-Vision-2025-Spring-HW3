import torch
from torch.utils.data import Dataset
import os
import numpy as np
import skimage.io as sio
import cv2
from pathlib import Path
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
import random

class CellSegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = Path(root_dir)
        self.image_folders = sorted(os.listdir(self.root_dir))
        self.transforms = transforms

    def __getitem__(self, idx):
        folder = self.image_folders[idx]
        folder_path = self.root_dir / folder
        image = cv2.imread(str(folder_path / 'image.tif'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = []
        labels = []

        for class_idx in range(1, 5):
            mask_path = folder_path / f'class{class_idx}.tif'
            if not mask_path.exists():
                continue
            mask = sio.imread(str(mask_path))
            instances = np.unique(mask)
            instances = instances[instances != 0]  # exclude background

            for inst_id in instances:
                binary_mask = (mask == inst_id).astype(np.uint8)
                masks.append(binary_mask)
                labels.append(class_idx)

        if len(masks) == 0:
            # no annotations, dummy box and mask
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.uint8),
                'image_id': torch.tensor([idx])
            }
        else:
            masks = np.stack(masks)
            boxes = masks_to_boxes(torch.tensor(masks))
            target = {
                'boxes': boxes,
                'labels': torch.tensor(labels, dtype=torch.int64),
                'masks': torch.tensor(masks, dtype=torch.uint8),
                'image_id': torch.tensor([idx])
            }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_folders)
