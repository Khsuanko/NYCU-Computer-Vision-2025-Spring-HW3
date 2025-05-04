import torch
import json
import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from model import get_instance_segmentation_model
from utils import encode_mask
import configs

def main():
    model = get_instance_segmentation_model(configs.num_classes)
    model.load_state_dict(torch.load(configs.save_path))
    model.to(configs.device)
    model.eval()

    with open(configs.test_json, 'r') as f:
        test_info = json.load(f)

    id_mapping = {item['file_name']: item['id'] for item in test_info}
    test_dir = Path(configs.test_dir)

    results = []

    for filename in tqdm(os.listdir(test_dir)):
        if not filename.endswith('.tif'):
            continue
        image_id = id_mapping[filename]
        image_path = test_dir / filename

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(configs.device)

        with torch.no_grad():
            outputs = model(image_tensor)

        output = outputs[0]
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        masks = output['masks'].cpu().numpy()
        boxes = output['boxes'].cpu().numpy()

        for score, label, mask, box in zip(scores, labels, masks, boxes):
            if score < 0.3:
                continue  # threshold

            mask_bin = mask[0] > 0.5
            rle = encode_mask(mask_bin)

            results.append({
                'image_id': image_id,
                'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                'score': float(score),
                'category_id': int(label),
                'segmentation': rle
            })

    with open('test-results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
