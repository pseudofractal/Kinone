import argparse
import json

import cv2
import numpy as np

from src.core.models import ResNet18
from src.data.nih_cxr import DISEASES
from src.core.tensor import Tensor
from src.core.losses import sigmoid

def main():
    with open('config.jsonc') as f:
        config = json.load(f)['predict']

    parser = argparse.ArgumentParser(description='Get predictions for a single image.')
    parser.add_argument('--checkpoint-path', type=str, default=config.get('checkpoint_path'), help='Path to the saved model checkpoint.')
    parser.add_argument('--image-path', type=str, default=config.get('image_path'), required=config.get('image_path') is None, help='Path to the input image.')
    args = parser.parse_args()

    model = ResNet18(num_classes=len(DISEASES))
    print(f"[INFO] Loading model from {args.checkpoint_path}")
    model.load_state_dict(np.load(args.checkpoint_path))
    model.set_to_evaluation()

    print(f"[INFO] Loading image from {args.image_path}")
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not find or open the image at {args.image_path}")

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = image[np.newaxis, np.newaxis, ...]

    image_tensor = Tensor(image)
    logits = model(image_tensor)
    probabilities = sigmoid(logits.data)

    print("\n--- Predictions ---")
    for disease, probability in zip(DISEASES, probabilities.squeeze()):
        print(f"{disease:<20}: {probability:.4f}")

if __name__ == '__main__':
    main()
