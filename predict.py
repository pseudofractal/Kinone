import argparse
from pathlib import Path

import cv2
import numpy as np

from src.core.losses import _sigmoid
from src.core.models.efficientnet import efficientnet_b0
from src.core.tensor import Tensor
from src.data.nih_cxr import DISEASES


def find_best_checkpoint(run_dir: Path) -> str:
  checkpoints_dir = run_dir / "checkpoints"
  if not checkpoints_dir.exists():
    raise FileNotFoundError(f"Checkpoints directory not found in {run_dir}")
  checkpoints = list(checkpoints_dir.glob("best_model_*.npz"))
  if not checkpoints:
    raise FileNotFoundError(
      f"No 'best_model_*.npz' checkpoints found in {checkpoints_dir}"
    )
  return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


def main():
  parser = argparse.ArgumentParser(description="Get predictions for a single image.")
  parser.add_argument(
    "--run-dir", type=str, required=True, help="Path to the training run directory."
  )
  parser.add_argument(
    "--image-path", type=str, required=True, help="Path to the input image."
  )
  args = parser.parse_args()
  run_dir = Path(args.run_dir)

  model = efficientnet_b0(number_of_classes=len(DISEASES), input_channels=1)

  checkpoint_path = find_best_checkpoint(run_dir)
  print(f"[INFO] Loading model from {checkpoint_path}")
  model.load_state_dict(np.load(checkpoint_path))
  model.set_to_evaluation()

  print(f"[INFO] Loading image from {args.image_path}")
  image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
  if image is None:
    raise FileNotFoundError(f"Could not find or open the image at {args.image_path}")

  image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
  image = (image.astype(np.float32) - (0.449 * 255.0)) / (0.226 * 255.0)
  image = image[np.newaxis, np.newaxis, ...]

  image_tensor = Tensor(image)
  logits = model(image_tensor)
  probabilities = _sigmoid(logits.data)

  print("\n--- Predictions ---")
  for disease, probability in zip(DISEASES, probabilities.squeeze()):
    print(f"{disease:<20}: {probability:.4f}")


if __name__ == "__main__":
  main()
