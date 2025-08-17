import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.core.metrics import accuracy_score, roc_auc_score
from src.core.models.efficientnet import efficientnet_b0
from src.core.tensor import Tensor
from src.data.nih_cxr import DISEASES
from src.data.nih_datamodule import NIHDataModule


def find_best_checkpoint(run_dir: Path) -> str:
  checkpoints_dir = run_dir / "checkpoints"
  if not checkpoints_dir.exists():
    raise FileNotFoundError(f"Checkpoints directory not found in {run_dir}")

  checkpoints = list(checkpoints_dir.glob("best_model_*.npz"))
  if not checkpoints:
    raise FileNotFoundError(
      f"No 'best_model_*.npz' checkpoints found in {checkpoints_dir}"
    )

  latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
  return str(latest_checkpoint)


def main():
  parser = argparse.ArgumentParser(
    description="Evaluate a trained model from a specific run."
  )
  parser.add_argument(
    "--run-dir",
    type=str,
    required=True,
    help="Path to the training run directory (e.g., 'runs/20250817_180000').",
  )
  parser.add_argument(
    "--batch-size", type=int, default=32, help="Batch size for evaluation."
  )

  args = parser.parse_args()
  run_dir = Path(args.run_dir)

  run_config_path = run_dir / "run_config.json"
  if not run_config_path.exists():
    raise FileNotFoundError(f"run_config.json not found in {run_dir}")
  with open(run_config_path, "r") as f:
    run_config = json.load(f)["hyperparameters"]

  dm_args = argparse.Namespace(**run_config)
  dm_args.batch_size = args.batch_size

  print(f"[INFO] Evaluating run: {run_dir.name}")
  dm = NIHDataModule(dm_args)
  dm.setup("test")
  test_dataloader = dm.test_dataloader()

  checkpoint_path = find_best_checkpoint(run_dir)
  print(f"[INFO] Loading model from {checkpoint_path}")

  model = efficientnet_b0(number_of_classes=len(DISEASES), input_channels=1)
  model.load_state_dict(np.load(checkpoint_path))

  print("\n[INFO] Evaluating on the test set...")
  model.set_to_evaluation()

  all_predictions = []
  all_labels = []

  for images, labels in tqdm(
    test_dataloader, desc="Evaluating", bar_format="  {l_bar}{bar}{r_bar}"
  ):
    image_tensor = Tensor(images)
    preds = model(image_tensor)
    all_predictions.append(preds.data)
    all_labels.append(labels)

  all_predictions = np.concatenate(all_predictions)
  all_labels = np.concatenate(all_labels)

  auc = roc_auc_score(all_labels, all_predictions, average="macro")
  accuracy = accuracy_score(all_labels, all_predictions > 0.5)

  print(f"  Test AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
  main()
