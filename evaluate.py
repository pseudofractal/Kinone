import argparse

import numpy as np
from tqdm import tqdm

from src.core.metrics import accuracy_score, roc_auc_score
from src.core.models import ResNet18
from src.core.tensor import Tensor
from src.data.nih_cxr import DISEASES
from src.data.nih_datamodule import NIHDataModule


def main():
  parser = argparse.ArgumentParser(description="Evaluate a ResNet-18 model.")
  parser.add_argument(
    "--dataset", type=str, default="nih", help="The dataset to use (e.g., 'nih')."
  )
  parser.add_argument(
    "--batch-size", type=int, default=32, help="Batch size for evaluation."
  )
  parser.add_argument(
    "--checkpoint-path",
    type=str,
    required=True,
    help="Path to the saved model checkpoint to evaluate.",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed to set in the dataloader for reproducibility.",
  )

  temp_args, _ = parser.parse_known_args()
  if temp_args.dataset == "nih":
    parser = NIHDataModule.add_argparse_args(parser)
  args = parser.parse_args()

  print(f"[INFO] Loading dataset based on {args.dataset}.")
  if args.dataset == "nih":
    dm = NIHDataModule(args)
    print("[INFO] Loaded NIHDataModule.")
  else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

  dm.setup("test")
  test_dataloader = dm.test_dataloader()

  model = ResNet18(num_classes=len(DISEASES))
  print(f"[INFO] Loading model from {args.checkpoint_path}")
  model.load_state_dict(np.load(args.checkpoint_path))

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
