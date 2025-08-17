import argparse
import importlib
from pathlib import Path

import numpy as np

from src.core.onnx import export_as_onnx
from src.core.tensor import Tensor

from .train import DISEASES


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


def resolve_model_builder(architecture_name: str):
  potential_module_names = [
    "src.core.models.resnet",
    "src.core.models.efficientnet",
  ]
  for module_name in potential_module_names:
    module_reference = importlib.import_module(module_name)
    if hasattr(module_reference, architecture_name):
      return getattr(module_reference, architecture_name)
  raise ValueError(f"Unknown architecture name: {architecture_name}")


def main():
  parser = argparse.ArgumentParser(
    description="Export a trained model to ONNX from a specific run."
  )
  parser.add_argument(
    "--run-dir",
    type=str,
    required=True,
    help="Path to the training run directory containing the checkpoint.",
  )
  parser.add_argument(
    "--output-filename",
    type=str,
    default="model.onnx",
    help="Filename for the output ONNX model.",
  )
  args = parser.parse_args()
  run_dir = Path(args.run_dir)

  run_config_path = run_dir / "run_config.json"
  if not run_config_path.exists():
    raise FileNotFoundError(f"run_config.json not found in {run_dir}")
  architecture = "efficientnet_b0"

  model_builder = resolve_model_builder(architecture)
  model_instance = model_builder(number_of_classes=len(DISEASES), input_channels=1)

  checkpoint_path = find_best_checkpoint(run_dir)
  print(f"[INFO] Loading weights from: {checkpoint_path}")
  state_dictionary = np.load(checkpoint_path)
  model_instance.load_state_dict(state_dictionary)

  dummy_input_tensor = Tensor(
    np.zeros((1, 1, 224, 224), dtype=np.float32),
    requires_grad=True,
  )
  output_tensor = model_instance(dummy_input_tensor)

  output_path = run_dir / args.output_filename
  print(f"[INFO] Exporting ONNX model to: {output_path}")
  export_as_onnx(output_tensor, str(output_path))
  print("[INFO] Export complete.")


if __name__ == "__main__":
  main()
