from pathlib import Path

import numpy as np
import onnx

from src.core.models.efficientnet import efficientnet_b0
from src.core.models.resnet import resnet18
from src.core.tensor import Tensor
from src.core.onnx import export_as_onnx


def _run_export(output_tensor: Tensor, file_path: Path) -> None:
  """
  Helper — export, verify with onnx.checker, then remove the artefact.
  """
  export_as_onnx(output_tensor, str(file_path))
  onnx.checker.check_model(str(file_path))
  assert file_path.exists()
  file_path.unlink()


def test_resnet18_onnx_export(tmp_path):
  model = resnet18(num_classes=4, in_channels=1)
  x = Tensor(np.zeros((1, 1, 224, 224), np.float32), requires_grad=True)
  _run_export(model(x), tmp_path / "resnet18.onnx")


def test_efficientnet_b0_onnx_export(tmp_path):
  model = efficientnet_b0(number_of_classes=4, input_channels=1)
  x = Tensor(np.zeros((1, 1, 224, 224), np.float32), requires_grad=True)
  _run_export(model(x), tmp_path / "efficientnet_b0.onnx")
