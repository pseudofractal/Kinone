import numpy as np
from src.core.models.resnet import resnet18
from src.core.tensor import Tensor
from src.scripts.export_onnx import export_onnx

model = resnet18(num_classes=14, in_channels=1)

state = np.load("checkpoints/best_model.npz", allow_pickle=True)
model.load_state_dict(state)

x = Tensor(np.zeros((1, 1, 224, 224), dtype=np.float32), requires_grad=True)

out = model(x)
export_onnx(out, "kinone_resnet18.onnx")

