import argparse
import importlib
from pathlib import Path

import numpy as np

from src.utils.common import load_config
from src.core.tensor import Tensor
from src.core.onnx import export_as_onnx


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


def determine_input_channels(model_instance):
    first_parameter = next(model_instance.parameters())
    if first_parameter.data.ndim >= 4:
        return first_parameter.data.shape[1]
    return 1


def main():
    configuration = load_config()
    export_section = configuration.get("export", {})

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--weights-path",
        type=str,
        default=export_section.get("weights_path", ""),
    )
    argument_parser.add_argument(
        "--architecture",
        type=str,
        default=export_section.get("architecture", "resnet18"),
    )
    argument_parser.add_argument(
        "--output-path",
        type=str,
        default=export_section.get("output_path", "model.onnx"),
    )
    command_line_arguments = argument_parser.parse_args()

    model_builder = resolve_model_builder(command_line_arguments.architecture)
    model_instance = model_builder()

    if command_line_arguments.weights_path:
        state_dictionary = np.load(command_line_arguments.weights_path)
        model_instance.load_state_dict(state_dictionary)

    channel_count = determine_input_channels(model_instance)
    dummy_input_tensor = Tensor(
        np.zeros((1, channel_count, 224, 224), dtype=np.float32),
        requires_grad=True,
    )
    output_tensor = model_instance(dummy_input_tensor)

    output_path = Path(command_line_arguments.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_as_onnx(output_tensor, str(output_path))


if __name__ == "__main__":
    main()

