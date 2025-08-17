from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.core.tensor import Tensor


def _topological_trace(root: Tensor) -> list[Tensor]:
  ordered: list[Tensor] = []
  visited: set[int] = set()

  def dfs(tensor: Tensor) -> None:
    if id(tensor) in visited:
      return
    visited.add(id(tensor))
    if tensor._ctx:
      for parent in tensor._ctx.parents:
        if isinstance(parent, Tensor):
          dfs(parent)
    ordered.append(tensor)

  dfs(root)
  return ordered


def export_as_onnx(
  graph_output: Tensor,
  destination: str,
  *,
  input_tensor_name: str = "input",
  opset_version: int = 20,
) -> None:
  """
  Serialize the NumPy-autograd graph ending at graph_output as an ONNX model.

  Parameters
  ----------
  graph_output
      The final tensor produced by the forward pass.
  destination
      Path where the *.onnx* file will be written.
  input_tensor_name
      Name assigned to the single graph input.
  opset_version
      Desired ONNX opset version.
  """
  traced_tensors = _topological_trace(graph_output)

  def ensure_identifier(tensor: Tensor) -> str:
    if not getattr(tensor, "name", None):
      tensor.name = f"tensor_{id(tensor):x}"
    return tensor.name

  for tensor in traced_tensors:
    ensure_identifier(tensor)

  node_protos: list[onnx.NodeProto] = []
  initializer_map: dict[str, onnx.TensorProto] = {}

  def add_initializer(tensor: Tensor) -> None:
    if tensor.name not in initializer_map:
      initializer_map[tensor.name] = numpy_helper.from_array(
        tensor.data.astype(np.float32, copy=False), name=tensor.name
      )

  leaf_tensors = [t for t in traced_tensors if t._ctx is None]

  graph_input_tensor = next(
    (
      t
      for t in leaf_tensors
      if t.requires_grad and len(t.data.shape) >= 3 and t.data.shape[0] == 1
    ),
    leaf_tensors[0],
  )

  for tensor in leaf_tensors:
    if tensor is not graph_input_tensor:
      add_initializer(tensor)

  for tensor in traced_tensors:
    context = tensor._ctx
    if context is None:
      continue

    op_name = type(context).__name__

    if op_name == "Conv2d":
      x, weight, bias, *_ = context.parents
      add_initializer(weight)
      inputs = [weight.name]
      if bias is not None:
        add_initializer(bias)
        inputs.append(bias.name)
      node_protos.append(
        helper.make_node(
          "Conv",
          [x.name] + inputs,
          [tensor.name],
          name=f"{tensor.name}_conv",
          strides=[context.stride, context.stride],
          pads=[context.padding] * 4,
          kernel_shape=list(weight.data.shape[2:]),
          group=context.groups,
        )
      )

    elif op_name == "BatchNorm2dOp":
      x, gamma, beta, *_ = context.parents
      add_initializer(gamma)
      add_initializer(beta)
      running_mean = numpy_helper.from_array(
        getattr(
          context,
          "running_mean",
          np.zeros_like(gamma.data, dtype=np.float32),
        ),
        name=f"{gamma.name}_running_mean",
      )
      running_var = numpy_helper.from_array(
        getattr(
          context,
          "running_var",
          np.ones_like(gamma.data, dtype=np.float32),
        ),
        name=f"{gamma.name}_running_var",
      )
      initializer_map[running_mean.name] = running_mean
      initializer_map[running_var.name] = running_var
      node_protos.append(
        helper.make_node(
          "BatchNormalization",
          [x.name, gamma.name, beta.name, running_mean.name, running_var.name],
          [tensor.name],
          name=f"{tensor.name}_batchnorm",
          epsilon=float(context.ε),
        )
      )

    elif op_name == "ReLU":
      (x,) = context.parents
      node_protos.append(
        helper.make_node("Relu", [x.name], [tensor.name], name=f"{tensor.name}_relu")
      )

    elif op_name in {"MatMul", "Linear"}:
      a, b = context.parents
      add_initializer(b)
      node_protos.append(
        helper.make_node(
          "Gemm",
          [a.name, b.name],
          [tensor.name],
          name=f"{tensor.name}_gemm",
          alpha=1.0,
          beta=1.0,
          transB=0,
        )
      )

    elif op_name == "Add":
      a, b = context.parents
      node_protos.append(
        helper.make_node(
          "Add", [a.name, b.name], [tensor.name], name=f"{tensor.name}_add"
        )
      )

    elif op_name in {"Mul", "Multiply"}:
      a, b = context.parents
      node_protos.append(
        helper.make_node(
          "Mul", [a.name, b.name], [tensor.name], name=f"{tensor.name}_mul"
        )
      )

    elif op_name == "AdaptiveAvgPool2dOp":
      (x,) = context.parents
      node_protos.append(
        helper.make_node(
          "GlobalAveragePool", [x.name], [tensor.name], name=f"{tensor.name}_gap"
        )
      )

    elif op_name == "Reshape":
      (x,) = context.parents
      shape_tensor = numpy_helper.from_array(
        np.asarray(tensor.shape, dtype=np.int64), name=f"{tensor.name}_shape"
      )
      initializer_map[shape_tensor.name] = shape_tensor
      node_protos.append(
        helper.make_node(
          "Reshape",
          [x.name, shape_tensor.name],
          [tensor.name],
          name=f"{tensor.name}_reshape",
        )
      )

    elif op_name in {"HardSwish", "HardSwishOp"}:
      (x,) = context.parents
      node_protos.append(
        helper.make_node(
          "HardSwish",
          [x.name],
          [tensor.name],
          name=f"{tensor.name}_hardswish",
        )
      )

    elif op_name in {"HardSigmoid", "HardSigmoidOp"}:
      (x,) = context.parents
      alpha = float(getattr(context, "alpha", 1.0 / 6.0))
      beta = float(getattr(context, "beta", 0.5))
      node_protos.append(
        helper.make_node(
          "HardSigmoid",
          [x.name],
          [tensor.name],
          name=f"{tensor.name}_hardsigmoid",
          alpha=alpha,
          beta=beta,
        )
      )

    else:
      raise NotImplementedError(
        f"ONNX export not implemented for operation {op_name!r}"
      )

  graph_inputs_proto = [
    helper.make_tensor_value_info(
      graph_input_tensor.name, TensorProto.FLOAT, list(graph_input_tensor.shape)
    )
  ]

  graph_outputs_proto = [
    helper.make_tensor_value_info(
      graph_output.name, TensorProto.FLOAT, list(graph_output.shape)
    )
  ]

  graph_proto = helper.make_graph(
    node_protos,
    name="numpy_autograd_graph",
    inputs=graph_inputs_proto,
    outputs=graph_outputs_proto,
    initializer=list(initializer_map.values()),
  )

  model_proto = helper.make_model(
    graph_proto,
    opset_imports=[helper.make_opsetid("", opset_version)],
    producer_name="kinone_export",
  )

  onnx.checker.check_model(model_proto)
  onnx.save(model_proto, destination)
