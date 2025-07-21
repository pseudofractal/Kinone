"""
Trace a NumPy autograd graph and export it as an **ONNX-v20** file.

Example
-------
>>> out = model(x)  # forward pass on your NumPy-autograd model
>>> export_onnx(out, "model.onnx")  # writes a fully-portable ONNX file
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.core.tensor import Tensor


def _trace(output: Tensor) -> List[Tensor]:
  """Return tensors in topological order – parents before children."""
  topo: List[Tensor] = []
  seen: set[int] = set()

  def _visit(t: Tensor) -> None:
    if id(t) in seen:
      return
    seen.add(id(t))
    if t._ctx is not None:
      for p in t._ctx.parents:
        if isinstance(p, Tensor):
          _visit(p)
    topo.append(t)

  _visit(output)
  return topo[::-1]


def export_onnx(
  output: Tensor,
  onnx_path: str,
  *,
  input_name: str = "input",
  opset: int = 20,
) -> None:
  """
  Convert the autograd graph ending at ``output`` into an ONNX model.

  Parameters
  ----------
  output : Tensor
      The graph's output tensor (must have ``_ctx`` or be a leaf).
  onnx_path : str
      Destination path for the ONNX file.
  input_name : str, default "input"
      Name given to the model input in the ONNX graph.
  opset : int, default 20
      Target ONNX opset version.
  """
  tensors = _trace(output)

  def _ensure_name(t: Tensor) -> str:
    if not getattr(t, "name", None):
      t.name = f"t{id(t)}"
    return t.name

  name_of: Dict[int, str] = {id(t): _ensure_name(t) for t in tensors}

  def _add_initializer(t: Tensor) -> None:
    if t.name not in initializers:
      initializers[t.name] = numpy_helper.from_array(t.data, name=t.name)

  nodes: list[onnx.NodeProto] = []
  graph_inputs: list[onnx.ValueInfoProto] = []
  graph_outputs: list[onnx.ValueInfoProto] = []
  initializers: Dict[str, onnx.TensorProto] = OrderedDict()

  for t in tensors:
    ctx = t._ctx

    if ctx is None:
      if t is not output and not t.requires_grad:
        _add_initializer(t)
      continue

    op = type(ctx).__name__

    if op == "Conv2d":
      x, w, b, *_ = ctx.parents
      _add_initializer(w)
      conv_inputs = [name_of[id(x)], name_of[id(w)]]

      if b is not None:
        _add_initializer(b)
        conv_inputs.append(name_of[id(b)])

      attrs = {
        "strides": [ctx.stride, ctx.stride],
        "pads": [ctx.padding] * 4,
        "kernel_shape": list(w.data.shape[2:]),
      }
      nodes.append(
        helper.make_node(
          "Conv",
          inputs=conv_inputs,
          outputs=[t.name],
          name=f"{t.name}_conv",
          **attrs,
        )
      )

    elif op == "BatchNorm2dOp":
      x, gamma, beta, *_ = ctx.parents
      _add_initializer(gamma)
      _add_initializer(beta)

      rm_arr = getattr(ctx, "running_mean", np.zeros_like(gamma.data))
      rv_arr = getattr(ctx, "running_var", np.ones_like(gamma.data))

      rm = numpy_helper.from_array(rm_arr, name=f"{gamma.name}_rm")
      rv = numpy_helper.from_array(rv_arr, name=f"{gamma.name}_rv")
      initializers[rm.name] = rm
      initializers[rv.name] = rv

      nodes.append(
        helper.make_node(
          "BatchNormalization",
          inputs=[
            name_of[id(x)],
            gamma.name,
            beta.name,
            rm.name,
            rv.name,
          ],
          outputs=[t.name],
          epsilon=float(ctx.ε),
          name=f"{t.name}_bn",
        )
      )

    elif op == "ReLU":
      (x,) = ctx.parents
      nodes.append(
        helper.make_node(
          "Relu",
          inputs=[name_of[id(x)]],
          outputs=[t.name],
          name=f"{t.name}_relu",
        )
      )

    elif op == "MatMul":
      a, b = ctx.parents
      _add_initializer(b)
      nodes.append(
        helper.make_node(
          "Gemm",
          inputs=[name_of[id(a)], name_of[id(b)]],
          outputs=[t.name],
          alpha=1.0,
          beta=1.0,
          transB=0,
          name=f"{t.name}_gemm",
        )
      )

    elif op == "Add":
      a, b = ctx.parents
      nodes.append(
        helper.make_node(
          "Add",
          inputs=[name_of[id(a)], name_of[id(b)]],
          outputs=[t.name],
          name=f"{t.name}_add",
        )
      )

    elif op == "AdaptiveAvgPool2dOp":
      (x,) = ctx.parents
      nodes.append(
        helper.make_node(
          "GlobalAveragePool",
          inputs=[name_of[id(x)]],
          outputs=[t.name],
          name=f"{t.name}_gap",
        )
      )

    elif op == "Reshape":
      (x,) = ctx.parents
      shape_tensor = numpy_helper.from_array(
        np.asarray(t.shape, dtype=np.int64),
        name=f"{t.name}_shape",
      )
      initializers[shape_tensor.name] = shape_tensor
      nodes.append(
        helper.make_node(
          "Reshape",
          inputs=[name_of[id(x)], shape_tensor.name],
          outputs=[t.name],
          name=f"{t.name}_reshape",
        )
      )

    else:
      raise NotImplementedError(f"ONNX export for op {op!r} not implemented.")

  for leaf in tensors:
    if leaf._ctx is None and leaf.requires_grad:
      graph_inputs.append(
        helper.make_tensor_value_info(input_name, TensorProto.FLOAT, list(leaf.shape))
      )
      initializers.pop(leaf.name, None)
      name_of[id(leaf)] = input_name
      break
  else:
    raise RuntimeError("Could not locate a suitable graph input tensor.")

  graph_outputs.append(
    helper.make_tensor_value_info(output.name, TensorProto.FLOAT, list(output.shape))
  )

  graph = helper.make_graph(
    nodes=nodes,
    name="numpy_autograd",
    inputs=graph_inputs,
    outputs=graph_outputs,
    initializer=list(initializers.values()),
  )
  model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", opset)],
    producer_name="kinone-export",
  )

  onnx.checker.check_model(model)
  onnx.save(model, onnx_path)
  print(f"∘ [INFO] Exported ONNX model ➜ {onnx_path}")
