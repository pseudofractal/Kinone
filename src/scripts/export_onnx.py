"""
Trace a NumPy‑autograd graph and dump a rock‑solid ONNX‑v20 file.

Example
-------
>>> out = model(x)  # forward pass
>>> export_onnx(out, "model.onnx")
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.core.tensor import Tensor


def _trace(root: Tensor) -> List[Tensor]:
  """Parents first, children after. No post‑reverse nonsense."""
  topo, seen = [], set()

  def visit(t: Tensor) -> None:
    if id(t) in seen:
      return
    seen.add(id(t))
    if t._ctx:
      for p in t._ctx.parents:
        if isinstance(p, Tensor):
          visit(p)
    topo.append(t)

  visit(root)
  return topo


def export_onnx(
  output: Tensor,
  onnx_path: str,
  *,
  input_name: str = "input",
  opset: int = 20,
) -> None:
  """
  Convert the autograd graph ending at *output* into an ONNX model.

  Parameters
  ----------
  output : Tensor
      Graph output tensor.
  onnx_path : str
      Destination .onnx file.
  input_name : str, default "input"
      Name to give the graph input.
  opset : int, default 20
      Target ONNX opset.
  """
  tensors = _trace(output)
  name_of: Dict[int, str] = {}

  def ensure_name(t: Tensor) -> str:
    if not getattr(t, "name", None):
      t.name = f"t{id(t)}"
    name_of[id(t)] = t.name
    return t.name

  for t in tensors:
    ensure_name(t)

  nodes: List[onnx.NodeProto] = []
  initializers: Dict[str, onnx.TensorProto] = OrderedDict()

  def add_init(t: Tensor) -> None:
    if t.name not in initializers:
      initializers[t.name] = numpy_helper.from_array(t.data, name=t.name)

  for t in tensors:
    print(t.name)
    ctx = t._ctx
    if ctx is None:
      if t is not output and not t.requires_grad:
        add_init(t)
      continue

    op = type(ctx).__name__

    if op == "Conv2d":
      x, w, b, *_ = ctx.parents
      add_init(w)
      inputs = [name_of[id(x)], name_of[id(w)]]
      if b is not None:
        add_init(b)
        inputs.append(name_of[id(b)])
      attrs = {
        "strides": [ctx.stride, ctx.stride],
        "pads": [ctx.padding] * 4,
        "kernel_shape": list(w.data.shape[2:]),
      }
      nodes.append(
        helper.make_node("Conv", inputs, [t.name], name=f"{t.name}_conv", **attrs)
      )

    elif op == "BatchNorm2dOp":
      x, gamma, beta, *_ = ctx.parents
      add_init(gamma)
      add_init(beta)

      rm_arr = getattr(ctx, "running_mean", np.zeros_like(gamma.data, dtype=np.float32))
      rv_arr = getattr(ctx, "running_var", np.ones_like(gamma.data, dtype=np.float32))
      rm = numpy_helper.from_array(rm_arr, name=f"{gamma.name}_rm")
      rv = numpy_helper.from_array(rv_arr, name=f"{gamma.name}_rv")
      initializers[rm.name] = rm
      initializers[rv.name] = rv

      nodes.append(
        helper.make_node(
          "BatchNormalization",
          [name_of[id(x)], gamma.name, beta.name, rm.name, rv.name],
          [t.name],
          epsilon=float(ctx.ε),
          name=f"{t.name}_bn",
        )
      )

    elif op == "ReLU":
      (x,) = ctx.parents
      nodes.append(
        helper.make_node("Relu", [name_of[id(x)]], [t.name], name=f"{t.name}_relu")
      )

    elif op == "MatMul":
      a, b = ctx.parents
      add_init(b)
      nodes.append(
        helper.make_node(
          "Gemm",
          [name_of[id(a)], name_of[id(b)]],
          [t.name],
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
          "Add", [name_of[id(a)], name_of[id(b)]], [t.name], name=f"{t.name}_add"
        )
      )

    elif op == "AdaptiveAvgPool2dOp":
      (x,) = ctx.parents
      nodes.append(
        helper.make_node(
          "GlobalAveragePool", [name_of[id(x)]], [t.name], name=f"{t.name}_gap"
        )
      )

    elif op == "Reshape":
      (x,) = ctx.parents
      shape_tensor = numpy_helper.from_array(
        np.asarray(t.shape, dtype=np.int64), name=f"{t.name}_shape"
      )
      initializers[shape_tensor.name] = shape_tensor
      nodes.append(
        helper.make_node(
          "Reshape",
          [name_of[id(x)], shape_tensor.name],
          [t.name],
          name=f"{t.name}_reshape",
        )
      )

    else:
      raise NotImplementedError(f"ONNX export not implemented for op {op!r}")

  for leaf in tensors:
    if leaf._ctx is None and leaf.requires_grad:
      graph_inputs = [
        helper.make_tensor_value_info(
          name_of[id(leaf)], TensorProto.FLOAT, list(leaf.shape)
        )
      ]
      break
  else:
    raise RuntimeError("Graph has no leaf tensor with requires_grad=True")

  graph_outputs = [
    helper.make_tensor_value_info(output.name, TensorProto.FLOAT, list(output.shape))
  ]

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
    producer_name="kinone‑export",
  )

  onnx.checker.check_model(model)
  onnx.save(model, onnx_path)
  print(f"∘ [INFO] Exported ONNX model ➜ {onnx_path}")
