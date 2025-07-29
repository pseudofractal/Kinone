"""
The most basic thing that we need to implement is reverse-mode automatic differentiation.
This is a minimalist attempt at creating something similar to Pytorch's `autograd`,
but heavily stripped down and worse.

Out job is in essence to construct a computation graph during forward passes,
and walk backward to compute gradients. The entites we are going to create are:

1. Tensor:
A node in the above mentioned computational graph, which will hold some data.
Optionally it may reqiure gradients, in which case it will also hold a `grad` attribute.
It will `backpropagate` gradients through the graph when requested.

2. Function:
An abstract class that will encapsulate forward and backward passes.
All operations like Add, Mul, ReLU will be subclasses to this.
When an operation is applied to tensors; it returns a new Tensor,
and builds grpah connections for backpropagation.

The computation graph:
Each operation creates a node in the graph and tracks its "parents".

"""

import uuid
from typing import List, Optional

import numpy as np


def _to_array(value):
  return np.asarray(value, dtype=np.float32)


def _as_scalar(self):
  if self.data.size != 1:
    raise TypeError("Only 0-dim Tensors can convert to Python scalars")
  return self.data.item()


class Tensor:
  __slots__ = ("data", "grad", "requires_grad", "_ctx", "shape", "name")

  def __init__(self, data, requires_grad: bool = False):
    self.data = _to_array(data)
    self.shape = self.data.shape
    self.requires_grad = requires_grad
    self.grad: Optional["Tensor"] = None
    self._ctx: Optional["Function"] = None
    self.name: str = f"tensor_{uuid.uuid4().hex[:8]}"

  def __repr__(self):
    return (
      f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}, "
      f"requires_grad={self.requires_grad})"
    )

  def __neg__(self):
    from .ops import neg

    return neg(self)

  def __add__(self, other):
    from .ops import add

    return add(self, other)

  def __radd__(self, other):
    from .ops import add

    return add(Tensor(other), self)

  def __sub__(self, other):
    from .ops import sub

    return sub(self, other)

  def __rsub__(self, other):
    from .ops import sub

    return sub(Tensor(other), self)

  def __mul__(self, other):
    from .ops import mul

    return mul(self, other)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __truediv__(self, other):
    from .ops import div

    return div(self, other)

  def __rtruediv__(self, other):
    from .ops import div

    return div(Tensor(other), self)

  def __float__(self):
    return _as_scalar(self)

  def __array__(self, dtype=None):
    return self.data if dtype is None else self.data.astype(dtype)

  def reshape(self, *shape):
    return Tensor(self.data.reshape(*shape), self.requires_grad)

  def backward(self, gradient: Optional["Tensor"] = None):
    """
    Implement reverse-mode automatic differentiation; which is a strategy for computing gradients efficiently.
    The goal is given a scalar valued function $f(vector(x))$,
    represented by a series of computation graphs (series of tensor operations);
    we compute the gradient of f using chain rule from multivariable calculus.

    To start off with:
    1. Initialize all gradients of output tensor to be 1.
    2. Build an ordering of computation graph nodes in reverse order using toplogical sort.
        Forward: x → y → z        Backward: z → y → x
    3. Backward pass over computation graph; for each node `v` a tensor, get the local `gradient` from its `Function`.
        Multiply the local gradient with the upstream gradient,
        and accumulate the gradients because a single tensor might be used in multiple branches.

    Example:
    Target function: x²+x
    z₁ = x²
    z₂ = z₁ + x
    f = z₂
    `f.backward()`

    Expected: 2x + 1; for x=3, we should expect 7.

    Functioning:
    1. z₁ = mult(x, x)
      - Stores context: parents[x, x]
      - Calls `Mul.forward()` and returns 9
      - On backward:
        • Derivatives: dz₁/dx₁ = x and dz₁/dx₂ = x
        • dz₁/dx = x * 1 + x * 1 = 2x
    2. z₂ = add(z₁, x)
       - Stores context: parents = [z₁, x]
       - Calls `Add.forward()` and returns 12
       - On backward:
         • dz₂/dz₁ = 1, dz₂/dx = 1
         • dz₂/dx = dz₂/dz₁ * dz₁/dx + dz₂/dx * 1 = (2x) * 1 + 1 = 2x + 1
    3. Final gradient is accumulated at x.grad = 2x + 1 = 7 (for x = 3)
    """
    if not self.requires_grad:
      return

    if gradient is None:
      gradient = Tensor(np.ones_like(self.data), False)
    self.grad = gradient

    topological_nodes: List[Tensor] = []
    visited_nodes: set[int] = set()

    def depth_first_search(tensor_node: "Tensor"):
      if id(tensor_node) in visited_nodes:
        return
      visited_nodes.add(id(tensor_node))
      if tensor_node._ctx is not None:
        for parent_tensor in tensor_node._ctx.parents:
          depth_first_search(parent_tensor)
      topological_nodes.append(tensor_node)

    depth_first_search(self)

    for tensor_node in reversed(topological_nodes):
      context = tensor_node._ctx
      if context is None:
        continue
      output_gradients = context.backward(tensor_node.grad.data)
      if not isinstance(output_gradients, tuple):
        output_gradients = (output_gradients,)
      for parent_tensor, parent_gradient in zip(context.parents, output_gradients):
        if parent_gradient is None or not parent_tensor.requires_grad:
          continue
        if parent_tensor.grad is None:
          parent_tensor.grad = Tensor(parent_gradient, False)
        else:
          parent_tensor.grad.data += parent_gradient

  def mean(self, axis=None, keepdims=False):
    from .ops import mean as _mean

    return _mean(self, axis, keepdims)

  def sum(self, axis=None, keepdims=False):
    from .ops import sum as _sum

    return _sum(self, axis, keepdims)
