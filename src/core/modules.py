"""
NN layer functionalities.

∘ Module          –  PyTorch-style base: tracks sub-modules, parameters, toggles train/eval, supports state-dict I/O.
∘ Linear          –  y = x Wᵀ + b  with He-initialization (√(2/n)).
∘ Conv2D          –  NCHW convolution wrapped around `ops.conv2d`.
∘ ReLU            –  thin wrapper over `ops.relu`.
∘ Sequential      –  for-loop composition.

Everything returns/accepts `Tensor`, so gradients flow automatically.
"""
from typing import Iterable

import numpy as np

from .ops import add, conv2d, matmul, relu
from .tensor import Tensor


class Module:
  training: bool = True

  def named_parameters(self, prefix: str = "") -> Iterable[tuple[str, Tensor]]:
    for name, attr in self.__dict__.items():
      if name.startswith("_"):
        continue

      full_name = f"{prefix}.{name}" if prefix else name
      if isinstance(attr, Tensor):
        yield full_name, attr
      elif isinstance(attr, Module):
        yield from attr.named_parameters(full_name)
      elif isinstance(attr, (list, tuple)):
        for i, elem in enumerate(attr):
          if isinstance(elem, Module):
            yield from elem.named_parameters(f"{full_name}.{i}")

  def parameters(self) -> Iterable[Tensor]:
    for _, param in self.named_parameters():
      yield param

  def load_state_dict(self, state_dict: dict[str, np.ndarray]):
    for name, param in self.named_parameters():
      if name in state_dict:
        param.data = state_dict[name]
      else:
        print(f"  ∘ [WARN] Missing key in state_dict: {name}")

  def set_to_training(self, mode: bool = True):
    self.training = mode
    for attr in self.__dict__.values():
      if isinstance(attr, Module):
        attr.set_to_training(mode)
      elif isinstance(attr, (list, tuple)):
        for elem in attr:
          if isinstance(elem, Module):
            elem.set_to_training(mode)
    return self

  def set_to_evaluation(self):
    return self.set_to_training(False)

  def zero_grad(self):
    for p in self.parameters():
      p.grad = None

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class Linear(Module):
  """
  Applies a linear transformation to the incoming data: y = x A^T + b

  The weights are initialized using Kaiming (He) initialization.
  """

  def __init__(self, in_features: int, out_features: int, bias: bool = True):
    """
    Initialize random weights using Kaiming (He) initialization,
    which is specifically designed to work well with ReLU activations.
    In deep neural networks, if weights are initialized poorly, then activations can:
    - Shrink to zero (Vanishing)
    - Explode to infinity (Exploding)
    In He initialization, the weights are initialized from a normal distribution:
    W ~ N(0, sqrt(2 / in_features)). Since ReLU zeros out half the inputs,
    to keep the variance of activtions constant,
    we scale up variance of weights by factoe of 2.``
    """
    w = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(
      2.0 / in_features
    )
    self.weight = Tensor(w, True)
    self.bias = Tensor(np.zeros(out_features, dtype=np.float32), True) if bias else None

  def __call__(self, x: Tensor) -> Tensor:
    out = matmul(x, self.weight)
    if self.bias is not None:
      out = add(out, self.bias)
    return out


class Conv2D(Module):
  """
  Applies a 2D convolution over an input signal composed of several input
  planes.
  """

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    groups: int = 1,
  ):
    weights = np.random.randn(
      out_channels, in_channels, kernel_size, kernel_size
    ).astype(np.float32) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
    self.weight = Tensor(weights, True)
    self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), True) if bias else None
    self.stride = stride
    self.padding = padding
    self.groups = groups

  def __call__(self, input: Tensor):
    return conv2d(input, self.weight, self.bias, self.stride, self.padding, self.groups)


class ReLU(Module):
  def __call__(self, x: Tensor) -> Tensor:
    return relu(x)


class Sequential(Module):
  def __init__(self, *modules: Module | Iterable[Module]):
    flat: list[Module] = []
    for m in modules:
      if isinstance(m, (list, tuple)):
        flat.extend(m)
      else:
        flat.append(m)
    self.layers: list[Module] = flat

  def __call__(self, x: Tensor) -> Tensor:
    for layer in self.layers:
      x = layer(x)
    return x
