"""
Primitives that power the autodiff engine.

Design
∘ Function.apply builds the forward result and the backward closure.
∘ Broadcasting-aware gradients via `_unbroadcast`.
∘ Coverage: element-wise {add, sub, mul, div, neg}, matmul, ReLU, reductions (sum, mean), and NumPy-level Conv2d.

Each op stores only what is indispensable for its gradient
keeps memory footprint predictable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import as_strided, sliding_window_view

from .tensor import Tensor


def _unbroadcast(
  gradient_array: np.ndarray, target_shape: tuple[int, ...]
) -> np.ndarray:
  while gradient_array.ndim > len(target_shape):
    gradient_array = gradient_array.sum(axis=0)
  for axis_index, size in enumerate(target_shape):
    if size == 1 and gradient_array.shape[axis_index] != 1:
      gradient_array = gradient_array.sum(axis=axis_index, keepdims=True)
  return gradient_array


class Function(ABC):
  def __init__(self) -> None:
    self.parents: list[Tensor] = []
    self.saved_tensors: tuple[np.ndarray, ...] = ()

  def save_for_backward(self, *tensors) -> None:
    self.saved_tensors = tuple(tensors)

  @classmethod
  def apply(cls, *args, **kwargs) -> Tensor:
    context = cls()
    context.parents = [argument for argument in args if isinstance(argument, Tensor)]
    raw_arguments = [
      argument.data if isinstance(argument, Tensor) else argument for argument in args
    ]
    output_data = cls.forward(context, *raw_arguments, **kwargs)
    requires_grad_flag = any(
      isinstance(argument, Tensor) and argument.requires_grad for argument in args
    )
    output_tensor = Tensor(output_data, requires_grad_flag)
    output_tensor._ctx = context
    return output_tensor

  @staticmethod
  @abstractmethod
  def forward(ctx, *args, **kwargs):
    raise NotImplementedError

  @abstractmethod
  def backward(self, gradient_output) -> np.ndarray | tuple[np.ndarray, ...]:
    raise NotImplementedError


class Add(Function):
  @staticmethod
  def forward(ctx, *args, **kwargs):
    left_operand, right_operand = args
    ctx.save_for_backward(left_operand, right_operand)
    return left_operand + right_operand

  def backward(self, gradient_output):
    left_operand, right_operand = self.saved_tensors
    gradient_left = _unbroadcast(gradient_output, left_operand.shape)
    gradient_right = _unbroadcast(gradient_output, right_operand.shape)
    if left_operand is right_operand:
      gradient_right = gradient_right.copy()
    return gradient_left, gradient_right


def add(a: Tensor | float, b: Tensor | float):
  a = a if isinstance(a, Tensor) else Tensor(a)
  b = b if isinstance(b, Tensor) else Tensor(b)
  return Add.apply(a, b)


class Neg(Function):
  @staticmethod
  def forward(ctx, *args, **kwargs):
    (input_tensor,) = args
    return -input_tensor

  def backward(self, gradient_output):
    return -gradient_output


def neg(x: Tensor):
  return Neg.apply(x)


def sub(a, b):
  return add(a, neg(b))


class Mul(Function):
  @staticmethod
  def forward(ctx, *args, **kwargs):
    left_operand, right_operand = args
    ctx.save_for_backward(left_operand, right_operand)
    return left_operand * right_operand

  def backward(self, gradient_output):
    left_operand, right_operand = self.saved_tensors
    gradient_left = _unbroadcast(gradient_output * right_operand, left_operand.shape)
    gradient_right = _unbroadcast(gradient_output * left_operand, right_operand.shape)
    if left_operand is right_operand:
      gradient_right = gradient_right.copy()
    return gradient_left, gradient_right


def mul(a, b):
  a = a if isinstance(a, Tensor) else Tensor(a)
  b = b if isinstance(b, Tensor) else Tensor(b)
  return Mul.apply(a, b)


class Div(Function):
  @staticmethod
  def forward(ctx, *args, **kwargs):
    numerator, denominator = args
    ctx.save_for_backward(numerator, denominator)
    return numerator / denominator

  def backward(self, gradient_output):
    numerator, denominator = self.saved_tensors
    gradient_numerator = _unbroadcast(gradient_output / denominator, numerator.shape)
    gradient_denominator = _unbroadcast(
      -gradient_output * numerator / (denominator * denominator), denominator.shape
    )
    if numerator is denominator:
      gradient_denominator = gradient_denominator.copy()
    return gradient_numerator, gradient_denominator


def div(a, b):
  a = a if isinstance(a, Tensor) else Tensor(a)
  b = b if isinstance(b, Tensor) else Tensor(b)
  return Div.apply(a, b)


class ReLU(Function):
  @staticmethod
  def forward(ctx, *args, **kwargs):
    (input_tensor,) = args
    mask = (input_tensor > 0).astype(input_tensor.dtype)
    ctx.save_for_backward(mask)
    return input_tensor * mask

  def backward(self, gradient_output):
    (mask,) = self.saved_tensors
    return gradient_output * mask


def relu(x: Tensor):
  return ReLU.apply(x)


class Sigmoid(Function):
  """
  Sigmoid
  Forward σ(x)=1 ÷ (1+e^(−x))
  Backward ∂ℒ/∂x = ∂ℒ/∂y · σ(x) · (1−σ(x))
  """

  @staticmethod
  def forward(ctx, *args, **kwargs):
    (input_tensor,) = args
    output_tensor = 1.0 / (1.0 + np.exp(-input_tensor))
    ctx.save_for_backward(output_tensor)
    return output_tensor

  def backward(self, gradient_output):
    (output_tensor,) = self.saved_tensors
    return gradient_output * output_tensor * (1.0 - output_tensor)


def sigmoid(x: Tensor):
  return Sigmoid.apply(x)


class Tanh(Function):
  """
  Tanh
  Forward tanh(x)=sinh(x) ÷ cosh(x)
  Backward ∂ℒ/∂x = ∂ℒ/∂y · (1−tanh²(x))
  """

  @staticmethod
  def forward(ctx, *args, **kwargs):
    (input_tensor,) = args
    output_tensor = np.tanh(input_tensor)
    ctx.save_for_backward(output_tensor)
    return output_tensor

  def backward(self, gradient_output):
    (output_tensor,) = self.saved_tensors
    return gradient_output * (1.0 - output_tensor * output_tensor)


def tanh(x: Tensor):
  return Tanh.apply(x)


class Swish(Function):
  """
  Swish
  Forward s(x)=x · σ(x) with σ(x)=1 ÷ (1+e^(−x))
  Backward ∂ℒ/∂x = ∂ℒ/∂y · [σ(x)+x·σ(x)·(1−σ(x))]
  """

  @staticmethod
  def forward(ctx, *args, **kwargs):
    (input_tensor,) = args
    sigmoid_part = 1.0 / (1.0 + np.exp(-input_tensor))
    ctx.save_for_backward(input_tensor, sigmoid_part)
    return input_tensor * sigmoid_part

  def backward(self, gradient_output):
    input_tensor, sigmoid_part = self.saved_tensors
    return gradient_output * (
      sigmoid_part + input_tensor * sigmoid_part * (1.0 - sigmoid_part)
    )


def swish(x: Tensor):
  return Swish.apply(x)


class HardSigmoid(Function):
  """
  Hard Sigmoid
  Forward h(x)=clip((x+1) ÷ 2, 0, 1)
  Backward ∂ℒ/∂x = ∂ℒ/∂y · {½ if −1 < x < 1 else 0}
  """

  @staticmethod
  def forward(ctx, *args, **kwargs):
    (input_tensor,) = args
    slope_mask = ((input_tensor > -1.0) & (input_tensor < 1.0)).astype(
      input_tensor.dtype
    ) * 0.5
    ctx.save_for_backward(slope_mask)
    return np.clip((input_tensor + 1.0) * 0.5, 0.0, 1.0)

  def backward(self, gradient_output):
    (slope_mask,) = self.saved_tensors
    return gradient_output * slope_mask


def hard_sigmoid(x: Tensor):
  return HardSigmoid.apply(x)


class HardTanh(Function):
  """
  Hard Tanh
  Forward h(x)=clip(x, −1, 1)
  Backward ∂ℒ/∂x = ∂ℒ/∂y · {1 if −1 < x < 1 else 0}
  """

  @staticmethod
  def forward(ctx, *args, **kwargs):
    (input_tensor,) = args
    pass_through_mask = ((input_tensor > -1.0) & (input_tensor < 1.0)).astype(
      input_tensor.dtype
    )
    ctx.save_for_backward(pass_through_mask)
    return np.clip(input_tensor, -1.0, 1.0)

  def backward(self, gradient_output):
    (pass_through_mask,) = self.saved_tensors
    return gradient_output * pass_through_mask


def hard_tanh(x: Tensor):
  return HardTanh.apply(x)


class HardSwish(Function):
  """
  Hard Swish
  Forward h(x)=x·clip((x+3) ÷ 6, 0, 1)
  Backward ∂ℒ/∂x = ∂ℒ/∂y · [clip((x+3) ÷ 6,0,1) + x·(1⁄6)·𝟙_{−3<x<3}]
  """

  @staticmethod
  def forward(ctx, *args, **kwargs):
    (input_tensor,) = args
    hard_sigmoid_part = np.clip((input_tensor + 3.0) / 6.0, 0.0, 1.0)
    slope_mask = ((input_tensor > -3.0) & (input_tensor < 3.0)).astype(
      input_tensor.dtype
    ) / 6.0
    ctx.save_for_backward(input_tensor, hard_sigmoid_part, slope_mask)
    return input_tensor * hard_sigmoid_part

  def backward(self, gradient_output):
    input_tensor, hard_sigmoid_part, slope_mask = self.saved_tensors
    return gradient_output * (hard_sigmoid_part + input_tensor * slope_mask)


def hard_swish(x: Tensor):
  return HardSwish.apply(x)


class MatMul(Function):
  """
  Computes the matrix product of two tensors.

  The forward pass computes:
  Y = A @ B

  The backward pass computes the gradients with respect to the inputs A and B:
  ∂L/∂A = ∂L/∂Y @ B.T
  ∂L/∂B = A.T @ ∂L/∂Y
  """

  @staticmethod
  def forward(ctx, *args, **kwargs):
    matrix_a, matrix_b = args
    ctx.save_for_backward(matrix_a, matrix_b)
    return np.matmul(matrix_a, matrix_b)

  def backward(self, gradient_output):
    matrix_a, matrix_b = self.saved_tensors
    gradient_a = _unbroadcast(
      np.matmul(gradient_output, matrix_b.swapaxes(-1, -2)), matrix_a.shape
    )
    gradient_b = _unbroadcast(
      np.matmul(matrix_a.swapaxes(-1, -2), gradient_output), matrix_b.shape
    )
    return gradient_a, gradient_b


def matmul(a: Tensor, b: Tensor):
  return MatMul.apply(a, b)


class Sum(Function):
  @staticmethod
  def forward(
    ctx, input_tensor, axis: tuple[int, ...] | None = None, keepdims: bool = False
  ):
    ctx.axis = axis
    ctx.keepdims = keepdims
    ctx.input_shape = input_tensor.shape
    return input_tensor.sum(axis=axis, keepdims=keepdims)

  def backward(self, gradient_output):
    axis, keepdims, input_shape = self.axis, self.keepdims, self.input_shape
    if axis is None:
      return np.broadcast_to(gradient_output, input_shape)
    axis_tuple = (axis,) if isinstance(axis, int) else axis
    if not keepdims:
      gradient_output = np.expand_dims(gradient_output, axis_tuple)
    return np.broadcast_to(gradient_output, input_shape)


def sum(x: Tensor, axis: tuple[int, ...] | None = None, keepdims: bool = False):
  return Sum.apply(x, axis, keepdims)


class Mean(Function):
  @staticmethod
  def forward(
    ctx, input_tensor, axis: tuple[int, ...] | None = None, keepdims: bool = False
  ):
    ctx.axis = axis
    ctx.keepdims = keepdims
    ctx.input_shape = input_tensor.shape
    axes = (
      range(input_tensor.ndim)
      if axis is None
      else ((axis,) if isinstance(axis, int) else axis)
    )
    ctx.count = np.prod([input_tensor.shape[index] for index in axes])
    return input_tensor.mean(axis=axis, keepdims=keepdims)

  def backward(self, gradient_output):
    axis, keepdims, input_shape, count = (
      self.axis,
      self.keepdims,
      self.input_shape,
      self.count,
    )
    if axis is None:
      gradient_output = np.broadcast_to(gradient_output, input_shape)
    else:
      axis_tuple = (axis,) if isinstance(axis, int) else axis
      if not keepdims:
        gradient_output = np.expand_dims(gradient_output, axis_tuple)
      gradient_output = np.broadcast_to(gradient_output, input_shape)
    return gradient_output / count


def mean(x: Tensor, axis: tuple[int, ...] | None = None, keepdims: bool = False):
  return Mean.apply(x, axis, keepdims)


def _im2col(input_image, kernel_size, stride, padding):
  batch_size, channels, height, width = input_image.shape
  output_height = (height + 2 * padding - kernel_size) // stride + 1
  output_width = (width + 2 * padding - kernel_size) // stride + 1
  padded_input = np.pad(
    input_image, ((0, 0), (0, 0), (padding, padding), (padding, padding))
  )
  shape = (batch_size, channels, kernel_size, kernel_size, output_height, output_width)
  strides = (
    padded_input.strides[0],
    padded_input.strides[1],
    padded_input.strides[2],
    padded_input.strides[3],
    padded_input.strides[2] * stride,
    padded_input.strides[3] * stride,
  )
  columns = as_strided(padded_input, shape=shape, strides=strides)
  return (
    columns.reshape(
      batch_size, channels * kernel_size * kernel_size, output_height * output_width
    ),
    output_height,
    output_width,
  )


def _col2im(
  columns, input_shape, kernel_size, stride, padding, output_height, output_width
):
  batch_size, channels, height, width = input_shape
  padded_input = np.zeros(
    (batch_size, channels, height + 2 * padding, width + 2 * padding),
    dtype=columns.dtype,
  )
  columns = columns.reshape(
    batch_size, channels, kernel_size, kernel_size, output_height, output_width
  )
  for row_index in range(kernel_size):
    row_max = row_index + stride * output_height
    for column_index in range(kernel_size):
      column_max = column_index + stride * output_width
      padded_input[:, :, row_index:row_max:stride, column_index:column_max:stride] += (
        columns[:, :, row_index, column_index]
      )
  return (
    padded_input[:, :, padding : height + padding, padding : width + padding]
    if padding
    else padded_input
  )


class Conv2d(Function):
  """
  Performs a 2D convolution.

  The forward pass computes the convolution of the input tensor with the kernel.

  Y = W @ X_col + b

  The backward pass computes the gradients with respect to the input, kernel, and bias.

  ∂L/∂W = ∂L/∂Y @ X_col.T
  ∂L/∂b = sum(∂L/∂Y)
  ∂L/∂X_col = W.T @ ∂L/∂Y
  ∂L/∂X = col2im(∂L/∂X_col)
  """

  stride: int
  padding: int
  groups: int
  has_bias: bool

  @staticmethod
  def forward(ctx, *args, **kwargs):
    if len(args) >= 2:
      input_tensor = args[0]
      kernel_tensor = args[1]
      bias_tensor = args[2] if len(args) > 2 else kwargs.get("bias_tensor", None)
      stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
      padding = args[4] if len(args) > 4 else kwargs.get("padding", 0)
      groups = args[5] if len(args) > 5 else kwargs.get("groups", 1)
    else:
      input_tensor = kwargs["input_tensor"]
      kernel_tensor = kwargs["kernel_tensor"]
      bias_tensor = kwargs.get("bias_tensor", None)
      stride = kwargs.get("stride", 1)
      padding = kwargs.get("padding", 0)
      groups = kwargs.get("groups", 1)

    ctx.stride = stride
    ctx.padding = padding
    ctx.groups = groups
    ctx.has_bias = bias_tensor is not None

    kernel_size = kernel_tensor.shape[2]
    patch_columns, output_height, output_width = _im2col(
      input_tensor, kernel_size, stride, padding
    )

    ctx.save_for_backward(
      patch_columns, kernel_tensor, input_tensor.shape, output_height, output_width
    )

    if groups == 1:
      output_matrix = np.matmul(
        kernel_tensor.reshape(kernel_tensor.shape[0], -1), patch_columns
      )
    elif groups == input_tensor.shape[1]:
      batch_size, input_channels, _, _ = input_tensor.shape
      kernel_elements = kernel_size * kernel_size
      reshaped_columns = patch_columns.reshape(
        batch_size, input_channels, kernel_elements, -1
      )
      output_matrix = (
        reshaped_columns * kernel_tensor.reshape(input_channels, kernel_elements, 1)
      ).sum(2)
    else:
      batch_size, input_channels, _, _ = input_tensor.shape
      output_channels = kernel_tensor.shape[0]
      input_channels_per_group = input_channels // groups
      output_channels_per_group = output_channels // groups
      kernel_elements = kernel_size * kernel_size
      reshaped_columns = patch_columns.reshape(
        batch_size,
        groups,
        input_channels_per_group * kernel_elements,
        -1,
      )
      reshaped_kernel = kernel_tensor.reshape(
        groups,
        output_channels_per_group,
        input_channels_per_group * kernel_elements,
      )
      output_matrix = np.einsum(
        "bgmn,gpm->bgpn", reshaped_columns, reshaped_kernel
      ).reshape(batch_size, output_channels, -1)

    if bias_tensor is not None:
      output_matrix += bias_tensor.reshape(1, -1, 1)

    return output_matrix.reshape(
      input_tensor.shape[0], kernel_tensor.shape[0], output_height, output_width
    )

  def backward(self, gradient_output):
    patch_columns, kernel_tensor, input_shape, output_height, output_width = (
      self.saved_tensors
    )
    stride, padding, has_bias = self.stride, self.padding, self.has_bias
    batch_size = input_shape[0]
    gradient_output_reshaped = gradient_output.reshape(
      batch_size, kernel_tensor.shape[0], -1
    )
    gradient_kernel = np.einsum(
      "bop,bkp->ok", gradient_output_reshaped, patch_columns
    ).reshape(kernel_tensor.shape)
    gradient_bias = gradient_output_reshaped.sum(axis=(0, 2)) if has_bias else None
    gradient_patch_columns = np.einsum(
      "ko,bop->bkp",
      kernel_tensor.reshape(kernel_tensor.shape[0], -1).T,
      gradient_output_reshaped,
    )
    gradient_input = _col2im(
      gradient_patch_columns,
      input_shape,
      kernel_tensor.shape[2],
      stride,
      padding,
      output_height,
      output_width,
    )
    return (
      (gradient_input, gradient_kernel, gradient_bias)
      if has_bias
      else (gradient_input, gradient_kernel)
    )


def conv2d(
  input_tensor: Tensor,
  kernel_tensor: Tensor,
  bias_tensor: Optional[Tensor] = None,
  stride: int = 1,
  padding: int = 0,
  groups: int = 1,
):
  return Conv2d.apply(input_tensor, kernel_tensor, bias_tensor, stride, padding, groups)


def _to_tuple(x, dims: int) -> tuple[int, ...]:
  if isinstance(x, Sequence):
    if len(x) != dims:
      raise ValueError(f"Expected {dims} values, got {len(x)}.")
    return tuple(int(v) for v in x)
  return (int(x),) * dims


def _im2col_nd(
  input_image: np.ndarray,
  kernel_size,
  stride=1,
  padding=0,
):
  """
  General N‑D im2col.
  input_image shape: (B, C, D1, …, Dn)
  Returns
      columns      –  (B, C * ∏k_i, ∏out_i)
      output_shape –  tuple(out_1, …, out_n)
  """
  B, C, *spatial = input_image.shape
  dims = len(spatial)

  kernel_size = _to_tuple(kernel_size, dims)
  stride = _to_tuple(stride, dims)
  padding = _to_tuple(padding, dims)

  out_shape = tuple(
    (spatial[d] + 2 * padding[d] - kernel_size[d]) // stride[d] + 1 for d in range(dims)
  )

  pad_spec = [(0, 0), (0, 0)] + [(p, p) for p in padding]
  padded = np.pad(input_image, pad_spec)

  patches = sliding_window_view(
    padded,
    window_shape=kernel_size,
    axis=tuple(range(2, 2 + dims)),
  )
  slicing = (
    [slice(None), slice(None)]
    + [slice(None, None, stride[d]) for d in range(dims)]
    + [slice(None)] * dims
  )
  patches = patches[tuple(slicing)]

  order = [0, 1] + list(range(2 + dims, 2 + 2 * dims)) + list(range(2, 2 + dims))
  patches = patches.transpose(order)

  k_elems = np.prod(kernel_size)
  out_elems = np.prod(out_shape)
  columns = patches.reshape(B, C * k_elems, out_elems)
  return columns, out_shape


def _col2im_nd(
  columns: np.ndarray,
  input_shape: tuple[int, ...],
  kernel_size,
  stride,
  padding,
  output_shape: tuple[int, ...],
):
  """
  Inverse of _im2col_nd.
  columns shape: (B, C * ∏k_i, ∏out_i)
  Returns reconstructed input tensor (un‑padded).
  """
  B, C, *spatial = input_shape
  dims = len(spatial)

  kernel_size = _to_tuple(kernel_size, dims)
  stride = _to_tuple(stride, dims)
  padding = _to_tuple(padding, dims)

  k_elems = np.prod(kernel_size)
  columns = columns.reshape(B, C, *kernel_size, *output_shape)

  padded_shape = [s + 2 * p for s, p in zip(spatial, padding)]
  padded = np.zeros((B, C, *padded_shape), dtype=columns.dtype)

  for k_idx in np.ndindex(*kernel_size):
    out_slices = tuple(
      slice(k_idx[d], k_idx[d] + stride[d] * output_shape[d], stride[d])
      for d in range(dims)
    )
    padded[(slice(None), slice(None)) + out_slices] += columns[
      (slice(None), slice(None)) + k_idx + (Ellipsis,)
    ]

  crop_slices = tuple(slice(p, p + spatial[d]) for d, p in enumerate(padding))
  return padded[(slice(None), slice(None)) + crop_slices]


class ConvNd(Function):
  @staticmethod
  def forward(
    ctx,
    input_array: np.ndarray,
    kernel_array: np.ndarray,
    bias_array: np.ndarray | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    groups: int = 1,
  ):
    batch_size, channels_in, *spatial_shape = input_array.shape
    channels_out = kernel_array.shape[0]
    kernel_size = kernel_array.shape[2:]
    kernel_elements = int(np.prod(kernel_size))

    if isinstance(stride, int):
      stride = (stride,) * len(spatial_shape)
    if isinstance(padding, int):
      padding = (padding,) * len(spatial_shape)

    assert channels_in % groups == 0 and channels_out % groups == 0

    columns, output_spatial_shape = _im2col_nd(
      input_array, kernel_size, stride=stride, padding=padding
    )

    if groups == 1:
      output = np.matmul(kernel_array.reshape(channels_out, -1), columns)
    elif groups == channels_in:
      columns_r = columns.reshape(batch_size, channels_in, kernel_elements, -1)
      output = (
        columns_r * kernel_array.reshape(channels_in, 1, kernel_elements, 1)
      ).sum(2)
    else:
      channels_in_group = channels_in // groups
      channels_out_group = channels_out // groups
      columns_r = columns.reshape(
        batch_size, groups, channels_in_group * kernel_elements, -1
      )
      kernel_r = kernel_array.reshape(
        groups, channels_out_group, channels_in_group * kernel_elements
      )
      output = np.einsum("bgmn,gpm->bgpn", columns_r, kernel_r).reshape(
        batch_size, channels_out, -1
      )

    if bias_array is not None:
      output += bias_array.reshape(1, -1, 1)

    ctx.save_for_backward(
      input_array,
      kernel_array,
      bias_array,
      columns,
      np.array(output_spatial_shape),
      np.array(stride),
      np.array(padding),
      np.array(groups),
    )
    return output.reshape(batch_size, channels_out, *output_spatial_shape)

  def backward(self, gradient_output):
    (
      input_array,
      kernel_array,
      bias_array,
      columns,
      output_spatial_shape,
      stride,
      padding,
      groups,
    ) = self.saved_tensors
    stride = tuple(stride)
    padding = tuple(padding)
    groups = int(groups)

    batch_size, channels_in, *spatial_shape = input_array.shape
    channels_out = kernel_array.shape[0]
    kernel_size = kernel_array.shape[2:]
    kernel_elements = int(np.prod(kernel_size))

    gradient_output = gradient_output.reshape(batch_size, channels_out, -1)

    if groups == 1:
      gradient_kernel = np.einsum("bop,bkp->ok", gradient_output, columns).reshape(
        kernel_array.shape
      )
      gradient_columns = np.einsum(
        "ko,bop->bkp", kernel_array.reshape(channels_out, -1).T, gradient_output
      )
    elif groups == channels_in:
      columns_r = columns.reshape(batch_size, channels_in, kernel_elements, -1)
      gradient_kernel = (
        (gradient_output.reshape(batch_size, channels_in, 1, -1) * columns_r)
        .sum((0, 3))
        .reshape(kernel_array.shape)
      )
      gradient_columns = (
        kernel_array.reshape(channels_in, kernel_elements, 1)
        * gradient_output.reshape(batch_size, channels_in, 1, -1)
      ).reshape(batch_size, channels_in * kernel_elements, -1)
    else:
      channels_in_group = channels_in // groups
      channels_out_group = channels_out // groups
      columns_r = columns.reshape(
        batch_size, groups, channels_in_group * kernel_elements, -1
      )
      grad_out_r = gradient_output.reshape(batch_size, groups, channels_out_group, -1)

      gradient_kernel = np.einsum("bgop,bgkp->gok", grad_out_r, columns_r).reshape(
        kernel_array.shape
      )

      gradient_columns = np.einsum(
        "gok,bgop->bgkp",
        kernel_array.reshape(
          groups, channels_out_group, channels_in_group * kernel_elements
        ),
        grad_out_r,
      ).reshape(batch_size, channels_in * kernel_elements, -1)

    gradient_input = _col2im_nd(
      gradient_columns,
      input_array.shape,
      kernel_size,
      stride,
      padding,
      tuple(output_spatial_shape),
    )
    gradient_bias = gradient_output.sum(axis=(0, 2)) if bias_array is not None else None

    return (
      gradient_input,
      gradient_kernel,
      gradient_bias,
      None,
      None,
      None,
    )


def conv_nd(
  input_array: np.ndarray,
  kernel_array: np.ndarray,
  bias_array: np.ndarray | None = None,
  stride: int | Sequence[int] = 1,
  padding: int | Sequence[int] = 0,
  groups: int = 1,
):
  return ConvNd.apply(
    input_array,
    kernel_array,
    bias_array,
    stride=stride,
    padding=padding,
    groups=groups,
  )
