"""
Pooling operations implemented via numpy's stride-tricks.

∘ MaxPool2d  –  max over k×k windows, stride s, optional padding p.
Output shape Hₒ = ⌊(H+2p−k)/s⌋+1 (same for W).

Backward reconstructs the arg-max so memory doesn't explode.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .modules import Module
from .ops import Function


def _pair(value):
  return value if isinstance(value, tuple) else (value, value)


def _compute_pool_output_shape(input_height, input_width, kernel_size, stride, padding):
  kernel_height, kernel_width = _pair(kernel_size)
  stride_height, stride_width = _pair(stride)
  padding_height, padding_width = _pair(padding)

  output_height = (
    input_height + 2 * padding_height - kernel_height
  ) // stride_height + 1
  output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1
  return output_height, output_width


class MaxPool2dOp(Function):
  @staticmethod
  def forward(ctx, *args, **kwargs):
    input_tensor, kernel_size, stride, padding = args
    if not np.issubdtype(input_tensor.dtype, np.floating):
      input_tensor = input_tensor.astype(np.float32)

    batch_size, num_channels, input_height, input_width = input_tensor.shape

    kernel_height, kernel_width = _pair(kernel_size)
    stride_height, stride_width = _pair(stride)
    padding_height, padding_width = _pair(padding)

    output_height, output_width = _compute_pool_output_shape(
      input_height, input_width, kernel_size, stride, padding
    )

    if any((kernel_height, kernel_width)):
      negative_infinity = np.finfo(input_tensor.dtype).min
      padded_input = np.pad(
        input_tensor,
        (
          (0, 0),
          (0, 0),
          (padding_height, padding_height),
          (padding_width, padding_width),
        ),
        mode="constant",
        constant_values=negative_infinity,
      )
    else:
      padded_input = input_tensor

    shape = (
      batch_size,
      num_channels,
      output_height,
      output_width,
      kernel_height,
      kernel_width,
    )
    strides = (
      padded_input.strides[0],
      padded_input.strides[1],
      padded_input.strides[2] * stride_height,
      padded_input.strides[3] * stride_width,
      padded_input.strides[2],
      padded_input.strides[3],
    )

    windows = as_strided(padded_input, shape=shape, strides=strides)
    output_tensor = windows.max(axis=(-1, -2))
    flat_windows = windows.reshape(
      batch_size, num_channels, output_height, output_width, -1
    )
    max_indices = flat_windows.argmax(axis=-1)
    ctx.save_for_backward(
      max_indices,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      padding_height,
      padding_width,
      input_tensor.shape,
    )
    return output_tensor

  def backward(self, gradient_output):
    (
      max_indices,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      padding_height,
      padding_width,
      input_shape,
    ) = self.saved_tensors
    B, C, H_out, W_out = gradient_output.shape
    H_in, W_in = input_shape[2], input_shape[3]
    H_pad, W_pad = H_in + 2 * padding_height, W_in + 2 * padding_width

    grad_input = np.zeros((B, C, H_pad, W_pad), dtype=gradient_output.dtype)

    rows = (np.arange(H_out) * stride_height)[None, None, :, None]
    cols = (np.arange(W_out) * stride_width)[None, None, None, :]

    kr = max_indices // kernel_width
    kc = max_indices % kernel_width
    in_rows = rows + kr
    in_cols = cols + kc

    b_idx = np.arange(B)[:, None, None, None]
    c_idx = np.arange(C)[None, :, None, None]

    np.add.at(grad_input, (b_idx, c_idx, in_rows, in_cols), gradient_output)

    if any((padding_height, padding_width)):
      grad_input = grad_input[
        :,
        :,
        padding_height : H_pad - padding_height,
        padding_width : W_pad - padding_width,
      ]
    return grad_input


def max_pool2d(input_tensor, kernel_size=2, stride=2, padding=0):
  return MaxPool2dOp.apply(input_tensor, kernel_size, stride, padding)


class MaxPool2d(Module):
  def __init__(self, kernel_size=2, stride=2, padding=0):
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

  def __call__(self, input_tensor):
    return max_pool2d(input_tensor, self.kernel_size, self.stride, self.padding)


class AdaptiveAvgPool2dOp(Function):
  @staticmethod
  def forward(ctx, *args, **kwargs):
    input_tensor, output_size = args
    H_out, W_out = _pair(output_size)
    _, _, H_in, W_in = input_tensor.shape

    integral = np.pad(
      input_tensor.cumsum(axis=2).cumsum(axis=3),
      ((0, 0), (0, 0), (1, 0), (1, 0)),
      mode="constant",
    )

    row_start = (np.arange(H_out) * H_in) // H_out
    row_end = ((np.arange(H_out) + 1) * H_in) // H_out
    col_start = (np.arange(W_out) * W_in) // W_out
    col_end = ((np.arange(W_out) + 1) * W_in) // W_out

    r0, r1 = row_start, row_end
    c0, c1 = col_start, col_end

    summed = (
      integral[:, :, r1[:, None], c1]
      - integral[:, :, r0[:, None], c1]
      - integral[:, :, r1[:, None], c0]
      + integral[:, :, r0[:, None], c0]
    )

    row_len = row_end - row_start
    col_len = col_end - col_start
    area = row_len[:, None] * col_len

    output = summed / area

    ctx.save_for_backward(
      row_start,
      row_end,
      col_start,
      col_end,
      np.asarray(area, dtype=input_tensor.dtype),
      input_tensor.shape,
    )
    return output

  def backward(self, gradient_output):
    row_start, row_end, col_start, col_end, _, in_shape = self.saved_tensors

    row_len = row_end - row_start
    col_len = col_end - col_start

    g = np.repeat(gradient_output, row_len, axis=2)
    g = np.repeat(g, col_len, axis=3)

    row_scale = np.repeat(1.0 / row_len, row_len)
    col_scale = np.repeat(1.0 / col_len, col_len)

    grad_in = g * row_scale[None, None, :, None] * col_scale[None, None, None, :]
    return grad_in


def adaptive_avg_pool2d(input_tensor, output_size):
  return AdaptiveAvgPool2dOp.apply(input_tensor, output_size)


class AdaptiveAvgPool2d(Module):
  def __init__(self, output_size):
    self.output_size = output_size

  def __call__(self, input_tensor):
    return adaptive_avg_pool2d(input_tensor, self.output_size)
