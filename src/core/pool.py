"""
Pooling operations implemented via numpy's stride-tricks.

∘ MaxPool2d  –  max over k×k windows, stride s, optional padding p.
Output shape Hₒ = ⌊(H+2p−k)/s⌋+1 (same for W).

Backward reconstructs the arg-max so memeory doesn't explode.
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

from .modules import Module
from .ops import Function


def _compute_pool_output_shape(input_height, input_width, kernel_size, stride, padding):
  output_height = (input_height + 2 * padding - kernel_size) // stride + 1
  output_width = (input_width + 2 * padding - kernel_size) // stride + 1
  return output_height, output_width


class MaxPool2dOp(Function):
  @staticmethod
  def forward(context, input_tensor, kernel_size, stride, padding):
    batch_size, num_channels, input_height, input_width = input_tensor.shape
    output_height, output_width = _compute_pool_output_shape(
      input_height, input_width, kernel_size, stride, padding
    )

    if padding > 0:
      padded_input = np.pad(
        input_tensor, ((0, 0), (0, 0), (padding, padding), (padding, padding))
      )
    else:
      padded_input = input_tensor

    shape = (
      batch_size,
      num_channels,
      output_height,
      output_width,
      kernel_size,
      kernel_size,
    )
    strides = (
      padded_input.strides[0],
      padded_input.strides[1],
      padded_input.strides[2] * stride,
      padded_input.strides[3] * stride,
      padded_input.strides[2],
      padded_input.strides[3],
    )

    windows = as_strided(padded_input, shape=shape, strides=strides)
    windows_view = windows.reshape(
      batch_size, num_channels, output_height, output_width, -1
    )
    output_tensor = windows_view.max(axis=4)
    max_indices = windows_view.argmax(axis=4)

    context.save_for_backward(
      max_indices, kernel_size, stride, padding, input_tensor.shape
    )
    return output_tensor

  def backward(ctx, grad_out):
    max_indices, ksize, stride, pad, input_shape = ctx.saved_tensors
    B, C, H_out, W_out = grad_out.shape
    H_in, W_in = input_shape[2], input_shape[3]
    H_pad, W_pad = H_in + 2 * pad, W_in + 2 * pad

    grad_input = np.zeros((B, C, H_pad, W_pad), dtype=grad_out.dtype)

    rows = (np.arange(H_out) * stride)[None, None, :, None]
    cols = (np.arange(W_out) * stride)[None, None, None, :]

    kr = max_indices // ksize
    kc = max_indices % ksize
    in_rows = rows + kr
    in_cols = cols + kc

    b_idx = np.arange(B)[:, None, None, None]
    c_idx = np.arange(C)[None, :, None, None]

    np.add.at(grad_input, (b_idx, c_idx, in_rows, in_cols), grad_out)

    if pad > 0:
      grad_input = grad_input[:, :, pad:-pad, pad:-pad]
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
  def forward(context, input_tensor, output_size):
    context.input_shape = input_tensor.shape
    return np.mean(input_tensor, axis=(2, 3), keepdims=True)

  def backward(self, grad_output):
    batch_size, num_channels, height, width = self.input_shape
    return grad_output / (height * width)


def adaptive_avg_pool2d(input_tensor, output_size):
  return AdaptiveAvgPool2dOp.apply(input_tensor, output_size)


class AdaptiveAvgPool2d(Module):
  def __init__(self, output_size):
    self.output_size = output_size

  def __call__(self, input_tensor):
    return adaptive_avg_pool2d(input_tensor, self.output_size)
