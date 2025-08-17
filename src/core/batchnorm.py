"""
Channel-wise Batch Normalisation.

Forward (training)
μᵢ = E[xᵢ],   σ²ᵢ = Var[xᵢ]
y = γ · (x − μ) / √(σ² + ε) + β

Running estimates updated with momentum for inference.
Backward gives dℓ/dx, dℓ/dγ, dℓ/dβ.
"""

import numpy as np

from .modules import Module
from .ops import Function
from .tensor import Tensor


class BatchNorm2dOp(Function):
  @staticmethod
  def forward(
    context,
    input,
    γ,
    β,
    running_μ,
    running_σ2,
    training,
    momentum=0.1,
    ε=1e-5,
  ):
    batch, channels, height, width = input.shape
    context.running_μ = running_μ
    context.running_σ2 = running_σ2
    context.ε = ε

    if training:
      batch_mean = input.mean(axis=(0, 2, 3), keepdims=True)
      batch_variance = input.var(axis=(0, 2, 3), keepdims=True)
      running_μ *= 1.0 - momentum
      running_μ += momentum * batch_mean.reshape(-1)
      running_σ2 *= 1.0 - momentum
      running_σ2 += momentum * batch_variance.reshape(-1)
    else:
      batch_mean = running_μ.reshape(1, channels, 1, 1)
      batch_variance = running_σ2.reshape(1, channels, 1, 1)

    standard_deviation = np.sqrt(batch_variance + ε)
    normalised_tensor = (input - batch_mean) / standard_deviation
    output = γ.reshape(1, channels, 1, 1) * normalised_tensor + β.reshape(
      1, channels, 1, 1
    )

    sample_count = batch * height * width
    context.save_for_backward(
      normalised_tensor,
      standard_deviation,
      γ,
      sample_count,
    )

    return output

  def backward(context, output_gradient):
    normalised_tensor, standard_deviation, γ, sample_count = context.saved_tensors

    γ_gradient = np.sum(output_gradient * normalised_tensor, axis=(0, 2, 3))
    β_gradient = np.sum(output_gradient, axis=(0, 2, 3))

    normalised_gradient = output_gradient * γ.reshape(1, -1, 1, 1)

    input_gradient = (1.0 / (sample_count * standard_deviation)) * (
      sample_count * normalised_gradient
      - np.sum(normalised_gradient, axis=(0, 2, 3), keepdims=True)
      - normalised_tensor
      * np.sum(
        normalised_gradient * normalised_tensor,
        axis=(0, 2, 3),
        keepdims=True,
      )
    )

    return (
      input_gradient,
      γ_gradient,
      β_gradient,
      None,
      None,
      None,
      None,
      None,
    )


def batch_norm_2d(
  input,
  γ,
  β,
  running_μ,
  running_σ2,
  training=True,
  momentum=0.1,
  ε=1e-5,
):
  return BatchNorm2dOp.apply(
    input,
    γ,
    β,
    running_μ,
    running_σ2,
    training,
    momentum,
    ε,
  )


class BatchNorm2d(Module):
  def __init__(self, channels, momentum=0.1, ε=1e-5):
    self.weight = Tensor(np.ones(channels, dtype=np.float32), requires_grad=True)
    self.bias = Tensor(np.zeros(channels, dtype=np.float32), requires_grad=True)
    self.running_μ = np.zeros(channels, dtype=np.float32)
    self.running_σ2 = np.ones(channels, dtype=np.float32)
    self.momentum = momentum
    self.ε = ε

  def __call__(self, input_tensor):
    return batch_norm_2d(
      input_tensor,
      self.weight,
      self.bias,
      self.running_μ,
      self.running_σ2,
      self.training,
      self.momentum,
      self.ε,
    )
