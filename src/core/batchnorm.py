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
from .tensor import Tensor
from .ops import Function


class BatchNorm2dOp(Function):
  @staticmethod
  def forward(
    context, input, γ, β, running_μ, running_σ2, training, momentum=0.1, ε=1e-5
  ):
    _, channels, _, _ = input.shape
    if training:
      μ = input.mean(axis=(0, 2, 3), keepdims=True)
      σ2 = input.var(axis=(0, 2, 3), keepdims=True)
      running_μ *= 1 - momentum
      running_μ += momentum * μ.squeeze()
      running_σ2 *= 1 - momentum
      running_σ2 += momentum * σ2.squeeze()
    else:
      μ = running_μ.reshape(1, channels, 1, 1)
      σ2 = running_σ2.reshape(1, channels, 1, 1)

    σ = np.sqrt(σ2 + ε)
    normalized_input = (input - μ) / σ
    output = γ.reshape(1, channels, 1, 1) * normalized_input + β.reshape(
      1, channels, 1, 1
    )

    context.save_for_backward(input, μ, σ, γ, normalized_input)
    return output

  def backward(ctx, output_gradient):
    input, μ, σ, γ, normalized_input = ctx.saved_tensors
    batches, channels, height, width = output_gradient.shape
    num_elements = batches * height * width

    # Gradients of loss w.r.t. γ and β
    γ_gradient = np.sum(output_gradient * normalized_input, axis=(0, 2, 3))
    β_gradient = np.sum(output_gradient, axis=(0, 2, 3))

    # Gradient of loss w.r.t. the normalized input
    normalized_input_gradient = output_gradient * γ.reshape(1, channels, 1, 1)

    # Gradient of loss w.r.t. the input
    dx = (1.0 / (num_elements * σ)) * (
      num_elements * normalized_input_gradient
      - np.sum(normalized_input_gradient, axis=(0, 2, 3), keepdims=True)
      - normalized_input
      * np.sum(
        normalized_input_gradient * normalized_input, axis=(0, 2, 3), keepdims=True
      )
    )

    return dx, γ_gradient, β_gradient, None, None, None, None, None


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
  return BatchNorm2dOp.apply(input, γ, β, running_μ, running_σ2, training, momentum, ε)


class BatchNorm2d(Module):
  def __init__(self, channels, momentum=0.1, ε=1e-5):
    self.weight = Tensor(np.ones(channels, dtype=np.float32), True)
    self.bias = Tensor(np.zeros(channels, dtype=np.float32), True)
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
