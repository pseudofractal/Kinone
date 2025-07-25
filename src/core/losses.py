"""
This module aggregates differentiable loss functions required by a minimal
autograd engine. All routines operate on `.tensor.Tensor` wrappers around
`np.ndarray` data and return a tuple:

  (scalar_loss, gradient_tensor)

where `scalar_loss` is a Python float and `gradient_tensor` is a detached
`Tensor` containing ∂ℓ⁄∂input.

Implemented losses
• mean_squared_error            ℓ = 1⁄n ∑(ŷ − y)²
• mean_absolute_error           ℓ = 1⁄n ∑|ŷ − y|
• binary_cross_entropy_with_logits ℓ = −[y log σ(x)+(1−y) log(1−σ(x))]
• cross_entropy_with_logits     ℓ = −log σ(z)ᵧ

Numerical stability is enforced via log-sum-exp and clamped exponentials.
"""

import numpy as np

from .tensor import Tensor


def _sigmoid(x: np.ndarray) -> np.ndarray:
  """
  σ(x) = 1 ⁄ (1 + e^{−x})
  """
  positive_mask = x >= 0
  negative_mask = ~positive_mask
  exp_values = np.zeros_like(x, dtype=np.float32)
  exp_values[positive_mask] = np.exp(-x[positive_mask])
  exp_values[negative_mask] = np.exp(x[negative_mask])
  numerator = np.ones_like(x, dtype=np.float32)
  numerator[negative_mask] = exp_values[negative_mask]
  return numerator / (1.0 + exp_values)


def _softmax(z: np.ndarray) -> np.ndarray:
  """
  σ(z)ⱼ = exp(zⱼ − max(z)) ⁄ ∑ₖ exp(zₖ − max(z))
  """
  shifted = z - np.max(z, axis=1, keepdims=True)
  exp_shifted = np.exp(shifted)
  return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def mean_squared_error(predictions: Tensor, targets: Tensor) -> tuple[float, Tensor]:
  """
  ℓ(ŷ, y) = 1 ⁄ n ∑ᵢ (ŷᵢ − yᵢ)²
  ∂ℓ ⁄ ∂ŷ = 2 (ŷ − y) ⁄ n
  """
  difference = predictions.data - targets.data.astype(np.float32)
  loss_value = np.mean(difference**2).item()
  gradient_matrix = (2.0 * difference) / difference.size
  return loss_value, Tensor(gradient_matrix, requires_grad=False)


def mean_absolute_error(predictions: Tensor, targets: Tensor) -> tuple[float, Tensor]:
  """
  ℓ(ŷ, y) = 1 ⁄ n ∑ᵢ |ŷᵢ − yᵢ|
  ∂ℓ ⁄ ∂ŷ = sgn(ŷ − y) ⁄ n
  """
  difference = predictions.data - targets.data.astype(np.float32)
  loss_value = np.mean(np.abs(difference)).item()
  gradient_matrix = np.sign(difference) / difference.size
  return loss_value, Tensor(gradient_matrix, requires_grad=False)


def binary_cross_entropy_with_logits(
  logits: Tensor, targets: Tensor, pos_weight: float | np.ndarray | None = None
) -> tuple[float, Tensor]:
  """
  ℓ(x, y) = −[y · log σ(x) + (1 − y) · log(1 − σ(x))]
  ∂ℓ ⁄ ∂x = σ(x) − y
  """
  x_values = logits.data
  y_values = targets.data.astype(np.float32)
  if pos_weight is None:
    pos_weight = 1.0
  weight_matrix = y_values * pos_weight + (1.0 - y_values)
  max_values = np.clip(x_values, 0.0, None)
  unweighted = max_values - y_values * x_values + np.log1p(np.exp(-np.abs(x_values)))
  loss_matrix = weight_matrix * unweighted
  loss_value = loss_matrix.mean().item()
  gradient_matrix = weight_matrix * (_sigmoid(x_values) - y_values) / y_values.size
  return loss_value, Tensor(gradient_matrix, requires_grad=False)


def cross_entropy_with_logits(
  logits: Tensor, class_indices: Tensor
) -> tuple[float, Tensor]:
  """
  ℓ(z, y) = −log σ(z)ᵧ
  ∂ℓ ⁄ ∂z = (σ(z) − one_hot(y)) ⁄ n
  """
  z_values = logits.data
  y_indices = class_indices.data.astype(np.int64).reshape(-1)
  probabilities = _softmax(z_values)
  log_probabilities = -np.log(probabilities[np.arange(y_indices.size), y_indices])
  loss_value = log_probabilities.mean().item()
  one_hot_targets = np.zeros_like(probabilities, dtype=np.float32)
  one_hot_targets[np.arange(y_indices.size), y_indices] = 1.0
  gradient_matrix = (probabilities - one_hot_targets) / y_indices.size
  return loss_value, Tensor(gradient_matrix, requires_grad=False)
