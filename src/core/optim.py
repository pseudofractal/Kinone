"""
An optimizer is an algorithm that updates the parameters of a model,
to minimize the loss function during training.

Given:
∘ Parameters: θ
∘ Loss Function: ℒ(θ)
∘ Gradient: ∇(ℒ)
∘ Learning Rate: η

An optimizer updates the parameters using the gradient:
θ ←--- θ - η . ∇(ℒ)
"""

from typing import Iterable

import numpy as np

from .tensor import Tensor


class Optimizer:
  def __init__(self, parameters: Iterable[Tensor], learning_rate: float = 0.01):
    self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
    self.learning_rate = learning_rate

  def zero_grad(self):
    for parameter in self.parameters:
      parameter.grad = None

  def step(self):
    raise NotImplementedError


class Adam(Optimizer):
  """
  ADAM stands for Adaptive Moment Estimation.
  It is a first order gradient-based optimization algorithm,
  that combines advantages of Momentum and RMSProp.

  It is used a lot because it adapts the learning rate of each parameter individually,
  based on estimates of first moment (mean) and second moment (uncentered variance) of the gradients.

  Given:
  ∘ Parameters: θ ∈ ℝᵈ
  ∘ Gradients at time step `t`: gₜ = ∇ℒ(θₜ)
  ∘ Learning Rate: η
  ∘ Exponential Decay Rates: β₁, β₂ ∈ [0, 1)
  ∘ Small constant for numerical stability: ε > 0

  Momentum Estimation: mₜ = β₁ × mₜ₋₁ + (1 - β₁) × gₜ
  RMS Estimation: vₜ = β₂ × vₜ₋₁ + (1 - β₂) × gₜ²
  Bias Correction: Both mₜ and vₜ are biased towards 0 at initial steps as they start at 0.
  So we apply some correction.
  m̂ₜ = mₜ / (1 - β₁ᵗ)
  v̂ₜ = vₜ / (1 - β₂ᵗ)
  Paramater Update: θₜ₊₁ = θₜ - η × m̂ₜ / (√v̂ₜ + ε)

  Adam is powerful because looking at the various momentums before updating helps smooth updates and avoid oscillations.
  The intution is that:
  ∘ If a parameter had large gradients, its update will be scaled down (due to large vₜ) → prevents exploding gradients.
  ∘ If a gradient keeps pointing in the same direction → mₜ accumulates →reinforces in that direction.
  """

  def __init__(
    self,
    parameters: Iterable[Tensor],
    learning_rate: float = 0.01,
    β_1: float = 0.9,
    β_2: float = 0.999,
    ε: float = 1e-6,
    weight_decay: float = 0.0,
    maximum_gradient_norm: float | None = None,
  ):
    super().__init__(parameters, learning_rate)
    self.β_1 = β_1
    self.β_2 = β_2
    self.ε = ε
    self.weight_decay = weight_decay
    self.maximum_gradient_norm = maximum_gradient_norm

    self._step_count = 0
    self._first_moment = [
      np.zeros_like(parameter.data) for parameter in self.parameters
    ]
    self._second_moment = [
      np.zeros_like(parameter.data) for parameter in self.parameters
    ]

  def step(self):
    self._step_count += 1

    if self.maximum_gradient_norm is not None:
      all_grads = [p.grad.data for p in self.parameters if p.grad is not None]
      if all_grads:
        l2_norm = np.sqrt(sum(np.sum(g**2) for g in all_grads))
        if l2_norm > self.maximum_gradient_norm:
          clip_coefficient = self.maximum_gradient_norm / (l2_norm + self.ε)
          for p in self.parameters:
            if p.grad is not None:
              p.grad.data *= clip_coefficient

    t = self._step_count
    bias_corrected_learning_rate = (
      self.learning_rate * np.sqrt(1.0 - self.β_2**t) / (1.0 - self.β_1**t)
    )

    for parameters, first_moment, second_moment in zip(
      self.parameters, self._first_moment, self._second_moment
    ):
      if parameters.grad is None:
        continue
      grad = parameters.grad.data

      if grad.ndim > parameters.data.ndim:
        grad = grad.sum(axis=0)

      if self.weight_decay:
        parameters.data -= self.learning_rate * self.weight_decay * parameters.data

      first_moment[:] = self.β_1 * first_moment + (1 - self.β_1) * grad
      second_moment[:] = self.β_2 * second_moment + (1 - self.β_2) * (grad**2)

      corrected_first_moment = first_moment / (1 - self.β_1**self._step_count)
      corrected_second_moment = second_moment / (1 - self.β_2**self._step_count)

      parameters.data -= (
        bias_corrected_learning_rate
        * corrected_first_moment
        / (np.sqrt(corrected_second_moment) + self.ε)
      )


class SGD(Optimizer):
  """
  Stochastic Gradient Descent with optional momentum and Nesterov acceleration.

  θ_{t+1} = θ_t − η · v_t
  v_{t+1} = μ·v_t + g_t                          # plain momentum
          = μ·v_t + g_t                          # same buffer reused
          g_t can include weight‑decay term.

  If nesterov=True:
    v_{t+1} = μ·v_t + g_t
    θ_{t+1} = θ_t − η · (μ·v_{t+1} + g_t)
  """

  def __init__(
    self,
    parameters: Iterable[Tensor],
    learning_rate: float = 0.01,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
  ):
    super().__init__(parameters, learning_rate)
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.nesterov = nesterov
    self._velocity = [np.zeros_like(p.data) for p in self.parameters]

  def step(self):
    for p, v in zip(self.parameters, self._velocity):
      if p.grad is None:
        continue
      grad = p.grad.data
      if grad.ndim > p.data.ndim:
        grad = grad.sum(axis=0)
      if self.weight_decay:
        grad += self.weight_decay * p.data

      v *= self.momentum
      v += grad
      update = self.momentum * v + grad if self.nesterov else v
      p.data -= self.learning_rate * update


class RMSProp(Optimizer):
  """
  RMSProp — keeps a running average of squared gradients.

  E[g²]_t = ρ·E[g²]_{t‑1} + (1‑ρ)·g_t²
  θ_{t+1} = θ_t − η · g_t / (√(E[g²]_t) + ε)
  """

  def __init__(
    self,
    parameters: Iterable[Tensor],
    learning_rate: float = 0.001,
    ρ: float = 0.9,
    ε: float = 1e-8,
    weight_decay: float = 0.0,
  ):
    super().__init__(parameters, learning_rate)
    self.ρ = ρ
    self.ε = ε
    self.weight_decay = weight_decay
    self._average_square = [np.zeros_like(p.data) for p in self.parameters]

  def step(self):
    for parameter, average_square in zip(self.parameters, self._average_square):
      if parameter.grad is parameter:
        continue
      grad = parameter.grad.data
      if grad.ndim > parameter.data.ndim:
        grad = grad.sum(axis=0)
      if self.weight_decay:
        grad += self.weight_decay * parameter.data

      average_square *= self.ρ
      average_square += (1.0 - self.ρ) * (grad**2)
      parameter.data -= self.learning_rate * grad / (np.sqrt(average_square) + self.ε)


__all__ = ["Optimizer", "Adam", "SGD", "RMSProp"]
