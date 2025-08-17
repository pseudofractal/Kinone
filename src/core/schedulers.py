"""
Step-wise decay for any Optimizer.

Learning-rate schedule
ηₜ = η₀ · γ^{⌊t / step_size⌋}

∘ step() bumps an epoch counter and multiplies LR by γ whenever the
  counter hits a multiple of `step_size`.
"""

from numpy import cos
from numpy import pi as π

from .optim import Optimizer


class StepLR:
  """
  Decays the learning rate of each parameter group by gamma every step_size epochs.

  Args:
      optimizer (Optimizer): Wrapped optimizer.
      step_size (int): Period of learning rate decay.
      gamma (float): Multiplicative factor of learning rate decay.
  """

  def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
    self.optimizer = optimizer
    self.step_size = step_size
    self.gamma = gamma
    self.epoch = 0

  def step(self):
    self.epoch += 1
    if self.epoch % self.step_size == 0:
      self.optimizer.learning_rate *= self.gamma
      print(f"\n[INFO] Decaying learning rate to {self.optimizer.learning_rate:.6f}.")


class CosineAnnealingLR:
  """
  Sets the learning rate using a cosine annealing schedule.
  η_t = η_min + 0.5 * (η_initial - η_min) * (1 + cos(t / T_max * π))

  Args:
      optimizer (Optimizer): Wrapped optimizer.
      T_max (int): Maximum number of epochs.
      η_min (float): Minimum learning rate. Default: 0.
  """

  def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
    self.optimizer = optimizer
    self.T_max = T_max
    self.η_min = eta_min
    self.base_lr = optimizer.learning_rate
    self.last_epoch = 0

  def step(self):
    if self.last_epoch < self.T_max:
      self.last_epoch += 1
      new_lr = self.η_min + 0.5 * (self.base_lr - self.η_min) * (
        1 + cos(self.last_epoch / self.T_max * π)
      )
      self.optimizer.learning_rate = new_lr
    else:
      self.optimizer.learning_rate = self.η_min
