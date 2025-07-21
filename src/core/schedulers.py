"""
Step-wise decay for any Optimizer.

Learning-rate schedule
ηₜ = η₀ · γ^{⌊t / step_size⌋}

∘ step() bumps an epoch counter and multiplies LR by γ whenever the
  counter hits a multiple of `step_size`.
"""

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
