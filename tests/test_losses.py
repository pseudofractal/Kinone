import numpy as np
import pytest
from numpy.random import randn

from src.core.losses import (
  mean_squared_error,
  mean_absolute_error,
  binary_cross_entropy_with_logits,
  cross_entropy_with_logits,
)
from src.core.tensor import Tensor
from tests.utils import assertion, finite_difference_gradients


@pytest.mark.parametrize("shape", [(8, 3), (5, 7)])
def test_mse_grad(shape):
  preds = Tensor(randn(*shape).astype(np.float32), True)
  targets = Tensor(randn(*shape).astype(np.float32))
  _, grad = mean_squared_error(preds, targets)
  preds.backward(grad)
  (num_grad,) = finite_difference_gradients(
    lambda x: mean_squared_error(Tensor(x, False), targets)[0],
    [preds.data.copy()],
  )
  assertion(preds.grad.data, num_grad)


@pytest.mark.parametrize("shape", [(10, 4), (6, 9)])
def test_mae_grad(shape):
  preds = Tensor(randn(*shape).astype(np.float32), True)
  targets = Tensor(randn(*shape).astype(np.float32))
  _, grad = mean_absolute_error(preds, targets)
  preds.backward(grad)
  (num_grad,) = finite_difference_gradients(
    lambda x: mean_absolute_error(Tensor(x, False), targets)[0],
    [preds.data.copy()],
  )
  assertion(preds.grad.data, num_grad)


@pytest.mark.parametrize("batch,features", [(12, 5), (7, 3)])
def test_bce_with_logits_grad(batch, features):
  logits = Tensor(randn(batch, features).astype(np.float32), True)
  targets = Tensor((np.random.rand(batch, features) > 0.5).astype(np.float32))
  _, grad = binary_cross_entropy_with_logits(logits, targets)
  logits.backward(grad)
  (num_grad,) = finite_difference_gradients(
    lambda z: binary_cross_entropy_with_logits(Tensor(z, False), targets)[0],
    [logits.data.copy()],
  )
  assertion(logits.grad.data, num_grad)


@pytest.mark.parametrize("batch,classes", [(9, 4), (6, 8)])
def test_cross_entropy_with_logits_grad(batch, classes):
  logits = Tensor(randn(batch, classes).astype(np.float32), True)
  labels = Tensor(np.random.randint(0, classes, size=(batch,)))
  _, grad = cross_entropy_with_logits(logits, labels)
  logits.backward(grad)
  (num_grad,) = finite_difference_gradients(
    lambda z: cross_entropy_with_logits(Tensor(z, False), labels)[0],
    [logits.data.copy()],
  )
  assertion(logits.grad.data, num_grad)

