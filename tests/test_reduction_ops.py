import numpy as np
import pytest
from numpy.random import randn

from src.core.ops import mean as _mean
from src.core.ops import sum as _sum
from src.core.tensor import Tensor
from tests.utils import assertion, finite_difference_gradients


@pytest.mark.parametrize("shape,axis", [((3, 4, 5), 1), ((2, 3, 6), 2)])
def test_sum_grad(shape, axis):
  a = Tensor(randn(*shape).astype(np.float32), True)
  _sum(a, axis=axis).mean().backward()
  (numdx,) = finite_difference_gradients(
    lambda x: x.sum(axis=axis).mean(), [a.data.copy()]
  )
  assertion(a.grad.data, numdx)


@pytest.mark.parametrize("shape,axis", [((3, 4, 5), (0, 2)), ((2, 3, 6), (1,))])
def test_mean_grad(shape, axis):
  a = Tensor(randn(*shape).astype(np.float32), True)
  _mean(a, axis=axis).sum().backward()
  (numdx,) = finite_difference_gradients(
    lambda x: x.mean(axis=axis).sum(), [a.data.copy()]
  )
  assertion(a.grad.data, numdx)
