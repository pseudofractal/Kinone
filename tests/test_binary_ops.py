import numpy as np
import pytest
from numpy.random import randn

from src.core.ops import add, div, mul, sub
from src.core.tensor import Tensor
from tests.utils import assertion, finite_difference_gradients


@pytest.mark.parametrize("shape", [(4, 4), (3, 5)])
def test_add_grad(shape):
  a, b = (Tensor(randn(*shape).astype(np.float32), True) for _ in range(2))
  add(a, b).mean().backward()
  numdx, numdy = finite_difference_gradients(
    lambda x, y: (x + y).mean(), [a.data.copy(), b.data.copy()]
  )
  assertion(a.grad.data, numdx)
  assertion(b.grad.data, numdy)


@pytest.mark.parametrize("shape", [(4, 4), (3, 5)])
def test_sub_grad(shape):
  a, b = (Tensor(randn(*shape).astype(np.float32), True) for _ in range(2))
  sub(a, b).mean().backward()
  numdx, numdy = finite_difference_gradients(
    lambda x, y: (x - y).mean(), [a.data.copy(), b.data.copy()]
  )
  assertion(a.grad.data, numdx)
  assertion(b.grad.data, numdy)


@pytest.mark.parametrize("shape", [(4, 4), (3, 5)])
def test_mul_grad(shape):
  a, b = (Tensor(randn(*shape).astype(np.float32), True) for _ in range(2))
  mul(a, b).mean().backward()
  numdx, numdy = finite_difference_gradients(
    lambda x, y: (x * y).mean(), [a.data.copy(), b.data.copy()]
  )
  assertion(a.grad.data, numdx)
  assertion(b.grad.data, numdy)


@pytest.mark.parametrize("shape", [(4, 4), (3, 5)])
def test_div_grad(shape):
  a, b = (Tensor(randn(*shape).astype(np.float32), True) for _ in range(2))
  div(a, b).mean().backward()
  numdx, numdy = finite_difference_gradients(
    lambda x, y: (x / y).mean(), [a.data.copy(), b.data.copy()]
  )
  assertion(a.grad.data, numdx)
  assertion(b.grad.data, numdy)


@pytest.mark.parametrize("shape_a,shape_b", [((3, 1), (1, 4)), ((2, 1, 5), (1, 4, 5))])
def test_broadcast_grad(shape_a, shape_b):
  a = Tensor(randn(*shape_a).astype(np.float32), True)
  b = Tensor(randn(*shape_b).astype(np.float32), True)
  add(a, b).sum().backward()
  numdx, numdy = finite_difference_gradients(
    lambda x, y: (x + y).sum(), [a.data.copy(), b.data.copy()]
  )
  assertion(a.grad.data, numdx)
  assertion(b.grad.data, numdy)
