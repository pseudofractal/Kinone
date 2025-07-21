import numpy as np
import pytest
from numpy.random import randn

from src.core.ops import (
  hard_sigmoid,
  hard_swish,
  hard_tanh,
  neg,
  relu,
  sigmoid,
  swish,
  tanh,
)
from src.core.tensor import Tensor
from tests.utils import assertion, finite_difference_gradients


@pytest.mark.parametrize("shape", [(5, 5), (7, 3)])
def test_neg_grad(shape):
  a = Tensor(randn(*shape).astype(np.float32), True)
  neg(a).mean().backward()
  (numdx,) = finite_difference_gradients(lambda x: (-x).mean(), [a.data.copy()])
  assertion(a.grad.data, numdx)


@pytest.mark.parametrize("shape", [(5, 5), (7, 3)])
def test_relu_grad(shape):
  a = Tensor(randn(*shape).astype(np.float32), True)
  relu(a).mean().backward()
  (numdx,) = finite_difference_gradients(
    lambda x: np.maximum(x, 0).mean(), [a.data.copy()]
  )
  assertion(a.grad.data, numdx)


@pytest.mark.parametrize("shape", [(5, 5), (7, 3)])
def test_hard_sigmoid_grad(shape):
  input_tensor = Tensor(randn(*shape).astype(np.float32), True)
  hard_sigmoid(input_tensor).mean().backward()
  (numerical_gradient,) = finite_difference_gradients(
    lambda x: np.clip((x + 1.0) * 0.5, 0.0, 1.0).mean(),
    [input_tensor.data.copy()],
  )
  assertion(input_tensor.grad.data, numerical_gradient)


@pytest.mark.parametrize("shape", [(5, 5), (7, 3)])
def test_hard_tanh_grad(shape):
  input_tensor = Tensor(randn(*shape).astype(np.float32), True)
  hard_tanh(input_tensor).mean().backward()
  (numerical_gradient,) = finite_difference_gradients(
    lambda x: np.clip(x, -1.0, 1.0).mean(),
    [input_tensor.data.copy()],
  )
  assertion(input_tensor.grad.data, numerical_gradient)


@pytest.mark.parametrize("shape", [(5, 5), (7, 3)])
def test_hard_swish_grad(shape):
  input_tensor = Tensor(randn(*shape).astype(np.float32), True)
  hard_swish(input_tensor).mean().backward()
  (numerical_gradient,) = finite_difference_gradients(
    lambda x: (x * np.clip((x + 3.0) / 6.0, 0.0, 1.0)).mean(),
    [input_tensor.data.copy()],
  )
  assertion(input_tensor.grad.data, numerical_gradient)


@pytest.mark.parametrize("shape", [(5, 5), (7, 3)])
def test_sigmoid_grad(shape):
  input_tensor = Tensor(randn(*shape).astype(np.float32), True)
  sigmoid(input_tensor).mean().backward()
  (numerical_gradient,) = finite_difference_gradients(
    lambda x: (1.0 / (1.0 + np.exp(-x))).mean(),
    [input_tensor.data.copy()],
  )
  assertion(input_tensor.grad.data, numerical_gradient)


@pytest.mark.parametrize("shape", [(5, 5), (7, 3)])
def test_tanh_grad(shape):
  input_tensor = Tensor(randn(*shape).astype(np.float32), True)
  tanh(input_tensor).mean().backward()
  (numerical_gradient,) = finite_difference_gradients(
    lambda x: np.tanh(x).mean(),
    [input_tensor.data.copy()],
  )
  assertion(input_tensor.grad.data, numerical_gradient)


@pytest.mark.parametrize("shape", [(5, 5), (7, 3)])
def test_swish_grad(shape):
  input_tensor = Tensor(randn(*shape).astype(np.float32), True)
  swish(input_tensor).mean().backward()
  (numerical_gradient,) = finite_difference_gradients(
    lambda x: (x / (1.0 + np.exp(-x))).mean(),
    [input_tensor.data.copy()],
  )
  assertion(input_tensor.grad.data, numerical_gradient)
