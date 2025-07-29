import numpy as np
import pytest
from numpy.random import randn

from src.core.pool import adaptive_avg_pool2d, max_pool2d
from src.core.tensor import Tensor
from tests.utils import assertion, finite_difference_gradients


@pytest.mark.parametrize(
  "input_shape,kernel_size,stride,padding",
  [
    ((2, 3, 8, 8), 2, 2, 0),
    ((1, 4, 7, 5), 3, 1, 1),
  ],
)
def test_max_pool2d_forward_shape(input_shape, kernel_size, stride, padding):
  input_tensor = Tensor(randn(*input_shape).astype(np.float32), requires_grad=False)
  output_tensor = max_pool2d(
    input_tensor, kernel_size=kernel_size, stride=stride, padding=padding
  )
  expected_height = (input_shape[2] + 2 * padding - kernel_size) // stride + 1
  expected_width = (input_shape[3] + 2 * padding - kernel_size) // stride + 1
  assert output_tensor.shape == (
    input_shape[0],
    input_shape[1],
    expected_height,
    expected_width,
  )


@pytest.mark.parametrize(
  "input_shape,kernel_size,stride,padding",
  [
    ((2, 3, 6, 6), 2, 2, 0),
    ((1, 2, 5, 7), 3, 1, 1),
  ],
)
def test_max_pool2d_gradient(input_shape, kernel_size, stride, padding):
  input_tensor = Tensor(randn(*input_shape).astype(np.float32), requires_grad=True)
  loss = max_pool2d(
    input_tensor, kernel_size=kernel_size, stride=stride, padding=padding
  ).mean()
  loss.backward()
  (numerical_gradient,) = finite_difference_gradients(
    lambda x: max_pool2d(
      Tensor(x, requires_grad=False),
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
    ).mean(),
    [input_tensor.data.copy()],
  )
  assertion(input_tensor.grad.data, numerical_gradient)


@pytest.mark.parametrize(
  "input_shape,output_size",
  [
    ((2, 3, 8, 8), 1),
    ((1, 4, 7, 5), (2, 3)),
  ],
)
def test_adaptive_avg_pool2d_forward_shape(input_shape, output_size):
  input_tensor = Tensor(randn(*input_shape).astype(np.float32), requires_grad=False)
  output_tensor = adaptive_avg_pool2d(input_tensor, output_size=output_size)
  if isinstance(output_size, int):
    expected_size = (output_size, output_size)
  else:
    expected_size = output_size
  assert output_tensor.shape == (
    input_shape[0],
    input_shape[1],
    expected_size[0],
    expected_size[1],
  )


@pytest.mark.parametrize(
  "input_shape,output_size",
  [
    ((2, 3, 6, 6), 1),
    ((1, 2, 5, 7), (2, 3)),
  ],
)
def test_adaptive_avg_pool2d_gradient(input_shape, output_size):
  input_tensor = Tensor(randn(*input_shape).astype(np.float32), requires_grad=True)
  loss = adaptive_avg_pool2d(input_tensor, output_size=output_size).mean()
  loss.backward()
  (numerical_gradient,) = finite_difference_gradients(
    lambda x: adaptive_avg_pool2d(
      Tensor(x, requires_grad=False), output_size=output_size
    ).mean(),
    [input_tensor.data.copy()],
  )
  assertion(input_tensor.grad.data, numerical_gradient)
