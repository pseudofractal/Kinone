import numpy as np

from src.core.ops import add, mul
from src.core.optim import SGD, Adam, RMSProp
from src.core.tensor import Tensor
from tests.utils import ATOL


def create_parameter(value: np.ndarray | list[float]) -> Tensor:
  parameter = Tensor(np.asarray(value, dtype=np.float32), requires_grad=True)
  parameter.grad = Tensor(np.zeros_like(parameter.data))
  return parameter


def set_gradient(parameter: Tensor, gradient: np.ndarray | list[float]) -> None:
  parameter.grad.data[:] = np.asarray(gradient, dtype=np.float32)


def test_adam_optimizer() -> None:
  weight = Tensor(np.array([2.0], dtype=np.float64), True)
  optimizer = Adam([weight], learning_rate=0.1)
  for _ in range(1000):
    optimizer.zero_grad()
    mul(add(weight, Tensor(-5.0)), add(weight, Tensor(-5.0))).backward()
    optimizer.step()
  assert np.isclose(weight.data[0], 5.0, atol=ATOL)


def test_adam_zero_gradients() -> None:
  parameter = create_parameter([1.234])
  set_gradient(parameter, [0.0])
  optimizer = Adam([parameter], learning_rate=0.1)
  optimizer.step()
  assert np.allclose(parameter.data, np.array([1.234], dtype=np.float32), atol=1e-6)



def test_adam_learning_rate_zero() -> None:
  parameter = create_parameter([1.0])
  set_gradient(parameter, [1.0])
  optimizer = Adam([parameter], learning_rate=0.0)
  optimizer.step()
  assert np.allclose(parameter.data, np.array([1.0], dtype=np.float32), atol=1e-6)


def test_adam_single_step_known_values() -> None:
  parameter = create_parameter([0.5, -0.5])
  set_gradient(parameter, [0.1, -0.2])
  lr, eps = 0.001, 1e-8
  optimizer = Adam([parameter], learning_rate=lr, β_1=0.0, β_2=0.0, ε=eps)
  optimizer.step()
  expected = np.array(
    [0.5 - lr * 0.1 / (abs(0.1) + eps), -0.5 - lr * -0.2 / (abs(-0.2) + eps)],
    dtype=np.float32,
  )
  assert np.allclose(parameter.data, expected, atol=1e-6)


def test_adam_multiple_parameters() -> None:
  p1, p2 = create_parameter([1.0]), create_parameter([-1.0])
  set_gradient(p1, [1.0])
  set_gradient(p2, [-1.0])
  lr, eps = 0.001, 1e-8
  optimizer = Adam([p1, p2], learning_rate=lr, β_1=0.0, β_2=0.0, ε=eps)
  optimizer.step()
  expected_p1 = np.array([1.0 - lr * 1.0 / (1.0 + eps)], dtype=np.float32)
  expected_p2 = np.array([-1.0 - lr * -1.0 / (1.0 + eps)], dtype=np.float32)
  assert np.allclose(p1.data, expected_p1, atol=1e-6)
  assert np.allclose(p2.data, expected_p2, atol=1e-6)


def test_adam_multi_dimensional() -> None:
  parameter = create_parameter([[1.0, -1.0], [2.0, -2.0]])
  set_gradient(parameter, [[0.5, -0.5], [1.0, -1.0]])
  lr, eps = 0.01, 1e-8
  optimizer = Adam([parameter], learning_rate=lr, β_1=0.0, β_2=0.0, ε=eps)
  optimizer.step()
  expected = np.array(
    [
      [1.0 - lr * 0.5 / (0.5 + eps), -1.0 - lr * -0.5 / (0.5 + eps)],
      [2.0 - lr * 1.0 / (1.0 + eps), -2.0 - lr * -1.0 / (1.0 + eps)],
    ],
    dtype=np.float32,
  )
  assert np.allclose(parameter.data, expected, atol=1e-6)


def test_sgd_basic() -> None:
  parameter = create_parameter([1.0, -2.0])
  set_gradient(parameter, [0.1, -0.2])
  optimizer = SGD([parameter], learning_rate=0.1)
  optimizer.step()
  expected_value = np.array([0.99, -1.98], dtype=np.float32)
  assert np.allclose(parameter.data, expected_value, atol=1e-6)


def test_sgd_with_momentum() -> None:
  parameter = create_parameter([1.0])
  optimizer = SGD([parameter], learning_rate=0.1, momentum=0.9)
  set_gradient(parameter, [1.0])
  optimizer.step()
  set_gradient(parameter, [1.0])
  optimizer.step()
  expected_value = 1.0 - 0.1 * 1.0 - 0.1 * 1.9
  assert np.allclose(parameter.data, expected_value, atol=1e-6)


def test_sgd_with_nesterov() -> None:
  parameter = create_parameter([1.0])
  optimizer = SGD([parameter], learning_rate=0.1, momentum=0.9, nesterov=True)
  set_gradient(parameter, [1.0])
  optimizer.step()
  set_gradient(parameter, [1.0])
  optimizer.step()
  expected_value = np.array([0.539], dtype=np.float32)
  assert np.allclose(parameter.data, expected_value, atol=1e-6)


def test_sgd_with_weight_decay() -> None:
  parameter = create_parameter([2.0])
  set_gradient(parameter, [0.0])
  optimizer = SGD([parameter], learning_rate=0.1, weight_decay=0.5)
  optimizer.step()
  expected_value = np.array([1.9], dtype=np.float32)
  assert np.allclose(parameter.data, expected_value, atol=1e-6)


def test_rmsprop_basic() -> None:
  parameter = create_parameter([1.0])
  set_gradient(parameter, [1.0])
  learning_rate, decay_rate, epsilon = 0.1, 0.9, 1e-8
  optimizer = RMSProp([parameter], learning_rate=learning_rate, ρ=decay_rate, ε=epsilon)
  optimizer.step()
  average_squared_gradient = (1 - decay_rate) * (1.0**2)
  expected_value = 1.0 - learning_rate * 1.0 / np.sqrt(
    average_squared_gradient + epsilon
  )
  assert np.allclose(parameter.data, expected_value, atol=1e-6)


def test_rmsprop_with_weight_decay() -> None:
  initial_value = 2.0
  parameter = create_parameter([initial_value])
  set_gradient(parameter, [0.0])
  learning_rate = 0.05
  weight_decay = 0.1
  decay_rate = 0.9
  optimizer = RMSProp(
    [parameter], learning_rate=learning_rate, weight_decay=weight_decay, ρ=decay_rate
  )
  optimizer.step()
  effective_gradient = weight_decay * initial_value
  average_squared_gradient = (1 - decay_rate) * (effective_gradient**2)
  expected_value = initial_value - learning_rate * effective_gradient / np.sqrt(
    average_squared_gradient
  )
  assert np.allclose(parameter.data, expected_value, atol=1e-6)
