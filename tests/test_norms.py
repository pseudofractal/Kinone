import numpy as np
import pytest
from numpy.random import randn

from src.core.batchnorm import BatchNorm2d
from src.core.tensor import Tensor
from tests.utils import assertion, finite_difference_gradients


@pytest.mark.parametrize(
  "input_shape",
  [
    (2, 3, 8, 8),
    (4, 1, 5, 7),
  ],
)
def test_batchnorm2d_forward_shape(input_shape):
  tensor = Tensor(randn(*input_shape).astype(np.float32), requires_grad=False)
  bn = BatchNorm2d(input_shape[1])
  bn.training = True
  out = bn(tensor)
  assert out.shape == input_shape


@pytest.mark.parametrize(
  "input_shape",
  [
    (3, 2, 6, 6),
    (1, 4, 5, 5),
  ],
)
def test_batchnorm2d_normalization(input_shape):
  tensor = Tensor(randn(*input_shape).astype(np.float32), requires_grad=False)
  bn = BatchNorm2d(input_shape[1])
  bn.training = True
  out = bn(tensor).data
  channel_mean = out.mean(axis=(0, 2, 3))
  channel_var = out.var(axis=(0, 2, 3))
  np.testing.assert_allclose(channel_mean, 0.0, atol=1e-5, rtol=0)
  np.testing.assert_allclose(channel_var, 1.0, atol=1e-4, rtol=0)


def test_batchnorm2d_running_stats_update():
  tensor = Tensor(randn(4, 3, 4, 4).astype(np.float32), requires_grad=False)
  bn = BatchNorm2d(3)
  prev_mu = bn.running_μ.copy()
  prev_var = bn.running_σ2.copy()
  bn.training = True
  bn(tensor)
  assert not np.allclose(bn.running_μ, prev_mu)
  assert not np.allclose(bn.running_σ2, prev_var)


@pytest.mark.parametrize(
  "input_shape",
  [
    (2, 3, 4, 4),
    (1, 5, 3, 3),
  ],
)
def test_batchnorm2d_input_gradient(input_shape):
  tensor = Tensor(randn(*input_shape).astype(np.float32), requires_grad=True)
  bn = BatchNorm2d(input_shape[1])
  bn.training = True
  loss = bn(tensor).mean()
  loss.backward()
  (num_grad_input,) = finite_difference_gradients(
    lambda x: BatchNorm2d(input_shape[1])(Tensor(x, requires_grad=False)).mean(),
    [tensor.data.copy()],
  )
  assertion(tensor.grad.data, num_grad_input)

