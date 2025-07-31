from os import RTLD_DEEPBIND
import numpy as np
import pytest
from numpy.random import randn

from src.core.ops import conv2d, conv_nd, matmul
from src.core.tensor import Tensor
from tests.utils import assertion, finite_difference_gradients


@pytest.mark.parametrize("n,m,k", [(4, 5, 6), (3, 7, 2)])
def test_matmul_grad(n, m, k):
  a = Tensor(randn(n, m).astype(np.float32), True)
  b = Tensor(randn(m, k).astype(np.float32), True)
  matmul(a, b).mean().backward()
  numdx, numdy = finite_difference_gradients(
    lambda x, y: (x @ y).mean(), [a.data.copy(), b.data.copy()]
  )
  assertion(a.grad.data, numdx)
  assertion(b.grad.data, numdy)


def _conv_2d_forward(inp, ker, bias, stride, padding):
  return conv2d(
    Tensor(inp),
    Tensor(ker),
    Tensor(bias) if bias is not None else None,
    stride,
    padding,
  ).data.mean()


@pytest.mark.parametrize("bias_flag", [True, False])
def test_conv2d_grad(bias_flag):
  batch_size, in_ch, out_ch, k, h, w = 2, 3, 4, 3, 5, 5
  inp = Tensor(randn(batch_size, in_ch, h, w).astype(np.float32), True)
  ker = Tensor(randn(out_ch, in_ch, k, k).astype(np.float32), True)
  bias = Tensor(randn(out_ch).astype(np.float32), True) if bias_flag else None
  conv2d(inp, ker, bias, 1, 1).mean().backward()
  if bias_flag:
    tensors = [inp.data.copy(), ker.data.copy(), bias.data.copy()]
    fd_fn = lambda i, k, b: _conv_2d_forward(i, k, b, 1, 1)
  else:
    tensors = [inp.data.copy(), ker.data.copy()]
    fd_fn = lambda i, k: _conv_2d_forward(i, k, None, 1, 1)
  grads = finite_difference_gradients(fd_fn, tensors)
  assertion(inp.grad.data, grads[0])
  assertion(ker.grad.data, grads[1])
  if bias_flag:
    assertion(bias.grad.data, grads[2])


def _conv_nd_forward(inp, ker, bias, stride, padding, groups):
  return conv_nd(
    Tensor(inp),
    Tensor(ker),
    Tensor(bias) if bias is not None else None,
    stride,
    padding,
    groups,
  ).data.mean()


# (dims, groups, stride, padding, bias_flag)
TEST_CASES = [
  (1, 1, 1, 0, True),
  (1, 1, 2, 1, False),
  (2, 1, 1, 1, True),
  (2, 2, 1, 0, False),
  (3, 1, 2, 0, True),
  (3, 3, 1, 1, False),
]


@pytest.mark.parametrize("dims,groups,stride,padding,bias_flag", TEST_CASES)
def test_conv_nd_grad(dims, groups, stride, padding, bias_flag):
  batch_size = 2
  channels_in = groups * 2
  channels_out = groups * 4
  kernel_size = 3
  spatial_extent = 5

  input_shape = (batch_size, channels_in) + (spatial_extent,) * dims
  kernel_shape = (channels_out, channels_in // groups) + (kernel_size,) * dims
  bias_shape = (channels_out,)

  inp = Tensor(randn(*input_shape).astype(np.float32), True)
  ker = Tensor(randn(*kernel_shape).astype(np.float32), True)
  bias = Tensor(randn(*bias_shape).astype(np.float32), True) if bias_flag else None

  conv_nd(inp, ker, bias, stride, padding, groups).mean().backward()

  if bias_flag:
    tensors = [inp.data.copy(), ker.data.copy(), bias.data.copy()]
    fd_fn = lambda i, k, b: _conv_nd_forward(i, k, b, stride, padding, groups)
  else:
    tensors = [inp.data.copy(), ker.data.copy()]
    fd_fn = lambda i, k: _conv_nd_forward(i, k, None, stride, padding, groups)

  gradients = finite_difference_gradients(fd_fn, tensors)
  assertion(inp.grad.data, gradients[0], 0.1, 0.1)
  assertion(ker.grad.data, gradients[1], 0.1, 0.1)
  if bias_flag:
    assertion(bias.grad.data, gradients[2], 0.1, 0.1)
