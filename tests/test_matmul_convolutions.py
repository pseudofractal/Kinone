import numpy as np
import pytest
from numpy.random import randn

from src.core.ops import conv2d, matmul
from src.core.tensor import Tensor
from tests.utils import assertion, finite_difference_gradients


@pytest.mark.parametrize("n,m,k", [(4, 5, 6), (3, 7, 2)])
def test_matmul_grad(n, m, k):
    a = Tensor(randn(n, m).astype(np.float32), True)
    b = Tensor(randn(m, k).astype(np.float32), True)
    matmul(a, b).mean().backward()
    numdx, numdy = finite_difference_gradients(lambda x, y: (x @ y).mean(), [a.data.copy(), b.data.copy()])
    assertion(a.grad.data, numdx)
    assertion(b.grad.data, numdy)


def _conv_forward(inp, ker, bias, stride, padding):
    return conv2d(Tensor(inp), Tensor(ker), Tensor(bias) if bias is not None else None, stride, padding).data.mean()


@pytest.mark.parametrize("bias_flag", [True, False])
def test_conv2d_grad(bias_flag):
    batch_size, in_ch, out_ch, k, h, w = 2, 3, 4, 3, 5, 5
    inp = Tensor(randn(batch_size, in_ch, h, w).astype(np.float32), True)
    ker = Tensor(randn(out_ch, in_ch, k, k).astype(np.float32), True)
    bias = Tensor(randn(out_ch).astype(np.float32), True) if bias_flag else None
    conv2d(inp, ker, bias, 1, 1).mean().backward()
    if bias_flag:
        tensors = [inp.data.copy(), ker.data.copy(), bias.data.copy()]
        fd_fn = lambda i, k, b: _conv_forward(i, k, b, 1, 1)
    else:
        tensors = [inp.data.copy(), ker.data.copy()]
        fd_fn = lambda i, k: _conv_forward(i, k, None, 1, 1)
    grads = finite_difference_gradients(fd_fn, tensors)
    assertion(inp.grad.data, grads[0])
    assertion(ker.grad.data, grads[1])
    if bias_flag:
        assertion(bias.grad.data, grads[2])

