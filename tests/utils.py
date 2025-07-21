import numpy as np

EPS = 1e-3
RTOL = 1e-2
ATOL = 1e-4


def assertion(computed, numerical, rtol=RTOL, atol=ATOL, name=""):
  try:
    np.testing.assert_allclose(computed, numerical, rtol, atol)
  except AssertionError as err:
    print(f"{name}∘Computed:\n", computed.shape)
    print(f"{name}∘Numerical:\n", numerical.shape)
    raise err


def finite_difference_gradients(f, inputs, eps=EPS):
  """Finite-difference gradient for scalar-output func(*inputs)."""
  gradients = []
  for input_tensor in inputs:
    gradient = np.zeros_like(input_tensor)
    iterator = np.nditer(input_tensor, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
      idx = iterator.multi_index
      original_value = input_tensor[idx]
      input_tensor[idx] = original_value + eps
      f_plus = f(*inputs)
      input_tensor[idx] = original_value - eps
      f_minus = f(*inputs)
      input_tensor[idx] = original_value
      gradient[idx] = (f_plus - f_minus) / (2 * eps)
      iterator.iternext()
    gradients.append(gradient)
  return gradients


