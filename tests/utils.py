import numpy as np

EPS = 1e-3
RTOL = 1e-2
ATOL = 1e-4


def assertion(
  computed: np.ndarray,
  numerical: np.ndarray,
  rtol: float = RTOL,
  atol: float = ATOL,
  name: str = "",
) -> None:
  """
  Assert ‖computed − numerical‖ ≤ atol + rtol·|numerical| element-wise.
  On failure raises a single-line AssertionError pointing to the worst element.
  """
  diff = np.abs(computed - numerical)
  tol = atol + rtol * np.abs(numerical)
  mask = diff > tol
  if mask.any():
    worst = np.unravel_index(np.argmax(diff / tol), diff.shape)
    raise AssertionError(
      f"{name} - grad mismatch at {worst} : "
      f"Computed={computed[worst]:.6g}, Numerical={numerical[worst]:.6g}, "
      f"Absolute Error={diff[worst]:.6g}, "
      f"Relative Error={(diff[worst] / (np.abs(numerical[worst]) + 1e-20)):.6g}, "
      f"Allowed Absolute={atol}, Relative={rtol}, "
      f"Computed Shape={computed.shape}, "
      f"Numerical Shape={numerical.shape}"
    )


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
