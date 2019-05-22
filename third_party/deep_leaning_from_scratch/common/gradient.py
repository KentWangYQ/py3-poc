import numpy as np


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(x)

        for idx, x in enumerate(x):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad
