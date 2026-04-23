import numpy as np
from numpy.typing import NDArray


def mandelbrot_set(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    max_iter: int = 100,
) -> NDArray[np.int_]:
    """Compute the Mandelbrot escape-time array over a 2-D grid.

    For each point c = x + iy in the grid, iterate z_{n+1} = z_n^2 + c
    starting from z_0 = 0, and count how many iterations pass before
    |z| > 2 (escape). Points that never escape return ``max_iter``.

    Parameters
    ----------
    x_min : float
        Left edge of the real axis.
    x_max : float
        Right edge of the real axis.
    y_min : float
        Bottom edge of the imaginary axis.
    y_max : float
        Top edge of the imaginary axis.
    width : int
        Number of columns (pixels) in the output grid.
    height : int
        Number of rows (pixels) in the output grid.
    max_iter : int, optional
        Maximum number of iterations before a point is considered
        inside the set. Default is 100.

    Returns
    -------
    NDArray[np.int_]
        2-D array of shape ``(height, width)`` where each element is
        the escape-time iteration count for the corresponding point.
    """
    x: NDArray[np.float64] = np.linspace(x_min, x_max, width)
    y: NDArray[np.float64] = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C: NDArray[np.complex128] = X + 1j * Y
    Z: NDArray[np.complex128] = np.zeros_like(C)
    M: NDArray[np.int_] = np.zeros((height, width), dtype=int)
    for _n in range(max_iter):
        mask: NDArray[np.bool_] = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] += 1
    return M