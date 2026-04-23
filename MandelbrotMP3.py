import numpy as np

def mandelbrot_set(x_min, x_max, y_min, y_max, width, height, max_iter=100): 
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.zeros((height, width), dtype=int)
    for n in range(max_iter):        
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] += 1
    return M