import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0: 
            return i
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)  # un-timed warm-up: Numba JIT in workers
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)


if __name__ == '__main__':
    # Parameters
    N = 1024
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25
    max_iter = 100
    n_workers = 4
    
    print("=" * 50)
    print("MANDELBROT PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Run parallel version
    print(f"\nRunning parallel version with {n_workers} workers...")
    t0 = time.perf_counter()
    parallel_result = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, 
                                          max_iter, n_workers=n_workers)
    t_parallel = time.perf_counter() - t0
    print(f"Parallel computation took: {t_parallel:.3f} seconds")
    
    # Run serial version
    print("\nRunning serial version...")
    t0 = time.perf_counter()
    serial_result = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    t_serial = time.perf_counter() - t0
    print(f"Serial computation took: {t_serial:.3f} seconds")
    
    
    # Visualize the result (using parallel_result)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(parallel_result, extent=[-2.5, 1.0, -1.25, 1.25],
              cmap='inferno', origin='lower', aspect='equal')
    ax.set_xlabel('Re(c)')
    ax.set_ylabel('Im(c)')
    ax.set_title(f'Mandelbrot Set (Parallel: {t_parallel:.3f}s, Serial: {t_serial:.3f}s)')
    out = Path(__file__).parent / 'mandelbrot.png'
    plt.savefig(out, dpi=150)
    print(f'\nSaved: {out}')
    plt.show()
    
    