import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path

from dask import delayed
from dask.distributed import Client, LocalCluster
import dask



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

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = n_workers
        
    # Calculate chunks based on n_chunks (not chunk_size directly)
    chunks = []
    rows_per_chunk = N / n_chunks
    
    # Create exactly n_chunks chunks
    for i in range(n_chunks):
        row_start = int(i * rows_per_chunk)
        row_end = int((i + 1) * rows_per_chunk) if i < n_chunks - 1 else N
        if row_end > row_start:  # Ensure non-empty chunk
            chunks.append((row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter))
    
    if pool is not None:  # caller manages Pool; skip startup + warm-up
        return np.vstack(pool.map(_worker, chunks))
    
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny)  # warm-up: load JIT cache in workers
        parts = p.map(_worker, chunks)
    return np.vstack(parts)

def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)

if __name__ == "__main__":
    N, max_iter = 1024, 100
    n_workers = 8
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # Warm up JIT locally
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # Measure serial baseline (T₁) - NOT using Dask
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial (T₁): {t_serial:.3f}s")

    # Create cluster for sweep (keep open for all measurements)
    #cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    #client = Client(cluster)
    
    # Connect to your existing remote scheduler
    client = Client("tcp://10.92.0.39:8786")
    
    # Warm up all workers
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))
    
    # Sweep over chunk sizes
    n_chunks_values = [8, 16, 32, 64, 128, 256, 512]
    times_list = []
    lif_list = []
    
    print("\nn_chunks | time (s) | speedup | LIF")
    print("-" * 45)
    
    for n_chunks in n_chunks_values:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks)
            times.append(time.perf_counter() - t0)
        
        t_par = statistics.median(times)
        speedup = t_serial / t_par
        lif = n_workers * t_par / t_serial - 1
        
        times_list.append(t_par)
        lif_list.append(lif)
        
        print(f"{n_chunks:8d} | {t_par:7.3f} | {speedup:6.1f}x | {lif:6.2f}")
    
    # Find and record optimal values
    optimal_idx = times_list.index(min(times_list))
    n_chunks_optimal = n_chunks_values[optimal_idx]
    t_min = min(times_list)
    lif_min = lif_list[optimal_idx]
    
    print(f"\nOptimal: n_chunks={n_chunks_optimal}, t_min={t_min:.3f}s, LIF_min={lif_min:.2f}")
    
    # Plot wall time vs n_chunks (log scale)
    plt.figure()
    plt.plot(n_chunks_values, times_list, 'o-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Number of chunks (n_chunks)')
    plt.ylabel('Wall time (seconds)')
    plt.title('Dask Performance: Wall Time vs Number of Chunks')
    plt.grid(True, alpha=0.3)
    plt.savefig('dask_chunk_sweep.png', dpi=150)
    plt.show()
    
    #client.close()
    cluster.close()
    
"""
if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)
    
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))  
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    
    print(f"Dask local (n_chunks=32): {statistics.median(times):.3f} s")
    
    client.close()
    cluster.close()
    


if __name__ == "__main__":
    N, max_iter = 1024, 100
    n_workers = 8    # adjust to your L04 optimum
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)   # warm up JIT

    # Serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")

    # Chunk-count sweep (M2): one Pool per config
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
    for mult in [1, 2, 4, 8, 16]:
        n_chunks = mult * n_workers
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)    # warm-up: load JIT cache in workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter,
                                    n_workers=n_workers, n_chunks=n_chunks, pool=pool)
                times.append(time.perf_counter() - t0)
            t_par = statistics.median(times)
            lif = n_workers * t_par / t_serial - 1
            print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")
"""
            
