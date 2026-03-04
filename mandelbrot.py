import numpy as np
import matplotlib.pyplot as plt
import time, statistics
from numba import njit

start = time.time()


max_iter=100

def mandelbrot_point(c, max_iter):
    z = 0 + 0j  # z_0
    
    for n in range(max_iter):
        z = z*z + c  
        
        if abs(z) > 2:
            return n 
    
    return max_iter  

@njit
def mandelbrot_point_numba(c, max_iter):
    z = 0j
    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set_hybrid(x_min, x_max, y_min, y_max, width, height):
    x_values = np.linspace(x_min, x_max, width)
    y_values = np.linspace(y_min, y_max, height)
    C = np.zeros((height, width), dtype=np.complex128)
    
    iterations = np.zeros((height, width), dtype=int)
    
    for i in range(height):      # i = row index (y-axis)
        for j in range(width):   # j = column index (x-axis)
            # For each point, c = x + i*y
            C[i, j] = x_values[j] + 1j * y_values[i]

    for i in range(height):
        for j in range(width):
            iterations[i, j] = mandelbrot_point_numba(C[i, j], max_iter)
            
    return iterations


def mandelbrot_set_old(x_min, x_max, y_min, y_max, width, height):
    x_values = np.linspace(x_min, x_max, width)
    y_values = np.linspace(y_min, y_max, height)


    C = np.zeros((height, width), dtype=np.complex128)
    
    iterations = np.zeros((height, width), dtype=int)
    
    for i in range(height):      # i = row index (y-axis)
        for j in range(width):   # j = column index (x-axis)
            # For each point, c = x + i*y
            C[i, j] = x_values[j] + 1j * y_values[i]

    for i in range(height):
        for j in range(width):
            iterations[i, j] = mandelbrot_point(C[i, j], max_iter)
            
    return iterations

@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):  
        for j in range(width):  
            c = x[j] + 1j * y[i]
            z = 0j  
            n = 0
            while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                z = z*z + c
                n += 1
            result[i, j] = n
    return result


def mandelbrot_set(x_min, x_max, y_min, y_max, width, height):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)

    X, Y = np.meshgrid(x, y)

    C= X + 1j*Y
    
    Z = np.zeros_like(C)  
    M = np.zeros((height, width), dtype=int)  
    
    for n in range(max_iter):
        mask = np.abs(Z) <= 2
        
        Z[mask] = Z[mask]**2 + C[mask]
        
        M[mask] += 1
    return M


def benchmark(func, *args, n_runs=3):
    """Time func, return median of n_runs."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args) 
        times.append(time.perf_counter() - t0)
        median_t = statistics.median(times) 
    print(f"Median: {median_t:.4f}s "f"(min={min(times):.4f}, max={max(times):.4f})") 
    return median_t , result

def bench(fn, *args, runs=5): 
    fn(*args)
    times=[]
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args) 
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


x_min, x_max = -2, 1      

y_min, y_max = -1.5, 1.5 

width, height = 512, 512

mandelbrot_set_old(x_min, x_max, y_min, y_max, width, height)

#m, T = benchmark(mandelbrot_set_old, x_min, x_max, y_min, y_max, width, height)

#warmups
_ = mandelbrot_set_hybrid(-2, 1, -1.5, 1.5, 64, 64)
_ = mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64)


t_hybrid = bench(mandelbrot_set_hybrid, -2, 1, -1.5, 1.5, 1024, 1024)
t_full = bench(mandelbrot_naive_numba, -2, 1, -1.5, 1.5, 1024, 1024)
print(f"Hybrid: {t_hybrid:.3f}s")
print(f"Fully compiled: {t_full:.3f}s")
print(f"Ratio: {t_hybrid/t_full:.1f}x")


"""
m, T = benchmark(mandelbrot_set, x_min, x_max, y_min, y_max, width, height)
#L=[1,2,4,8,16]
L=[1,2,4]

bruh=np.zeros(len(L))
bruh_2=np.zeros(len(L))
for i in range(len(L)):
    n=L[i]
    m, T = benchmark(mandelbrot_set, x_min, x_max, y_min, y_max, width*(n), height*(n))
    
    bruh[i]=m
    bruh_2[i]=width*(n)
    if i>0 :
        R = bruh[i]/ bruh[i-1]
        print(f"Ratio: {R:.4f}") 
    print( width*(n), height*(n))
    print(n)

 
plt.plot(bruh_2, bruh, 'o-', linewidth=2, markersize=8)
plt.xlabel('Image Width (pixels)')
plt.ylabel('Time (ms)')
plt.title('Mandelbrot Set Computation Time')
plt.grid(True, alpha=0.3)
plt.show()

"""

"""
A = np.random.rand(10000, 10000)

def row_sum(A):
    N= A.shape[0]
    
    for i in range(N):
        s = np.sum(A[i, :])


def col_sum(A):
    N= A.shape[0]
    
    for i in range(N):
        s = np.sum(A[:, i])

benchmark(col_sum, A)

benchmark(row_sum, A)

A_f = np.asfortranarray(A)

benchmark(col_sum, A_f)

benchmark(row_sum, A_f)

"""


"""

iterations = mandelbrot_set(x_min, x_max, y_min, y_max, width, height)

plt.figure(figsize=(10, 8))
plt.imshow(iterations, extent=[x_min, x_max, y_min, y_max], 
           cmap='viridis', origin='lower')
plt.colorbar(label='Iterations until escape')
plt.title('Mandelbrot Set')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.show()
"""
