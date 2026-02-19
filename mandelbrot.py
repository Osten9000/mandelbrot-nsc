import numpy as np
import matplotlib.pyplot as plt
import time, statistics
start = time.time()


max_iter=100

def mandelbrot_point(c, max_iter):
    z = 0 + 0j  # z_0
    
    for n in range(max_iter):
        z = z*z + c  
        
        if abs(z) > 2:
            return n 
    
    return max_iter  


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
            iterations[i, j] = mandelbrot_point(C[i, j])
            
    return iterations


def mandelbrot_set(x_min, x_max, y_min, y_max, width, height):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)

    X, Y = np.meshgrid(x, y)

    C= X + 1j*Y
    
    Z = np.zeros_like(C)  
    M = np.zeros((height, width), dtype=int)  
    
    for n in range(max_iter):
        # Boolean mask: points that haven't escaped yet
        mask = np.abs(Z) <= 2
        
        # Update only unescaped points
        Z[mask] = Z[mask]**2 + C[mask]
        
        # Increment iteration count for unescaped points
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


x_min, x_max = -2, 1      

y_min, y_max = -1.5, 1.5 

width, height = 1024, 1024

benchmark(mandelbrot_set, x_min, x_max, y_min, y_max, width, height)

"""
iterations = mandelbrot_set_old(x_min, x_max, y_min, y_max, width, height)

elapsed = time.time() - start
print(f"Computation took {elapsed:.3f} seconds")
    
plt.figure(figsize=(10, 8))
plt.imshow(iterations, extent=[x_min, x_max, y_min, y_max], 
           cmap='viridis', origin='lower')
plt.colorbar(label='Iterations until escape')
plt.title('Mandelbrot Set')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.show()

"""
