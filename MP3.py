#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:44:08 2026

@author: phillycheese
"""

import numpy as np
import matplotlib.pyplot as plt

N, MAX_ITER, TAU = 512, 1000, 0.1
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace(0.0990, 0.1030, N)

# ========== PART 1: Divergence map (float32 vs float64) ==========
C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128) 
C32 = C64.astype(np.complex64)
z32 = np.zeros_like(C32) 
z64 = np.zeros_like(C64)
diverge = np.full((N, N), MAX_ITER, dtype=np.int32) 
active = np.ones((N, N), dtype=bool)

for k in range(MAX_ITER):
    if not active.any(): break
    z32[active] = z32[active]**2 + C32[active]
    z64[active] = z64[active]**2 + C64[active]
    diff = (np.abs(z32.real.astype(np.float64) - z64.real) + 
            np.abs(z32.imag.astype(np.float64) - z64.imag)) 
    newly = active & (diff > TAU)
    diverge[newly] = k
    active[newly] = False

# ========== PART 2: Normal Mandelbrot escape-time map (float64) ==========
z_escape = np.zeros_like(C64)
escape_count = np.full((N, N), MAX_ITER, dtype=np.int32)
active_escape = np.ones((N, N), dtype=bool)

for k in range(MAX_ITER):
    if not active_escape.any(): break
    z_escape[active_escape] = z_escape[active_escape]**2 + C64[active_escape]
    diverged = active_escape & (np.abs(z_escape) > 2.0)
    escape_count[diverged] = k
    active_escape[diverged] = False

# ========== PLOTTING: Side-by-side comparison ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Divergence map
im1 = ax1.imshow(diverge, cmap='plasma', origin='lower', 
                 extent=[-0.7530, -0.7490, 0.0990, 0.1030])
ax1.set_title(f'Trajectory Divergence (τ={TAU})\nFirst iteration where |z32 - z64| > {TAU}')
ax1.set_xlabel('Real')
ax1.set_ylabel('Imaginary')
plt.colorbar(im1, ax=ax1, label='Divergence iteration')

# Normal escape-time map
im2 = ax2.imshow(escape_count, cmap='plasma', origin='lower',
                 extent=[-0.7530, -0.7490, 0.0990, 0.1030])
ax2.set_title('Normal Mandelbrot (float64)\nEscape iteration (|z| > 2)')
ax2.set_xlabel('Real')
ax2.set_ylabel('Imaginary')
plt.colorbar(im2, ax=ax2, label='Escape iteration')

plt.tight_layout()
plt.show()