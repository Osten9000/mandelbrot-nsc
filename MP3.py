#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:44:08 2026

@author: phillycheese
"""

"""
L08 m1

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
plt.savefig('mandelbrot_divergence_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

"""

"""
L08 m1
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

N, MAX_ITER = 512, 1000
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace(0.0990, 0.1030, N)
C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
eps32 = float(np.finfo(np.float32).eps)
delta = np.maximum(eps32 * np.abs(C), 1e-10)

def escape_count(C, max_iter):
    z = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)
    for k in range(max_iter):
        z[~esc] = z[~esc]**2 + C[~esc]
        newly = ~esc & (np.abs(z) > 2.0)
        cnt[newly] = k
        esc[newly] = True
    return cnt

n_base = escape_count(C, MAX_ITER).astype(float)
n_perturb = escape_count(C + delta, MAX_ITER).astype(float)
dn = np.abs(n_base - n_perturb)
kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)

cmap_k = plt.cm.hot.copy()
cmap_k.set_bad('0.25')
vmax = np.nanpercentile(kappa, 99)

plt.imshow(kappa, cmap=cmap_k, origin='lower',
    extent=[-0.7530, -0.7490, 0.0990, 0.1030],
    norm=LogNorm(vmin=1, vmax=vmax))
#plt.colorbar(label=r'$\kappa$ (log scale)')
plt.colorbar(label=r'$\kappa(c)$ (log scale, $\kappa \geq 1$)')
plt.title(r'Condition number approx $\kappa$')
plt.savefig('mandelbrot_condition_number.png', dpi=300, bbox_inches='tight')
plt.show()