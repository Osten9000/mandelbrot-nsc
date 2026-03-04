#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:45:55 2026

@author: phillycheese
"""
import cProfile, pstats
from mandelbrot import mandelbrot_set_old, mandelbrot_set

cProfile.run("mandelbrot_set_old(-2, 1, -1.5, 1.5, 512, 512)", "naive_profile.prof")

cProfile.run("mandelbrot_set(-2, 1, -1.5, 1.5, 512, 512)", "numpy_profile.prof")

for name in ("naive_profile.prof", "numpy_profile.prof"):
    stats = pstats.Stats(name) 
    stats.sort_stats("cumulative")
    stats.print_stats(10)
