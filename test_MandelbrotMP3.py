import numpy as np
import pytest
from MandelbrotMP3 import mandelbrot_set 


# ---------------------------------------------------------------------------
# Helper: run mandelbrot_set on a single point and return its iteration count
# ---------------------------------------------------------------------------
def single_point(c: complex, max_iter: int = 100) -> int:
    """Evaluate mandelbrot_set on a 1×1 grid centred on c."""
    M = mandelbrot_set(
        x_min=c.real, x_max=c.real,
        y_min=c.imag, y_max=c.imag,
        width=1, height=1,
        max_iter=max_iter,
    )
    return int(M[0, 0])


# ---------------------------------------------------------------------------
# Test 1 — points INSIDE the set reach max_iter
# ---------------------------------------------------------------------------
def test_interior_points_reach_max_iter():
    """
    Well-known interior points must iterate all the way to max_iter,
    meaning they never escaped — they are IN the set.
    """
    max_iter = 50
    interior_points = [
        0 + 0j,       # origin
        -1 + 0j,      # period-2 bulb centre
        -0.5 + 0j,    # main cardioid interior
    ]
    for c in interior_points:
        count = single_point(c, max_iter=max_iter)
        assert count == max_iter, (
            f"Expected {c} to stay in set (count={max_iter}), got {count}"
        )


# ---------------------------------------------------------------------------
# Test 2 — points OUTSIDE the set escape before max_iter
# ---------------------------------------------------------------------------
def test_exterior_points_escape():
    """
    Points far outside the set must escape well before max_iter.
    The escape condition is |z| > 2, so c=10+0j escapes on iteration 1.
    """
    max_iter = 100
    exterior_points = [
        10 + 0j,
        0 + 10j,
        5 + 5j,
    ]
    for c in exterior_points:
        count = single_point(c, max_iter=max_iter)
        assert count < max_iter, (
            f"Expected {c} to escape, but got count={count}"
        )


# ---------------------------------------------------------------------------
# Test 3 — output grid has the correct shape
# ---------------------------------------------------------------------------
def test_output_shape():
    """mandelbrot_set must return an array of shape (height, width)."""
    M = mandelbrot_set(-2, 1, -1.5, 1.5, width=80, height=60, max_iter=50)
    assert M.shape == (60, 80), f"Expected (60, 80), got {M.shape}"


# ---------------------------------------------------------------------------
# Test 4 — iteration counts are bounded between 0 and max_iter (inclusive)
# ---------------------------------------------------------------------------
def test_iteration_counts_in_valid_range():
    """Every value in M must satisfy 0 <= value <= max_iter."""
    max_iter = 40
    M = mandelbrot_set(-2, 1, -1.5, 1.5, width=50, height=50, max_iter=max_iter)
    assert M.min() >= 0,        f"Found negative iteration count: {M.min()}"
    assert M.max() <= max_iter, f"Count exceeded max_iter: {M.max()}"


# ---------------------------------------------------------------------------
# Test 5 — @pytest.mark.parametrize across known (point → expected) pairs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c, in_set", [
    # Inside the set → should hit max_iter
    (0 + 0j,          True),   # origin
    (-1 + 0j,         True),   # period-2 bulb
    (-0.5 + 0.5j,     True),   # main cardioid, upper half

    # Outside the set → should escape before max_iter
    (2 + 0j,          False),  # boundary/outside
    (0 + 2j,          False),  # boundary/outside
    (-2.5 + 0j,       False),  # left of the set
    (1 + 1j,          False),  # upper-right quadrant
])
def test_known_points(c, in_set):
    """
    Spot-check mathematically known points.
    in_set=True  → iteration count must equal max_iter  (never escaped)
    in_set=False → iteration count must be  < max_iter  (did escape)
    """
    max_iter = 200   # high enough to be confident for boundary-ish points
    count = single_point(c, max_iter=max_iter)
    if in_set:
        assert count == max_iter, (
            f"{c} should be IN the set but escaped at iteration {count}"
        )
    else:
        assert count < max_iter, (
            f"{c} should be OUTSIDE the set but reached max_iter={max_iter}"
        )