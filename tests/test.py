"""
Lightweight tests.

These intentionally avoid running the full `run()` PDE solver, so they
stay fast (no JIT compilation). Instead they check that the package
imports cleanly and that a handful of pure NumPy/JAX helper functions
behave as documented, so a broken import or an accidental signature
change in a refactor gets caught quickly.

For tests that actually exercise `run()`/`Model_output` end-to-end on a
small grid, see test_run.py.
"""

import numpy as np
import pytest


def test_package_imports():
    import tracerdiff
    from tracerdiff import run, Model_output, init_vars

    assert callable(run)
    assert callable(init_vars)
    assert isinstance(tracerdiff.__version__, str)


def test_init_vars_shapes():
    from tracerdiff import init_vars

    params = {
        "Nx": 50,
        "xmin": 0.0,
        "xmax": 100.0,
        "dt": 0.1,
        "total_n": 20,
        "compiled_steps": 5,
        "start": 0,
    }
    x, t, end = init_vars(params)
    assert len(x) == params["Nx"]
    assert len(t) == params["total_n"]
    assert end == pytest.approx(params["dt"] * params["total_n"] * params["compiled_steps"])


def test_norm01_range():
    from tracerdiff.utils import norm01

    arr = np.array([2.0, 4.0, 6.0, 8.0])
    normed = norm01(arr)
    assert normed.min() == pytest.approx(0.0)
    assert normed.max() == pytest.approx(1.0)


def test_round_2dec():
    from tracerdiff.utils import round_2dec

    assert round_2dec(1.231, 1, way="up") == pytest.approx(1.3)
    assert round_2dec(1.239, 1, way="down") == pytest.approx(1.2)


def test_create_excursions_preserves_length_and_baseline():
    from tracerdiff.utils import create_excursions

    baseline = np.zeros(100)
    result = create_excursions(
        array=baseline,
        excursions=[-3.0],
        locs=[50],
        rise_widths=[10],
        fall_widths=[10],
        rise_sigmas=[2.0],
        fall_sigmas=[2.0],
    )
    assert result.shape == baseline.shape
    # excursion should push values negative somewhere near the peak location
    assert result.min() < -0.5
    # far from the excursion, values should stay close to baseline
    assert abs(result[0]) < 0.5


def test_erf_g_asym_peaks_near_g_depth():
    from tracerdiff.utils import erf_G_asym
    import jax.numpy as jnp

    depth = jnp.linspace(0, 40, 200)
    growth = erf_G_asym(depth, Gmax=1.0, G_depth=10.0, width_shallow=5.0, width_deep=15.0)
    peak_depth = float(depth[int(jnp.argmax(growth))])
    assert peak_depth == pytest.approx(10.0, abs=1.0)
