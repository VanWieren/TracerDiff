"""
Integration tests that actually run the model end-to-end.

Unlike test.py (which deliberately avoids running `run()` at all),
these tests execute the real JAX solve on a small, fast grid and check
that the output has the shape and keys the rest of the package (in
particular `Model_output`) depends on. They will be slower than the
smaller tests, but they are the only tests
that would actually catch a broken `run()`/`Model_output`.

The params dict below is a scaled-down version of the one in the
README's Quickstart section (same keys, smaller Nx/total_n for speed).
"""

import numpy as np
import pytest


def _small_params():
    return {
        "Nx": 40,
        "xmin": 0.0,
        "xmax": 800.0,
        "dx": "none",
        "start": 0,
        "dt": 0.1,
        "total_n": 20,
        "compiled_steps": 10,
        "marine_K": 0.25,
        "land_K": 0.5,
        "smooth_K": 6,
        "A": 0.1,
        "org_epsilon": 0.0,
        "alg_epsilon": 0.0,
        "pel_epsilon": 0.0,
        "coral_epsilon": 0.0,
        "ocean_depth": 0.0,
        "grid_ylen": 100,
        "base_depth": 20.0,
    }


def test_run_end_to_end_shapes():
    """run() on a tiny grid should complete and return arrays consistent
    with the requested Nx/total_n, with no NaNs in the core topography
    and proxy fields."""
    from tracerdiff import run, init_vars

    params = _small_params()
    x, t, end = init_vars(params)
    assert len(x) == params["Nx"]
    assert len(t) == params["total_n"]

    hi = np.linspace(2.0, -60.0, params["Nx"])
    wi = np.full(params["Nx"], 2.0)

    results = run(
        params, model_desc="pytest-smoke", hi=hi, wi=wi,
        carb_growth=False, plot_out=False,
    )

    for key in ("x", "beds", "proxy", "ds", "params", "total_n"):
        assert key in results, f"run() output missing expected key {key!r}"

    beds = np.asarray(results["beds"])
    proxy = np.asarray(results["proxy"])

    # beds/proxy are stored per compiled step, each row of length Nx
    assert beds.shape[-1] == params["Nx"]
    assert proxy.shape[-1] == params["Nx"]
    assert np.isfinite(beds).all(), "topography contains NaN/inf"
    assert np.isfinite(proxy).all(), "proxy contains NaN/inf"


def test_model_output_basic_pipeline():
    """Model_output should build cleanly from a real (tiny) run() result
    and produce a gridded image array of the requested resolution."""
    from tracerdiff import run, init_vars, Model_output

    params = _small_params()
    x, t, end = init_vars(params)

    hi = np.linspace(2.0, -60.0, params["Nx"])
    wi = np.full(params["Nx"], 2.0)

    results = run(
        params, model_desc="pytest-smoke", hi=hi, wi=wi,
        carb_growth=False, plot_out=False,
    )

    max_depth = float(np.nanmax(np.asarray(results["ds"])))
    # minimal 2-bin facies scheme; Model_output.init_facies() auto-inserts
    # a Terrestrial facies at index 0, so facies_colours needs one extra
    # entry relative to facies_data
    facies_data = [
        {"name": "Shallow", "type": "uniform", "min": -0.01, "max": max_depth / 2, "width": 0.6},
        {"name": "Deep", "type": "uniform", "min": max_depth / 2, "max": max_depth, "width": 0.3},
    ]
    facies_colours = ["#c2b280", "#5c4033", "#333333"]

    out = Model_output(
        **results,
        facies_data=facies_data,
        facies_colours=facies_colours,
        im_ylen=50,
        shore=False,
        images=True,
    )

    assert out.im_w.shape[0] == 50
    assert out.facies_pred is not None


def test_model_output_missing_bc_filter_set_under_c_are_inert():
    """Regression guard: bc_filter/set_under_c are not real features of the
    upstream model (confirmed against the sed_transport source) -- passing
    them (as several downstream analysis notebooks do) must not raise, and
    must have no effect on the gridded output."""
    from tracerdiff import run, init_vars, Model_output

    params = _small_params()
    x, t, end = init_vars(params)
    hi = np.linspace(2.0, -60.0, params["Nx"])
    wi = np.full(params["Nx"], 2.0)

    results = run(
        params, model_desc="pytest-smoke", hi=hi, wi=wi,
        carb_growth=False, plot_out=False,
    )

    max_depth = float(np.nanmax(np.asarray(results["ds"])))
    facies_data = [
        {"name": "Shallow", "type": "uniform", "min": -0.01, "max": max_depth / 2, "width": 0.6},
        {"name": "Deep", "type": "uniform", "min": max_depth / 2, "max": max_depth, "width": 0.3},
    ]
    facies_colours = ["#c2b280", "#5c4033", "#333333"]

    out_plain = Model_output(
        **results, facies_data=facies_data, facies_colours=facies_colours,
        im_ylen=50, shore=False, images=True,
    )
    out_with_extra_kwargs = Model_output(
        **results, facies_data=facies_data, facies_colours=facies_colours,
        im_ylen=50, shore=False, images=True,
        bc_filter=False, set_under_c=".3",
    )

    np.testing.assert_array_equal(
        np.asarray(out_plain.im_w), np.asarray(out_with_extra_kwargs.im_w)
    )
