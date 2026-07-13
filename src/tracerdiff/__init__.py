"""
TracerDiff
==========

A JAX-based tracer-advection / diffusion model of stratigraphic and
geochemical proxy evolution in carbonate depositional systems.

The model couples a nonlinear diffusion equation for topography (sediment
transport as a function of water depth) with an advected conservative tracer field
(e.g. a stable isotope proxy such as d13C) that records the isotopic
composition of accumulating sediment. Carbonate/organic growth functions,
sea-level history, and early diagenetic alteration can all be
layered on top of the core transport solver.

Published in van Wieren et al., 2026, Earth and Planetary Science Letters
(https://doi.org/10.1016/j.epsl.2025.119745), and since also used in van
Wieren, Dyer & Husson, 2026, Earth and Planetary Science Letters
(https://doi.org/10.1016/j.epsl.2026.120058). The companion analysis
repositories with the notebooks used to produce each paper's figures are at
https://github.com/VanWieren/Paleozoic_CIEs and
https://github.com/VanWieren/shuram_TOC_public, respectively.

Quickstart
----------
>>> from tracerdiff import run, init_vars
>>> params = {...}
>>> x, t, end = init_vars(params)
>>> results = run(params, model_desc="demo", hi=hi, wi=wi)

See the `examples/` directory in the repository root for complete,
runnable scripts.
"""

from .utils import (
    init_vars,
    sawtooth,
    sin_exp,
    save_object,
    load_object,
    create_excursions,
    bosscher_G,
    erf_G,
    erf_G_asym,
    calc_mol,
    mol_bal,
    mol_bal_mass,
    compute_f_react,
    imshow,
    norm01,
)
from .model import run
from .output import Model_output

__version__ = "0.1.0"

__all__ = [
    "run",
    "Model_output",
    "init_vars",
    "sawtooth",
    "sin_exp",
    "save_object",
    "load_object",
    "create_excursions",
    "bosscher_G",
    "erf_G",
    "erf_G_asym",
    "calc_mol",
    "mol_bal",
    "mol_bal_mass",
    "compute_f_react",
    "imshow",
    "norm01",
]
