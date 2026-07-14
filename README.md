# TracerDiff

A JAX-based numerical model that couples **diffusive sediment transport**
with an **advected geochemical tracer** (e.g. a carbon isotope proxy) to
simulate how stratigraphy and stable-isotope signals co-evolve in carbonate
depositional systems.

The model was developed for, and used to produce the results in:

> van Wieren et al., 2026, *Earth and Planetary Science Letters*.
> [https://doi.org/10.1016/j.epsl.2025.119745](https://doi.org/10.1016/j.epsl.2025.119745)

It has since also been used in a second paper:

> van Wieren, Dyer & Husson, 2026, *Earth and Planetary Science Letters*.
> [https://doi.org/10.1016/j.epsl.2026.120058](https://doi.org/10.1016/j.epsl.2026.120058)

The notebooks and data used to generate the figures in the first paper live
in a separate, archived analysis repository:
[VanWieren/Paleozoic_CIEs](https://github.com/VanWieren/Paleozoic_CIEs); the
second paper's notebooks live in
[VanWieren/shuram_TOC_public](https://github.com/VanWieren/shuram_TOC_public).
**This repository (`TracerDiff`) is the actively maintained home of the
underlying model itself**, while those two are frozen snapshots of each
paper's specific analysis.

## What the model does

Carbonate sediments record the physical
history of where sediment was deposited or eroded (topography/stratigraphy)
and the geochemical history of the seawaterit precipitated from (e.g. stable isotopes, elemental geochemistry, TOC). 

TracerDiff solves for both topography (sediment transport) and the advection of a conservative tracer simultaneously.

- **Topography ($h$)** evolves under a nonlinear diffusion equation, with a
  diffusion coefficient that switches between a shallow-marine and a
  subaerial ("land") value across the shoreline (a common simplification in
  sediment transport / landscape evolution modeling).
- A **Conservative Tracer ($w$)** (the isotope or other proxy value carried
  by the sediment at each point) is advected and diffused alongside the
  topography, using an upwinding scheme so that newly deposited or
  laterally transported material correctly inherits the composition of its
  source.
- Optional **carbonate and organic growth functions** (coral, algal,
  pelagic, organic) add new sediment as a function of water depth,
  overprinting the tracer field with each source's characteristic isotopic
  offset from ambient seawater to simulate primary values for differeing sedimentary carbonate components.
- Optional **early diagenesis** tracking lets the
  recorded proxy value drift after burial (several modes: simple
  accumulation, first-order reaction, mole-balance mixing, or organic
  respiration/loss).

The solver is a Crank-Nicolson finite-difference scheme with a
matrix-free Newton (conjugate-gradient) nonlinear solve at each step,
implemented in [JAX](https://github.com/jax-ml/jax) and JIT-compiled for
speed.

## Repository structure

```
TracerDiff/
├── src/tracerdiff/       # the installable package
│   ├── model.py          # run() — the core transport/growth/reaction solver
│   ├── utils.py          # growth-function library, helpers, math utilities
│   └── output.py         # Model_output — turns raw run() output into
│                          #   stratigraphic cross sections, facies maps, and
│                          #   mass-balance diagnostics
├── examples/             # runnable, commented example scripts (see below)
├── tests/                # simple functional unit tests
├── conda-envs/           # full conda environment used for development
├── pyproject.toml        # pip-installable package definition
└── requirements.txt      # minimal runtime dependencies
```

## Installation

```bash
git clone https://github.com/VanWieren/TracerDiff.git
cd TracerDiff
pip install -e .
```

This installs `tracerdiff` and its runtime dependencies (NumPy, SciPy,
pandas, Matplotlib, Seaborn, JAX, tqdm, ipywidgets). For the full
development environment used to produce the paper's figures (extra
notebook/plotting/doc tooling), use `conda-envs/jax_env_pip.yml`:

```bash
conda env create -f conda-envs/jax_env_pip.yml
conda activate jax_env
```

## Quickstart

```python
from tracerdiff import run, init_vars
import numpy as np

params = {
    "Nx": 200, "xmin": 0.0, "xmax": 4000.0, "dx": "none",
    "start": 0, "dt": 0.1, "total_n": 150, "compiled_steps": 20,
    "marine_K": 0.25, "land_K": 0.5, "smooth_K": 6, "A": 0.1,
    "org_epsilon": 0.0, "alg_epsilon": 0.0, "pel_epsilon": 0.0,
    "coral_epsilon": 0.0, "ocean_depth": 0.0,
    "grid_ylen": 300, "base_depth": 20.0,
}
x, t, end = init_vars(params)

hi = np.linspace(2.0, -60.0, params["Nx"])   # initial topography
wi = np.full(params["Nx"], 2.0)              # initial proxy (e.g. d13C) profile

results = run(params, model_desc="demo", hi=hi, wi=wi, carb_growth=False)
```

See `examples/depositional_systems.ipynb` for an example
notebook: it runs the model three times with different sea-level curves and
depth-growth functions to produce progradational, aggradational, and
retrogradational stratigraphy, exercising most of `run()`'s parameters and
`Model_output`'s full plotting/diagnostics (cross sections, facies maps,
mass-balance and image-resolution diagnostics).

## Model parameters

`run(params, model_desc, hi, wi, ...)` reads a number of keys directly out
of `params` (no default — they must be provided):

| Key | Meaning |
|---|---|
| `Nx`, `xmin`, `xmax`, `dx` | spatial grid (`dx="none"` → `(xmax-xmin)/Nx`) |
| `start`, `dt`, `total_n`, `compiled_steps` | time stepping; total duration = `dt * total_n * compiled_steps` |
| `marine_K`, `land_K`, `smooth_K` | diffusivities on either side of the shoreline, and the smoothness of the transition between them |
| `A` | mixed-layer depth within the sediment column controlling tracer advection (where $w$ is mobile within the sediments); must be re-tuned whenever the spatial/temporal scale of a run changes. |
| `org_epsilon`, `alg_epsilon`, `pel_epsilon`, `coral_epsilon` | isotopic offset of each carbonate/organic source from ambient seawater DIC |
| `ocean_depth` | depth beyond which conditions are treated as open ocean for coral/reef growth funcitonality |
| `grid_ylen`, `base_depth` | resolution and averaging depth of the vertical stratigraphic grid used for erosion/diagenesis bookkeeping (effectively a downsampled model run generated with each timestep to save in computational costs while tracking previous stratigraphy) |

A number of other keys are optional with defaults (`f_react`, `tau`,
`fuzz`, `toc_t_cutoff`, `sw_DIC_mult`, `conv_sig`, `org_coef`, `pel_coef`,
`alg_coef`, `coral_coef`, `ep`) — see the docstrings in `model.py` for
details. Growth (`growth_fun_*`), sea level (`sl_fun`), and secular seawater
composition (`sec_w_fun`) are all supplied as plain Python/JAX callables, so
any function of depth or time can be substituted.

`run()` returns a dictionary of NumPy arrays (topography, proxy values,
shoreline position, mass-balance diagnostics, etc.) which can be passed
directly into `Model_output(facies_data, facies_colours, **results)` for
visualization.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

`tests/test.py` covers imports and pure NumPy/JAX helper functions
(fast, no JIT). `tests/test_run.py` actually runs `run()` and
`Model_output` end-to-end on a small grid and checks output shapes/sanity.

## Citation

If you use this model, please cite:

> van Wieren, C.S., Dyer, B., Husson, J.M., 2026, Correlative isotope excursions driven by transport, not global environmental change: *Earth and Planetary Science Letters*, vol. 673, p. 119745, [doi:10.1016/j.epsl.2025.119745](https://doi.org/10.1016/j.epsl.2025.119745)

and, if relevant, the second paper that also used this model:

> van Wieren, C.S., Dyer, B., Husson, J.M., 2026, Heterogeneous organic carbon remineralization as a driver for highly negative carbonate $\delta^{13}$}C excursions: *Earth and Planetary Science Letters*, vol. 687, p. 120058, [doi:10.1016/j.epsl.2026.120058](https://doi.org/10.1016/j.epsl.2026.120058)


## License

GPL-3.0 — see `LICENSE`.
