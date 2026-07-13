# Examples

**`depositional_systems.ipynb`** runs the model three times with three
different relative sea level histories and depth-growth functions,
producing three stratigraphic sequence-stacking patterns: progradation, aggradation, and retrogradation. This demonstrates the use of
most of `run()`'s main parameters and of `Model_output` plotting/diagnostics (`plot_grids`, `mass_balance`, `im_res_compare`, `strat_col_im`, the combined comparison figure).

Run it with:

```bash
pip install -e ".[dev]"
pip install jupyter
jupyter lab examples/depositional_systems.ipynb
```

Model parameters (diffusivities, growth coefficients, mixed-layer depth
`A`, sea-level curves, etc.) are illustrative starting points, not
calibrated values — see the main README and the in-source comments in
`src/tracerdiff/model.py` for what each parameter controls, and retune as needed. 
This notebook is adapted from
`depositional_systems.ipynb` in the repository,
[VanWieren/Paleozoic_CIEs](https://github.com/VanWieren/Paleozoic_CIEs),
which also has the parameter sets actually used to produce the paper's
published figures.
