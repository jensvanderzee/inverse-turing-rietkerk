# Inverse Turing patterns — Rietkerk backbone

Code and data for learning the parameters of the Rietkerk et al. (2002) dryland
vegetation model (an "inverse PDE") from satellite NDVI time series, and for
comparing that approach against a black-box recurrent convolutional network (RCNN).

The model has three coupled reaction–diffusion fields: surface water `O`, soil
water `W` and biomass `B` (Rietkerk et al. 2002, *Am. Nat.* 160:524, eq. 1, flat
ground):

```
∂O/∂t = D_O ΔO + R − α O (B + k₂W₀)/(B + k₂)
∂W/∂t = D_W ΔW + α O (B + k₂W₀)/(B + k₂) − g_max W B/(W + k₁) − r_w W
∂B/∂t = D_P ΔB + c g_max W B/(W + k₁) − d B
```

Eleven scalar coefficients are learned by differentiating a multi-year rollout
driven by weekly precipitation and fitting the predicted year-on-year change in
biomass to the observed change.

This repository previously used the unified three-field model of Siero (2020,
*Physica D* 414:132695, eq. 1) with linear infiltration `I(B) = B` and uptake
`U(W,B) = W·B`. Rietkerk is the same framework with the saturating forms above and
no surface-water evaporation (Siero 2020, eqs. 13–14), so the training, testing
and analysis scripts are unchanged in structure; the coefficients that play the
same role keep their old names (`seepage_rate` = r_w, `mortality_rate` = d,
`infiltration_rate` = α, `plant_uptake_rate` = g_max, `water_use_efficiency` = c),
`evaporation_rate` is gone, and `uptake_half_saturation` (k₁),
`infiltration_half_saturation` (k₂) and `bare_soil_infiltration` (W₀) are new.

The pipeline is written in PyTorch. `julia/` holds `InverseTuring.jl`, a Julia
implementation of the same pipeline on the Rietkerk backbone: it reproduces
`rietkerk_model.py` (losses and autograd gradients to 1e-8 or better, checked
against values the Python code writes), differentiates the rollout with Enzyme,
and can also pose the fit on the continuous PDE with DifferentialEquations.jl and
SciMLSensitivity.jl. See [`julia/README.md`](julia/README.md).

## The Rietkerk backbone: what changed and why

Everything model-specific lives in [`rietkerk_model.py`](rietkerk_model.py), which
every script imports; nothing else defines the equations. What a user of the
scripts needs to know:

- **Units.** Time is in days (one simulated year = 52 forcing weeks = 364 days), so
  weekly ERA5 rates in mm/day are injected as-is and fitted rates compare directly
  with Rietkerk's per-day values. Space is in pixels: diffusion coefficients are in
  pixel²/day. Synthetic grids use 5 m cells (Rietkerk's ~45 m pattern wavelength is
  ~9 cells); the Landsat data are 30 m. Real-data biomass stays NDVI × 1500, as
  before; Rietkerk's model is exactly invariant under a change of biomass unit
  (B → sB, c → sc, g_max → g_max/s, k₂ → sk₂), so the multiplier only rescales
  those three coefficients. `to_physical_units` converts fits back to m²/day and
  g/m² (taking NDVI × 100 ≈ g/m²), which is what the reporting scripts print.
- **Numerics.** D_O/D_P = 1000 in Rietkerk's model; resolving its patterns with the
  old explicit Laplacian would need steps of ~0.2 day. The step is now
  semi-implicit: diffusion is solved exactly (DCT of the same 5-point, replicate-
  boundary Laplacian) and each field's own loss term is implicit. Fields stay
  positive and the step is stable for any parameter values, so a fit can no longer
  turn into `NaN`. The uniform equilibrium is an exact fixed point, the scheme
  converges to explicit Euler at small steps, and the field update order is
  unchanged (O, then W, then B). `semi_implicit=False` restores explicit Euler.
- **Reference and ground truth.** `SYNTHETIC_TRUTH` is Rietkerk's published values
  with the plant equation slowed 25× (`PLANT_TIMESCALE_FACTOR`: c, d, D_P ÷ 25).
  With d = 0.25/day his plants live ~4 days, and under weekly forcing with any dry
  season they die out in the first year (his R is a climatic mean). The slowdown
  leaves the uniform equilibria, the Turing range (T₁ = 1.001, T₂ = 1.259 mm/day,
  reproduced by `turing_value`) and the wavelength unchanged. `REALDATA_REFERENCE`
  is the same set on 30 m pixels in NDVI × 1500 units; it is the comparison point
  in the real-data tables (it no longer sets the random starts or the bounds).
- **Optimisation.** The coefficients span five decades, so they are optimised in
  log space (`log_<name>` parameters; read values with `parameter_values()`).
  Learning rates are therefore relative steps: 0.01 (was 0.1–0.2 in raw space),
  decaying by 0.9999 per epoch on the synthetic data and 0.9995 on the real data.
  The summed steps bound how far a parameter can travel from its start: ~8.5
  decades over the 7500 real-data epochs, which the real-data reference values
  below the start range need (D_P ≈ 4e-6 is 3.4 decades under 0.01).
  Log space keeps every parameter positive, so there are no clamps beyond
  `NUMERICAL_BOUNDS` (1e-30 to 1e30, which only keep values finite and nonzero in
  float32; W₀ is not capped at 1 either). Degenerate-run filters
  (`degenerate_parameters`) flag a parameter that is non-finite or more than four
  decades from the reference, instead of `< 1e-4` / `< 0.0011`, which would reject
  ordinary values such as D_W ≈ 1e-4 pixel²/day.
- **Initialisation.** Every parameter is drawn log-uniformly from the same range,
  `INIT_RANGE` = 0.01–100, independent of the reference values, and the draw is
  **redrawn until mean biomass stays within 0.1–10× of its initial level at every
  site over the training rollout** (`draw_viable_model`, up to 1000 draws). B = 0
  is absorbing: from a start where the plants die, the later years give no
  gradient and the fit only tunes the die-off. About 6% of draws from this prior
  pass on the real data, so a run typically needs ~15 rollouts to find its start.
  Starts where biomass explodes are dropped too; on the real data they begin four
  orders of magnitude above the no-change loss. Each run records `init_draws`.
- **Turing and composite diagnostics.** Rietkerk's uniform state depends on
  rainfall, so `turing_value(params, precip)` is evaluated numerically from the
  dispersion relation at the mean training rainfall (negative = Turing-unstable, as
  before; NaN where no stable uniform vegetated state exists). The composite is
  `(g_max B/k₁)/(r_w + g_max B/k₁)·c`, the old composite with l₁ = 0 and
  r₁ = g_max/k₁ (Siero's linearisation).

Outputs go to new directories (`results/*_rietkerk/`, `results/real_data_rietkerk/`)
so results of the two backbones never mix. Real-data training writes
`results/real_data_rietkerk/models/{models,parameters,results}`, and every
downstream script reads from there.

## Repository layout

| path | contents |
|:--|:--|
| `data/` | Input data per subsite (`subsite_a` … `subsite_k`): yearly NDVI GeoTIFFs for 2013–2022 (`*_ndvi/`), annual and weekly ERA5 precipitation (`*_precip/`), and the area-of-interest polygon (`aoi` / `aoi.txt`) |
| `rietkerk_model.py` | The Rietkerk backbone: model, parameters, reference values, units, diagnostics |
| `*.py` (root) | Training, testing and analysis scripts, listed below |
| `*.sh` | SLURM job scripts used on the HPC cluster |
| `python_1site/` | Driver that fits the real-data model on a single site |
| `julia/` | `InverseTuring.jl`, the Julia implementation of the pipeline (Rietkerk backbone) |

**`results/` is not included in this repository.** Every training script writes
to it, and the analysis and plotting scripts read from it, so run the
corresponding training step first. The manuscript and submission material are
also kept out of the repository.

## Sites

- **Training:** subsites `b`, `i`, `c`, `e`
- **Held-out testing:** subsites `f`, `k`, `j`
- **Initial conditions for forward simulations:** subsite `a`

## Scripts

Run all scripts from the repository root. Many are written as `#%%` cells
so they can also be stepped through interactively.

### Data preparation

| script | purpose |
|:--|:--|
| `download_era5_precip.py` | Download ERA5 daily precipitation through the Copernicus CDS API and compute weekly averages (needs `~/.cdsapirc`) |
| `download_era5_precip_gee.py` | Same as above, but through Google Earth Engine |
| `download_missing_weekly_precip.py` | Rebuild the weekly precipitation for held-out subsites `j` and `k` from hourly ERA5, with the same source and processing as the training sites (`--project <Earth Engine project>`) |
| `view_site.py` | Quick look at the NDVI raster for one site and year |
| `plot_site_map.py` | Map of all subsite locations |

### Synthetic experiments

Parameters are recovered from data generated with known ground-truth values.

| script | purpose |
|:--|:--|
| `train_invPDE_synthetic_batch.py` | Train the inverse PDE on synthetic data from four sites (`--start`, `--end`, `--gpu`, `--learning_rate`, `--grid_size`, and `--gradient_checkpointing` for GPUs under 24 GB: the fit needs ~12 GB without it) |
| `synthetic_train_colab.ipynb` | Run the four-site or one-site synthetic experiment on a Colab GPU, with results on Google Drive; finished runs are skipped and four-site runs resume from their checkpoints after a disconnect |
| `train_invPDE_synthetic_batch_1site.py` | Same, for one site |
| `train_rcnn_batch.py` | Train the RCNN baseline on the same synthetic data |
| `compare_invPDE_synthetic_params.py` | Learned vs ground-truth parameters, four sites |
| `compare_invPDE_synthetic_params_1site.py` | Learned vs ground-truth parameters, one site |
| `compare_1site_vs_4site_params.py`, `compare_1site_vs_4site_relerr.py` | Parameter values and relative errors, one site vs four sites |
| `synthetic_dataregime_analysis.py` | Effect of adding more training sites |
| `synthetic_sensitivity_analysis.py` | One-at-a-time sensitivity of simulated biomass to each parameter |
| `rcnn_vs_PDE_synthetic.py` | Extrapolation ability of the RCNN vs the inverse PDE. **Not ported**: it still carries the old model and pulse-forcing protocol and reads output formats the current training scripts do not write; `rcnn_extrapolation_test.py` supersedes it |
| `rcnn_extrapolation_test.py` | Best RCNN and inverse-PDE models tested on longer horizons and out-of-range precipitation |
| `invPDE_1site_extrapolation.py` | Extrapolation test for the one-site inverse PDE |
| `rcnn_simulate.py` | Forward simulation with the best RCNN model |

### Real satellite data

| script | purpose |
|:--|:--|
| `realdata_train_invPDE.py` | Fit PDE parameters to the four training sites (7500 epochs per model; `--models START STOP`, `--device`, `--num_epochs`, and `--gradient_checkpointing` for GPUs with 16 GB or less) |
| `realdata_train_colab.ipynb` | Run `realdata_train_invPDE.py` on a Colab GPU, with checkpoints on Google Drive so training resumes after a disconnect |
| `realdata_train_invPDE_10to20.py` | Same, for model indices 10–19, so runs can be split across jobs |
| `find_stable_seed.py` | Find a seed that trains stably for biomass multipliers 750, 1500 and 3000 |
| `multiplier_check_analysis.py` | Compare runs with different biomass multipliers |
| `realdata_test_invPDE.py` | Score the fitted models on the held-out sites |
| `realdata_parameter_analysis.py` | Agreement of learned parameters across runs |
| `realdata_parameter_correlations.py` | Correlations and trade-offs between learned parameters |
| `compare_invPDE_realdata_params.py` | Parameter trajectories across all real-data runs |
| `compare_invPDE_realdata_params_with_composite.py` | Same, with a composite figure |
| `realdata_simulate_invPDE.py` | Forward simulation with the best tested model |
| `bifurcation_parallel.py` | Equilibrium biomass vs annual precipitation for every model (multiprocessing) |
| `realdata_bifurcation_plot.py` | Serial version of the bifurcation sweep and plot |
| `plot_bifurcation.py`, `plot_bifurcation_cosmetic.py`, `bifurcation_plot_only.py` | Redraw the bifurcation figure from saved sweep data |
| `python_1site/train_1site_realdata.py` | Fit the real-data model on one site (`--site b`) for comparison with the four-site fit |

### Numerics and runtime

| script | purpose |
|:--|:--|
| `compare_forcing_frequency.py` | Same annual rainfall delivered daily, weekly or as one annual pulse (`compare_forcing_frequency_nb.py` is the cell-by-cell version) |
| `measure_runtime.py` | Measure seconds per epoch for each experiment and extrapolate to a full run; writes `runtime_estimates.csv` and `runtime_estimates.md` (the committed files are for the previous backbone) |
| `measure_runtime_colab.ipynb` | Notebook for running `measure_runtime.py` on Colab |

Estimated time for one training run on an NVIDIA A100 **with the previous
backbone** (from [`runtime_estimates.md`](runtime_estimates.md)). For the Rietkerk
backbone, rerun `measure_runtime.py` on the target hardware. The synthetic
experiments now take 2 steps per week instead of 1, but the four synthetic sites
are simulated as one batch, and a step is dominated by kernel-launch overhead on
grids this small; for a memory-limited GPU, set `model.gradient_checkpointing = True`
(one extra forward pass, far less activation memory; `realdata_train_invPDE.py
--gradient_checkpointing` does this). Real-data training without it needs more
than 15 GB on CPU, and peaks at about 2.3 GB with it.

| experiment | epochs | estimated wall time |
|:--|--:|--:|
| invPDE synthetic, 1 site | 7500 | 2 h |
| invPDE synthetic, 4 sites | 7500 | 8 h |
| RCNN synthetic, 4 sites | 1000 | 39 min |
| invPDE real data, 4 sites | 7500 | 25 h |

## Requirements

Python 3.12 with PyTorch, plus `numpy`, `pandas`, `matplotlib`, `seaborn`,
`scipy`, `scikit-learn`, `rasterio`, `tqdm` and `Pillow`. Some scripts need extra
packages:

- `plot_site_map.py`: `contextily`, `pyproj`
- `download_era5_precip.py`: `cdsapi`, `xarray`, `netcdf4`
- `download_*_gee.py`, `download_missing_weekly_precip.py`: `earthengine-api`

A GPU is recommended for training. For the Julia implementation (CPU, Julia ≥ 1.12),
see [`julia/README.md`](julia/README.md).

## Running on the cluster

The `*.sh` files are SLURM job scripts, each requesting one GPU for up to 48 hours.
They call the Python scripts by absolute path on the cluster
(`/home/WUR/zee034/inverse-turing-testing/...`), so change that path before
using them elsewhere.

Several of them call copies of `realdata_train_invPDE.py` that exist only on the
cluster (`realdata_train_invPDE_20to30.py` … `_90to100.py`, `_750.py`,
`_1500.py`, `_3000.py`). Those copies still contain the previous backbone;
regenerate them from the current `realdata_train_invPDE.py` (as
`realdata_train_invPDE_10to20.py` is: only the model-index range and, for the
multiplier checks, `ndvi_to_biomass_multiplier` differ) before submitting.

## License

MIT, see [`LICENSE`](LICENSE).
