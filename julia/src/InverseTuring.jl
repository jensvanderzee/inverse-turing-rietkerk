"""
    InverseTuring

Julia implementation of the inverse-PDE pipeline that fits the Rietkerk et al.
(2002) dryland vegetation model to satellite NDVI time series — the counterpart of
`rietkerk_model.py` and the training, testing and analysis scripts around it.

The model has three coupled reaction–diffusion fields (surface water `O`, soil
water `W`, biomass `B`) and eleven coefficients, fitted by differentiating a
multi-year rollout driven by weekly rainfall against the observed year-on-year
change in biomass.

Two discretisations, one loss:

- [`SimConfig`](@ref): the fixed-step semi-implicit scheme of the Python code,
  term for term (implicit diffusion solved exactly with FFTW's DCT). Gradients by
  Enzyme reverse mode ([`EnzymeBackend`](@ref), the default) or ForwardDiff.
- [`ODEConfig`](@ref): the continuous PDE by method of lines with any
  DifferentialEquations.jl solver (extension, `using OrdinaryDiffEq…`), gradients
  by SciMLSensitivity adjoints with Enzyme VJPs (`AdjointODEBackend`, extension,
  `using SciMLSensitivity`) or ForwardDiff through the solver.

| file              | contents                                                     |
|:------------------|:-------------------------------------------------------------|
| `parameters.jl`   | [`RietkerkParams`](@ref), reference values, units, bounds    |
| `operators.jl`    | Laplacian, exact implicit diffusion (FFTW / dual / dense)    |
| `model.jl`        | [`step!`](@ref), [`simulate_year!`](@ref), viable random starts |
| `forcing.jl`      | weekly rainfall profiles                                     |
| `realdata.jl`     | GeoTIFF NDVI + ERA5 precipitation loading                    |
| `problem.jl`      | [`InverseProblem`](@ref), [`loss`](@ref), [`evaluate`](@ref) |
| `enzyme_rules.jl` | Enzyme rule for the FFTW diffusion solve                     |
| `gradients.jl`    | gradient backends                                            |
| `synthetic.jl`    | synthetic experiments                                        |
| `optim.jl`        | PyTorch-compatible Adam                                      |
| `train.jl`        | [`train`](@ref), [`train_many`](@ref)                        |
| `analysis.jl`     | Turing/composite diagnostics, run filters, bifurcation sweep |
| `io.jl`           | JSON/CSV/pickle interoperability with the Python results     |
| `diffeq.jl`       | [`ODEConfig`](@ref) and the method-of-lines right-hand side  |
"""
module InverseTuring

using Printf: @printf, @sprintf
import Random
import Statistics
import LinearAlgebra
import SparseArrays

import ForwardDiff
import DiffResults
import FFTW
import Enzyme
import EnzymeCore
import EnzymeCore.EnzymeRules
import ArchGDAL
import CSV
import DataFrames
import JSON
import Pickle

export RietkerkParams, PARAM_NAMES, NPARAMS, PARAM_SYMBOLS, PRETTY_NAMES, paramvector,
       paramdict, randparams, RIETKERK_2002, SYNTHETIC_TRUTH, REALDATA_REFERENCE,
       rietkerk_reference, realdata_reference, to_physical_units, parameter_bounds,
       log_bounds, clamp_to_bounds, degenerate_parameters, PLANT_TIMESCALE_FACTOR,
       DAYS_PER_YEAR, NDVI_TO_BIOMASS_MULTIPLIER, SYNTHETIC_PIXEL_SIZE_M, REALDATA_PIXEL_SIZE_M
export laplacian, laplacian!, implicit_diffusion!, DiffusionSolver,
       MatrixDiffusionSolver, diffusion_solver
export AbstractDiscretisation, SimConfig, SimState, SimWorkspace, simstate, workspace,
       copystate, copystate!, step!, simulate_week!, simulate_year!, simulate_years!,
       spectral_bound, max_stable_dt, keeps_vegetation, draw_viable_params
export sinusoidal_weekly_precip, summer_weekly_precip, uniform_weekly_precip, annual_total
export ndvi_biomass, YearObservation, SiteSeries, load_site, load_sites, biomass_stats,
       read_annual_precip, read_weekly_precip, years
export SiteTrajectory, InverseProblem, loss, trajectory_loss, evaluate, ntransitions,
       rethread, with_discretisation, viability_sites, mean_squared_delta_error,
       mean_squared_error
export AbstractGradientBackend, EnzymeBackend, ForwardDiffBackend, FiniteDiffBackend,
       AdjointODEBackend, gradient_cache, loss_and_gradient, loss_and_gradient!
export equilibrium_biomass, synthetic_series, synthetic_experiment, SYNTHETIC_PRESETS
export AdamState, adam_step!, clip_global_norm!, decay_lr!
export TrainConfig, REALDATA_TRAIN_CONFIG, SYNTHETIC_TRAIN_CONFIG, SYNTHETIC_1SITE_TRAIN_CONFIG,
       TrainResult, train, train_many
export homogeneous_steady_state, reaction_jacobian, growth_rates, turing_value,
       turing_wavelength, composite_value, mean_training_biomass,
       mean_training_precipitation, tier1_filter, drop_degenerate, agreement_table,
       bifurcation_sweep
export load_parameter_history, load_parameter_histories, read_parameter_table,
       params_from_row, write_parameter_table, save_run, load_run, save_json
export ODEConfig, rietkerk_rhs!, rhs_sparsity, ode_parameters, pack_state, unpack_state,
       biomass_of, solve_week!, simulate_years_ode

include("parameters.jl")
include("operators.jl")
include("model.jl")
include("forcing.jl")
include("realdata.jl")
include("problem.jl")
include("enzyme_rules.jl")
include("gradients.jl")
include("synthetic.jl")
include("optim.jl")
include("train.jl")
include("analysis.jl")
include("io.jl")
include("diffeq.jl")

end # module
