# InverseTuring.jl

A Julia implementation of the inverse-PDE pipeline in this repository: fitting the
eleven coefficients of the Rietkerk et al. (2002) dryland vegetation model to
satellite NDVI time series, and the synthetic, testing and analysis experiments
around it.

```
∂O/∂t = D_O ΔO + R − α O (B + k₂W₀)/(B + k₂)
∂W/∂t = D_W ΔW + α O (B + k₂W₀)/(B + k₂) − g_max W B/(W + k₁) − r_w W
∂B/∂t = D_P ΔB + c g_max W B/(W + k₁) − d B
```

It reproduces `rietkerk_model.py` exactly — the same semi-implicit scheme, units,
reference values, bounds, viability-screened starts and log-space Adam — and adds
what the Julia ecosystem offers on top: reverse-mode AD with **Enzyme**, the
continuous PDE through **DifferentialEquations.jl**, and continuous adjoints through
**SciMLSensitivity.jl**. The second half of this README is an evaluation of those
three libraries on this problem, with measurements.

## Contents

- [Quick start](#quick-start)
- [Python → Julia](#python--julia)
- [Validation against PyTorch](#validation-against-pytorch)
- [Enzyme, DifferentialEquations.jl and SciMLSensitivity.jl on this problem](#enzyme-differentialequationsjl-and-scimlsensitivityjl-on-this-problem)
- [Performance](#performance)
- [Numerics](#numerics)
- [Things worth knowing](#things-worth-knowing)
- [Layout](#layout)

## Quick start

Julia ≥ 1.12 (developed on 1.12.7). The continuous-model backends are package
extensions triggered by regular dependencies, which older Julia versions do not
load.

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia -t 2 julia/test/runtests.jl
```

(`Pkg.test()` works too, but first recompiles the dependencies under its own
compiler flags — bounds checking on — which takes a while.)

```julia
using InverseTuring

sites = load_sites("data", ["b", "i", "c", "e"])          # NDVI × 1500, weekly ERA5 rain
prob  = InverseProblem(sites, SimConfig(steps_per_week = 3))
L, g  = loss_and_gradient(prob, REALDATA_REFERENCE)         # Enzyme, reverse mode

result = train(prob; reference = REALDATA_REFERENCE, seed = 77)   # realdata_train_invPDE.py
turing_value(result.params, 0.665)        # < 0: uniform state Turing-unstable at 0.665 mm/day
```

The same problem on the continuous PDE, differentiated by SciMLSensitivity:

```julia
using OrdinaryDiffEqLowOrderRK, SciMLSensitivity
ode = with_discretisation(prob, ODEConfig(BS3(); abstol = 1e-3, reltol = 1e-4))
L, g = loss_and_gradient(ode, REALDATA_REFERENCE; backend = AdjointODEBackend())
```

Start Julia with `-t auto`: sites (one loss), restarts and sweep levels run in
parallel.

## Python → Julia

| Python | Julia | |
|:--|:--|:--|
| `rietkerk_model.py` | `src/` (`RietkerkParams`, `step!`, `simulate_year!`, `turing_value`, ...) | same names where possible |
| `realdata_train_invPDE.py` | `scripts/train_realdata.jl` | `--models START STOP`, skip/resume as in Python |
| `train_invPDE_synthetic_batch.py`, `..._1site.py` | `scripts/train_synthetic.jl --preset 4site\|1site` | |
| `realdata_test_invPDE.py` | `scripts/test_realdata.jl` | same `test_metrics.csv` / `test_summary.json` |
| `realdata_parameter_analysis.py` | `scripts/parameter_analysis.jl` | same `four_site_*` tables |
| `bifurcation_parallel.py`, `realdata_bifurcation_plot.py` | `scripts/bifurcation.jl` | levels run on threads |
| `invPDE_1site_extrapolation.py` | `scripts/extrapolation.jl` | also draws the true model |
| `measure_runtime.py` | `scripts/benchmark.jl` | |
| — | `scripts/diffeq_comparison.jl` | fixed-step scheme vs DifferentialEquations.jl |
| — | `scripts/validate_against_python.jl` | checks against PyTorch outputs |

Every script writes under `julia/results/`, mirroring `results/`
(`julia/results/real_data_rietkerk/models/...`), and reads Python outputs
(`.pkl` histories, parameter CSVs) when pointed at `results/`, so the two pipelines
can be mixed. Usage and options are in each script's header.

```bash
julia --project=julia -t auto julia/scripts/train_realdata.jl            # 10 models, 7500 epochs
julia --project=julia julia/scripts/parameter_analysis.jl
julia --project=julia -t auto julia/scripts/test_realdata.jl
julia --project=julia -t auto julia/scripts/bifurcation.jl
```

Not ported: data acquisition (`download_*.py`, Python-only APIs — keep using them;
they write the CSVs this package reads), the RCNN baseline, and figure-only scripts
(`compare_*.py`, `plot_*.py`, `view_site.py`), which read the same files the Julia
scripts write.

## Validation against PyTorch

`test/pytorch_reference.py` runs `rietkerk_model.py` itself in float64 on
deterministic inputs and stores the results in `test/data/pytorch_reference.json`;
the test suite rebuilds the inputs and compares, so the check needs no PyTorch at
test time:

| quantity | agreement |
|:--|:--|
| constants: references, unit conversions, bounds, `degenerate_parameters` | bit for bit |
| one semi-implicit and one explicit step | 1e-12 |
| a one-year rollout (104 steps) | 1e-11 |
| delta and absolute losses over three transitions, 2 and 3 steps/week | 1e-11 |
| their gradients with respect to the eleven log-parameters (autograd vs Enzyme) | 1e-8 |
| `turing_value`, `turing_wavelength`, steady state, Jacobian, composite | 1e-8 |
| held-out score of the reference parameters on site f (`evaluate_model_on_site`) | 1e-10 |

At full scale, the real-data loss at the reference parameters (4 sites × 9
transitions) is 3400.0414 in both PyTorch float64 and Julia. Inside Julia, the
Enzyme gradient matches ForwardDiff to 1e-13 and finite differences to 1e-6, and
the SciMLSensitivity adjoints match ForwardDiff through the ODE solver to 1e-12.
`scripts/validate_against_python.jl` repeats the comparison against the artefacts
the Python scripts write to `results/real_data_rietkerk/` (not in the repository),
when they exist.

PyTorch's random number stream cannot be reproduced, so random starts and synthetic
noise differ between the two implementations for the same seed; everything
deterministic agrees.

## Enzyme, DifferentialEquations.jl and SciMLSensitivity.jl on this problem

The three libraries do different jobs here:

| library | used for | code |
|:--|:--|:--|
| Enzyme | reverse-mode gradients of the fixed-step scheme (the model the Python code fits); the vector–Jacobian products inside SciMLSensitivity | `src/gradients.jl`, `src/enzyme_rules.jl` |
| DifferentialEquations.jl | the continuous PDE: method of lines, adaptive time steps with error control | `src/diffeq.jl`, `ext/InverseTuringDiffEqExt.jl` |
| SciMLSensitivity.jl | gradients of the continuous PDE by adjoint ODE solves | `ext/InverseTuringSciMLSensitivityExt.jl` |

**In short:** fit with the fixed-step scheme and Enzyme, as the Python code does —
it is by far the cheapest gradient — and use DifferentialEquations.jl to check
what that scheme costs in accuracy. It costs more than one would guess: at the step
sizes the scripts use, the scheme's gradient is 6–8 % away from that of the PDE it
discretises, so a fit recovers the parameters of the discretised model. Where that
matters, SciMLSensitivity gives the gradient of the PDE itself, accurate to 1e-5
or better, for 8–30 times the cost of the fixed-step gradient — less than refining
the fixed-step scheme to even 1e-3 would cost.

The measurements below come from `scripts/diffeq_comparison.jl`, on one transition
(one site, one year) of two setups, at parameters ×0.7–1.4 away from the reference
so that the gradient is not small. Errors are relative to the PDE solved at tight
tolerance (Tsit5, reltol 1e-10) with its gradient from the interpolating adjoint;
the gradient error is ‖g − g_ref‖/‖g_ref‖ over the eleven parameter derivatives.
Times are wall-clock seconds on one thread, compilation excluded; they vary by
10–20 % between runs. Adjoints use Enzyme for the vector–Jacobian products; the
implicit solvers get the Jacobian's sparsity pattern.

**Real data**: site b, 131×140 cells of 30 m. Adaptive solvers at `abstol = 1e-3,
reltol = 1e-4` (biomass is up to 1500).

| method | loss (s) | loss error | gradient (s) | gradient error |
|:--|--:|--:|--:|--:|
| fixed-step, 1 step/week | 0.041 | 1.6e-2 | 0.18 | 1.7e-1 |
| fixed-step, 2 steps/week | 0.083 | 7.9e-3 | 0.44 | 8.6e-2 |
| **fixed-step, 3 steps/week (Python)** | 0.12 | 5.3e-3 | 0.66 | 5.8e-2 |
| fixed-step, 4 steps/week | 0.16 | 3.9e-3 | 0.90 | 4.4e-2 |
| fixed-step, 8 steps/week | 0.45 | 2.0e-3 | 2.03 | 2.2e-2 |
| fixed-step, 16 steps/week | 0.59 | 9.9e-4 | 5.17 | 1.1e-2 |
| fixed-step, 48 steps/week | 1.48 | 3.3e-4 | 7.39 | 3.7e-3 |
| fixed-step, 3 steps/week, ForwardDiff gradient | | | 7.60 | 5.8e-2 |
| BS3 | 0.91 | 9.1e-9 | | |
| &nbsp;&nbsp;+ InterpolatingAdjoint | | | 6.36 | 4.4e-6 |
| &nbsp;&nbsp;+ GaussAdjoint | | | 5.11 | 6.7e-6 |
| &nbsp;&nbsp;+ QuadratureAdjoint | | | 6.64 | 6.5e-6 |
| &nbsp;&nbsp;+ ForwardDiff through the solver | | | 19.7 | 9.1e-7 |
| Tsit5 | 0.74 | 6.1e-9 | | |
| &nbsp;&nbsp;+ InterpolatingAdjoint | | | 8.05 | 1.2e-8 |
| &nbsp;&nbsp;+ GaussAdjoint | | | 6.85 | 8.9e-7 |
| &nbsp;&nbsp;+ QuadratureAdjoint | | | 7.60 | 9.3e-7 |
| ROCK2 | 2.50 | 2.2e-6 | | |
| ROCK4 | 1.02 | 2.0e-8 | | |
| KenCarp47 (sparse J) | 449 | 2.4e-5 | | |
| Rodas5P (sparse J) | 172 | 4.6e-9 | | |
| FBDF (sparse J) | 559 | 3.4e-7 | | |

**Synthetic**: a 64×64 equilibrium field on 5 m cells, one year of the synthetic
rain. Surface-water diffusion makes this one stiff: its fastest rate is 8·D_O ≈ 40
per day, against ~1 per day on 30 m cells. Adaptive solvers at `abstol = 1e-5,
reltol = 1e-4`.

| method | loss (s) | loss error | gradient (s) | gradient error |
|:--|--:|--:|--:|--:|
| fixed-step, 1 step/week | 0.012 | 1.0e-1 | 0.049 | 1.5e-1 |
| **fixed-step, 2 steps/week (Python)** | 0.020 | 5.5e-2 | 0.12 | 7.9e-2 |
| fixed-step, 3 steps/week | 0.041 | 3.7e-2 | 0.15 | 5.4e-2 |
| fixed-step, 4 steps/week | 0.044 | 2.8e-2 | 0.17 | 4.1e-2 |
| fixed-step, 8 steps/week | 0.086 | 1.4e-2 | 0.43 | 2.1e-2 |
| fixed-step, 16 steps/week | 0.15 | 7.1e-3 | 0.73 | 1.0e-2 |
| fixed-step, 48 steps/week | 0.44 | 2.4e-3 | 2.23 | 3.5e-3 |
| fixed-step, 2 steps/week, ForwardDiff gradient | | | 1.12 | 7.9e-2 |
| BS3 | 1.49 | 7.1e-11 | | |
| Tsit5 | 2.51 | 3.3e-13 | | |
| &nbsp;&nbsp;+ InterpolatingAdjoint | | | 20.5 | 2.1e-6 |
| &nbsp;&nbsp;+ GaussAdjoint | | | 21.8 | 1.8e-6 |
| &nbsp;&nbsp;+ QuadratureAdjoint | | | 24.5 | 1.4e-6 |
| ROCK2 | 0.65 | 2.5e-6 | | |
| ROCK4 | 0.52 | 5.6e-9 | | |
| &nbsp;&nbsp;+ InterpolatingAdjoint | | | 3.08 | 6.5e-8 |
| &nbsp;&nbsp;+ GaussAdjoint | | | 4.06 | 2.6e-7 |
| &nbsp;&nbsp;+ QuadratureAdjoint | | | 2.81 | 2.5e-7 |
| &nbsp;&nbsp;+ ForwardDiff through the solver | | | 18.5 | 4.1e-9 |
| KenCarp47 (sparse J) | 28.6 | 6.5e-8 | | |
| Rodas5P (sparse J) | 22.6 | 1.7e-9 | | |
| FBDF (sparse J) | 31.0 | 4.1e-7 | | |

### Enzyme

- **What it needed.** Enzyme differentiates the loop kernels that run the forward
  model, clamps included, so there is no second, differentiable copy of the model to
  keep in sync. The exception is the diffusion solve: it calls FFTW, which Enzyme
  cannot see into, so it has a reverse rule (`src/enzyme_rules.jl`, tested against
  Enzyme differentiating a dense-matrix solve that needs no rule, and against
  finite differences).
- **Cost.** A step's vector–Jacobian product costs 2.8–3.5× the step and allocates
  nothing. A full gradient costs about five losses: the forward pass, recomputing
  each week from its checkpoint, and the products. ForwardDiff takes ten to twelve
  times as long for the eleven parameters, because every partial rides through
  every FFT.
- **Accuracy.** Matches ForwardDiff to 1e-13 and PyTorch's autograd to 1e-8.
- **Caveats.** The first gradient in a session compiles for 1.5–2 minutes. And with
  Enzyme 0.13.209 on Julia 1.12.7, differentiating a runtime-length loop around the
  step function — a whole week per `autodiff` call — corrupted the heap after about
  4000 calls (the garbage collector then segfaults), with or without threads; a
  minimal loop of the same shape did not. Differentiating single steps, which is
  what `EnzymeBackend` does, ran 1.25 million calls (300 four-site gradients) on one
  thread and 150 gradients on three threads without a fault. The per-step design
  costs nothing, since checkpointing needs the step boundaries anyway.

### DifferentialEquations.jl

- **What it showed.** The fixed-step scheme is first order — the error halves when
  the step does — and at the scripts' settings it is not small: at 3 steps per week
  on site b the loss is off by 0.5 % and the gradient by 6 %; at 2 steps per week on
  5 m cells, by 5 % and 8 %. The synthetic experiments are self-consistent (data and
  fit use the same scheme), so they still recover their truth; the error matters
  when fitted values are compared with Rietkerk's, or with fits at another step
  size or resolution.
- **Which solver.** On 30 m cells the system is not stiff and the explicit adaptive
  methods do best: `Tsit5` or `BS3` solve a year in 0.7–0.9 s to 1e-8, against
  0.12 s for the fixed-step scheme at 5e-3. On 5 m cells it is stiff, and the
  stabilised explicit `ROCK4` is fastest (0.5 s to 6e-9), ahead of `Tsit5`, whose
  steps are then limited by stability rather than accuracy. Implicit solvers
  (`KenCarp47`, `Rodas5P`, `FBDF`) work once given the Jacobian's sparsity pattern
  (`ODEConfig(alg; sparse_jacobian = true)`), but every step factorises a sparse
  matrix with 3HW rows: they are 40–60× slower than `ROCK4` on the stiff grid and
  170–550× slower on the real one.
- **Forcing.** Rain changes weekly, so each week is its own ODE segment (see
  [Numerics](#numerics)); a `reinit!`-ed integrator keeps that cheap.

### SciMLSensitivity.jl

- **What it needed.** `InterpolatingAdjoint(autojacvec = EnzymeVJP())` works as is.
  `GaussAdjoint` and `QuadratureAdjoint` stop with an `EnzymeRuntimeActivityError`
  unless Enzyme's runtime activity is on:
  `EnzymeVJP(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))`.
- **Composition.** The loss is a sum over year ends and the forcing changes every
  week, so the gradient is assembled week by week: the forward pass stores the
  state at each week boundary; going backwards, each week is re-solved densely from
  its checkpoint and one adjoint solve, seeded at the week's end through
  `dgdu_discrete`, carries the adjoint to the week's start. Keeping every week's
  dense solution instead would take tens of gigabytes.
- **Accuracy.** The three adjoints agree with ForwardDiff through the solver to
  ~1e-12 when both solve tightly, and with the tight reference to 1e-5 or better at
  the loose tolerances above.
- **Cost.** On site b an adjoint gradient takes 5–8 s (error 1e-8–7e-6), against
  0.66 s for the fixed-step Enzyme gradient (6e-2) and 7.4 s for the fixed-step
  gradient at 48 steps per week, which is still off by 4e-3. On the stiff synthetic
  grid, `ROCK4` with an adjoint takes 3–4 s (≤ 3e-7) against 0.12 s for the
  fixed-step gradient (8e-2). The three adjoints cost about the same here.
  ForwardDiff through the solver works too, at three to six times the cost of an
  adjoint.

### Recommendation

1. **Fit** with `SimConfig(steps_per_week = 3)` and `EnzymeBackend()` (the
   defaults), as the Python pipeline does.
2. **Check** a fit on the PDE: `loss(θ, with_discretisation(prob, ODEConfig(BS3();
   abstol = 1e-3, reltol = 1e-4)))` is one forward solve.
3. **Finish on the PDE** when the fitted values are to mean what Rietkerk's mean,
   or be compared across step sizes or resolutions: continue the fit with
   `train(with_discretisation(prob, ODEConfig(...)), result.params; backend =
   AdjointODEBackend(), ...)`. Use `BS3`/`Tsit5` on 30 m cells and `ROCK4` on 5 m
   cells, with `abstol` at the scale of the data (biomass is up to 1500).

## Performance

Seconds per training epoch (loss, gradient, Adam step) — what `measure_runtime.py`
measures, repeated by `scripts/benchmark.jl` — on a 4-core Intel Xeon @ 2.8 GHz:

| experiment | Julia, 4 threads | Julia, 1 thread | PyTorch, same CPU | PyTorch, A100 |
|:--|--:|--:|--:|--:|
| real data: 4 sites × 9 transitions, 3 steps/week | 14.0 | 30.7 | 59.3 | 12.1 |
| synthetic, 4 sites: 128×128, 10 transitions, 2 steps/week | 4.2 | | | 3.8 |
| synthetic, 1 site | 4.3 | | | 0.97 |

The A100 column is `runtime_estimates.csv` in the repository root, recorded with
`measure_runtime.py`. The PyTorch CPU figure is PyTorch 2.14 in float32 on four
threads with gradient checkpointing on (90.9 s in float64).

A full real-data fit (7500 epochs) thus takes ~29 h on these four cores: about
what the A100 needs (25 h), and a quarter of PyTorch's time on the same CPU. The
four-site synthetic fit keeps pace with the A100; the one-site fit, with no sites
to spread over threads, is about four times slower than on the GPU.

- **Where the time goes.** On site b (131×140) a step takes 0.70 ms and its
  vector–Jacobian product 2.47 ms; on site c (148×211), 2.01 and 5.72 ms. A
  gradient costs 4.5–5.5 losses. Neither allocates.
- **Memory.** The weekly checkpoints of the four real-data sites over nine years
  take ~0.9 GB (PyTorch without gradient checkpointing peaks at 6.8 GB on the GPU).
- **Threads.** Start Julia with `-t auto`. The sites of one loss run in parallel
  (`InverseProblem(...; threaded = true)`, the default with more than one thread),
  `train_many(...; parallel = :runs)` runs restarts in parallel, and
  `bifurcation_sweep` its rain levels. A single site uses one core.
- **Compilation.** The first loss of a session compiles in seconds, the first Enzyme
  gradient in 1.5–2 minutes, the first adjoint gradient in about a minute.

## Numerics

**Implicit diffusion is solved exactly, three ways.** The semi-implicit step solves
`(I − κΔ) x = u` for each field, with Δ the replicate-boundary 5-point Laplacian.
Python applies the orthonormal DCT as two dense matrix products. Here:

- the DCT is applied **per axis** with FFTW or with a dense product split by the
  transform's even/odd symmetry (half the flops), whichever is faster for the
  axis length. FFTW is slow for lengths with large prime factors — a 131×140 DCT
  takes 6× longer than a 128×128 one — and most sites have such a dimension (131,
  211 and 71 are prime; 129 = 3·43, 148 = 4·37, 122 = 2·61);
- when `8κ` is small the **Neumann series** `Σ (κΔ)ⁿ` reaches machine precision in a
  few stencil applications. At the real-data reference that is the case for soil
  water (8κ ≈ 2e-3) and biomass (8κ ≈ 1e-4); only surface water needs the
  transform, so a step costs about one DCT solve instead of three.

All three give the same operator to rounding (tested against the dense matrix
form, including Enzyme gradients through each).

**One hand-written Enzyme rule.** FFTW and BLAS are opaque to Enzyme, so the solve
has a reverse rule (`src/enzyme_rules.jl`): the operator is symmetric, so
`ū += A x̄`, and since `dA/dκ = AΔA`, `κ̄ = (A x̄)ᵀ Δ x` — no spectral bookkeeping, valid
for both paths. Everything else — the nonlinear source and loss terms with their
clamps, the explicit scheme, the RHS of the ODE inside SciMLSensitivity — is
differentiated by Enzyme itself.

**Discrete adjoint, checkpointed.** `EnzymeBackend` checkpoints the state at every
week boundary (three fields per week, ~200 MB for a 131×140 site over nine years),
then sweeps backwards: recompute the week's step boundaries, one Enzyme VJP per
step, and the year-end loss terms seeded into the biomass adjoint. This is the
recomputation schedule of PyTorch's `gradient_checkpointing=True`, without an
autograd graph.

**Continuous model by method of lines.** `ODEConfig` integrates the same
semi-discrete system the fixed-step scheme discretises (`rietkerk_rhs!`, including
its clamps at zero), so the fixed-step scheme converges to it as `dt → 0`.
Rain is piecewise constant per week, so each week is its own ODE segment: an
adaptive solver must not step across the jump, and with a time-dependent forcing
plus `tstops` the last stages of a step that ends on a boundary would still see the
next week's rate. One integrator is `reinit!`-ed per week.

## Things worth knowing

**The fixed-step scheme is a first-order approximation of the PDE, and not a
close one at the steps the scripts use**: its gradient is 6–8 % off the PDE's (see
[the evaluation](#enzyme-differentialequationsjl-and-scimlsensitivityjl-on-this-problem)).
The synthetic experiments are self-consistent (data and fit use the same scheme),
but the parameters the real-data fits recover are those of the discretised model,
and comparing them with Rietkerk's per-day values carries that error.
`with_discretisation(prob, ODEConfig(...))` evaluates or continues any fit on the
PDE itself.

**Precision.** `Float64` throughout by default (PyTorch trains in float32). The
solver, the step and the Enzyme rule also run in `Float32`
(`load_site(...; T = Float32)`); the gradient backends use `Float64`.

**`summer_weekly_precip` delivers 99.73 % of the requested rain.** It normalises
with `365/52` days per week while the model delivers 7 days per week — inherited
from `bifurcation_parallel.py` and kept so that bifurcation diagrams mean the same
thing in both implementations (`days_per_week = 7` drops it).

**Negative weekly rain.** ERA5 accumulation differencing gives small negative rates
in some dry weeks. They are passed to the model unchanged, as in Python.

**Parameter histories are JSON**, not pickle; `load_parameter_history` reads both,
so the analysis scripts accept Python and Julia runs alike. The last snapshot of a
run that stops early is labelled with the epoch that produced it (Python labels it
`num_epochs − 1`).

## Layout

```
src/
  parameters.jl    RietkerkParams, references, units, bounds, degenerate_parameters
  operators.jl     Laplacian; exact implicit diffusion (per-axis DCT, Neumann series, dense)
  model.jl         SimConfig, step!, simulate_year!, viability screening
  forcing.jl       weekly rainfall profiles
  realdata.jl      GeoTIFF NDVI + ERA5 precipitation loading
  problem.jl       InverseProblem, loss, evaluate
  enzyme_rules.jl  the Enzyme rule for the diffusion solve
  gradients.jl     EnzymeBackend, ForwardDiffBackend, FiniteDiffBackend
  synthetic.jl     synthetic experiments (both presets)
  optim.jl         PyTorch-compatible Adam, clipping, decay
  train.jl         train, train_many, checkpoint/resume
  analysis.jl      Turing and composite diagnostics, run filters, bifurcation sweep
  io.jl            JSON/CSV/pickle interoperability
  diffeq.jl        ODEConfig, method-of-lines RHS, AdjointODEBackend
ext/
  InverseTuringDiffEqExt.jl            ODEConfig solves (any OrdinaryDiffEq solver)
  InverseTuringSciMLSensitivityExt.jl  AdjointODEBackend (SciMLSensitivity + Enzyme VJPs)
scripts/           command-line entry points (see the table above)
test/              runtests.jl, pytorch_reference.py and its JSON
```
