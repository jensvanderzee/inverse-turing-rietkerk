"""
    equilibrium_biomass(p, grid; precipitation, years, cfg, rng, T = Float64) -> Matrix{T}

Spin the model up to a patterned state under uniform rain and return the biomass
field. All three fields start from `Uniform[0, 10)` noise (drawn in the order O, W,
B); the pattern that emerges is the Turing instability the study is about, so the
result depends on `rng`. Port of `SyntheticDataGenerator.generate_equilibrium_state`.
"""
function equilibrium_biomass(p::RietkerkParams, grid::Tuple{Int,Int};
                             precipitation::Real, years::Integer, cfg::SimConfig,
                             rng::Random.AbstractRNG = Random.default_rng(), T::Type = Float64)
    H, W = grid
    O = T.(rand(rng, H, W) .* 10)
    Wf = T.(rand(rng, H, W) .* 10)
    B = T.(rand(rng, H, W) .* 10)
    state = SimState(O, Wf, B)
    weekly = sinusoidal_weekly_precip(precipitation; peak_week = 26.0, amplitude_fraction = 0.0)
    simulate_years!(state, p, weekly, cfg, years)
    return copy(state.biomass)
end

"""
    synthetic_series(p, equilibrium, weekly_precip; years, noise_level, cfg, rng)
        -> Vector{Matrix}

One site's observations: the true model run from `equilibrium` (with `O = W = 0`)
for `years` years, recording the biomass after each year plus Gaussian noise of
standard deviation `noise_level`. Only the record is noisy; the trajectory stays
clean, as in `SyntheticDataGenerator.generate_training_data`.
"""
function synthetic_series(p::RietkerkParams, equilibrium::AbstractMatrix,
                          weekly_precip::AbstractVector; years::Integer, noise_level::Real,
                          cfg::SimConfig, rng::Random.AbstractRNG = Random.default_rng())
    T = eltype(equilibrium)
    state = simstate(equilibrium)
    ws = workspace(state)
    out = Matrix{T}[]
    for _ in 1:years
        simulate_year!(state, p, weekly_precip, cfg, ws)
        push!(out, state.biomass .+ T(noise_level) .* randn(rng, T, size(state.biomass)))
    end
    return out
end

"""
Settings of the two published synthetic experiments.

`:four_site` is `train_invPDE_synthetic_batch.py`: spin-up at 400 mm/yr (inside
Rietkerk's Turing range), four sites at 286–514 mm/yr. `:one_site` is
`train_invPDE_synthetic_batch_1site.py`: the same spin-up, one site at 347 mm/yr.
Both use 2 steps per week, 10 years, noise 0.05, and sum the per-transition MSEs.
"""
const SYNTHETIC_PRESETS = Dict(
    :four_site => (rain_multiplier = 400.0 / 21, equilibrium_precip = 21.0,
                   annual_range = (15.0, 27.0), n_sites = 4),
    :one_site => (rain_multiplier = 400.0 / 19, equilibrium_precip = 19.0,
                  annual_range = (16.5, 24.5), n_sites = 1),
)

"""
    synthetic_experiment(; preset = :four_site, p = SYNTHETIC_TRUTH, grid = (128, 128),
                         steps_per_week = 2, equilibrium_years = 100, years = 10,
                         noise_level = 0.05, peak_week = 26.0, amplitude_fraction = 0.7,
                         rng = Random.default_rng(), T = Float64)

Generate one of the synthetic benchmarks (see [`SYNTHETIC_PRESETS`](@ref)): spin up
one equilibrium field, then roll out each site under its seasonal forcing. Returns
a named tuple with the [`InverseProblem`](@ref) (summed loss, as in the Python
scripts), the equilibrium field, the forcing profiles, the noisy targets and the
annual totals.

The Python scripts draw from PyTorch's RNG, which cannot be reproduced; the
experiment is the same, the noise realisation is not.
"""
function synthetic_experiment(; preset::Symbol = :four_site, p::RietkerkParams = SYNTHETIC_TRUTH,
                              grid::Tuple{Int,Int} = (128, 128), steps_per_week::Integer = 2,
                              equilibrium_years::Integer = 100, years::Integer = 10,
                              noise_level::Real = 0.05, peak_week::Real = 26.0,
                              amplitude_fraction::Real = 0.7,
                              rng::Random.AbstractRNG = Random.default_rng(), T::Type = Float64,
                              threaded::Bool = Threads.nthreads() > 1)
    haskey(SYNTHETIC_PRESETS, preset) ||
        throw(ArgumentError("unknown preset :$preset; use one of $(collect(keys(SYNTHETIC_PRESETS)))"))
    s = SYNTHETIC_PRESETS[preset]
    cfg = SimConfig(steps_per_week = steps_per_week)
    rm = s.rain_multiplier
    equilibrium = equilibrium_biomass(p, grid; precipitation = s.equilibrium_precip * rm,
                                      years = equilibrium_years, cfg = cfg, rng = rng, T = T)
    lo, hi = s.annual_range
    totals = s.n_sites == 1 ? [lo * rm] : collect(range(lo * rm, hi * rm; length = s.n_sites))
    profiles = [sinusoidal_weekly_precip(t; peak_week = peak_week,
                                         amplitude_fraction = amplitude_fraction) for t in totals]
    targets = [synthetic_series(p, equilibrium, prof; years = years, noise_level = noise_level,
                                cfg = cfg, rng = rng) for prof in profiles]
    trajectories = [SiteTrajectory(copy(equilibrium), copy(equilibrium), fill(prof, years), tgt)
                    for (prof, tgt) in zip(profiles, targets)]
    problem = InverseProblem(trajectories, cfg; average = false, threaded = threaded)
    return (problem = problem, equilibrium = equilibrium, profiles = profiles,
            targets = targets, annual_totals = totals, cfg = cfg)
end
