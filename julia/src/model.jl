"""
    basefloat(x)

The underlying floating-point type of a value, seeing through `ForwardDiff.Dual`
nesting. Used to round `dt` and the rainfall rate to the working precision before
they enter the update, which is what PyTorch does when a Python `float` meets a
`float32` tensor.
"""
basefloat(::Type{T}) where {T<:AbstractFloat} = T
basefloat(::Type{<:ForwardDiff.Dual{<:Any,V,<:Any}}) where {V} = basefloat(V)
basefloat(x) = basefloat(typeof(x))

"""
    AbstractDiscretisation

How a year of the PDE is integrated: [`SimConfig`](@ref) (the fixed-step scheme of
the Python code) or [`ODEConfig`](@ref) (the continuous model through
DifferentialEquations.jl).
"""
abstract type AbstractDiscretisation end

"""
    SimConfig(; steps_per_week, year_length = 364.0, semi_implicit = true)

Time discretisation of one simulated year, as `simulate_year_weekly` takes it.

- `steps_per_week`: sub-steps per forcing week. Training uses 3 on the real data
  and 2 on the synthetic data; testing, simulation and the bifurcation sweep use 4.
- `year_length`: days per simulated year. The default, 52 weeks of 7 days, makes
  the injected rate exactly the weekly value in mm/day.
- `semi_implicit`: the unconditionally stable semi-implicit step (default) or the
  old explicit Euler step (`semi_implicit=False` in Python).

The step is `dt = year_length / (num_weeks · steps_per_week)` days.
"""
Base.@kwdef struct SimConfig <: AbstractDiscretisation
    steps_per_week::Int
    year_length::Float64 = DAYS_PER_YEAR
    semi_implicit::Bool = true
end

describe(cfg::SimConfig) = "$(cfg.steps_per_week) steps/week" *
                           (cfg.semi_implicit ? "" : " (explicit)")

"""Time step (days) of a year with `nweeks` forcing weeks."""
step_size(cfg::SimConfig, nweeks::Integer) = cfg.year_length / (nweeks * cfg.steps_per_week)

"""
Rate (mm/day) injected during a week whose rainfall is `weekly_rate` mm/day: the
week's `7·weekly_rate` mm split evenly over its sub-steps, then divided by `dt` —
computed in the same order as `simulate_year_weekly`.
"""
week_rate(weekly_rate::Real, cfg::SimConfig, nweeks::Integer) =
    (weekly_rate * 7.0 / cfg.steps_per_week) / step_size(cfg, nweeks)

"""
    SimState(surface_water, soil_water, biomass)

The three coupled fields `O`, `W`, `B`. Build one from an initial biomass field with
[`simstate`](@ref).
"""
struct SimState{T,A<:AbstractMatrix{T}}
    surface_water::A
    soil_water::A
    biomass::A

    function SimState(surface_water::A, soil_water::A, biomass::A) where {T,A<:AbstractMatrix{T}}
        size(surface_water) == size(soil_water) == size(biomass) ||
            throw(DimensionMismatch("all three fields must have the same size"))
        new{T,A}(surface_water, soil_water, biomass)
    end
end

"""
    simstate(biomass; T = eltype(biomass))

State with surface and soil water at zero — the initialisation every training,
testing and simulation script uses. Passing `T` promotes the state (to
`ForwardDiff.Dual` for a gradient, to `Float32` to mimic PyTorch).
"""
function simstate(biomass::AbstractMatrix; T::Type = eltype(biomass))
    b = convert.(T, biomass)
    SimState(zero(b), zero(b), b)
end

Base.size(s::SimState) = size(s.biomass)
Base.eltype(::SimState{T}) where {T} = T

"""Deep copy of a state."""
copystate(s::SimState) = SimState(copy(s.surface_water), copy(s.soil_water), copy(s.biomass))

"""Overwrite `dst` with `src`, field by field."""
function copystate!(dst::SimState, src::SimState)
    copyto!(dst.surface_water, src.surface_water)
    copyto!(dst.soil_water, src.soil_water)
    copyto!(dst.biomass, src.biomass)
    return dst
end

"""
    SimWorkspace(T, H, W)
    workspace(state)

Scratch space for [`step!`](@ref): one field-sized buffer and the diffusion solver
for the grid. One workspace per concurrently running rollout.
"""
struct SimWorkspace{T,A<:AbstractMatrix{T},S}
    tmp::A
    solver::S
end

SimWorkspace(::Type{T}, H::Integer, W::Integer) where {T} =
    SimWorkspace(zeros(T, H, W), diffusion_solver(T, H, W))

workspace(s::SimState) = SimWorkspace(eltype(s), size(s)...)

"""
    step!(state, p, rate, dt, ws; semi_implicit = true) -> state

One step of length `dt` days of

    ∂O/∂t = D_O ΔO + R − α O (B + k₂W₀)/(B + k₂)
    ∂W/∂t = D_W ΔW + α O (B + k₂W₀)/(B + k₂) − g_max W B/(W + k₁) − r_w W
    ∂B/∂t = D_P ΔB + c g_max W B/(W + k₁) − d B

with rainfall `rate` in mm/day. Port of `invRietkerk.forward`.

Semi-implicit (default): sources (rain, infiltration into the soil, growth) are
explicit; each field's own loss (infiltration out of the surface, uptake + soil
water loss, mortality) and its diffusion are implicit, the latter solved exactly in
the DCT basis. Every factor is positive, so the fields stay non-negative and the
step is stable for any parameter values; a uniform equilibrium is an exact fixed
point. Explicit: the old forward Euler step.

!!! important "The update is sequential"
    `O` is advanced first, `W` uses the new `O`, `B` uses the new `W` — the order
    of the Python code. Infiltration, uptake and growth use biomass clamped at zero,
    and uptake/growth soil water clamped at zero, exactly as there.
"""
function step!(s::SimState, p::RietkerkParams, rate, dt, ws::SimWorkspace;
               semi_implicit::Bool = true)
    if semi_implicit
        semi_implicit_step!(s, p, rate, dt, ws.tmp, ws.solver)
    else
        explicit_step!(s, p, rate, dt, ws.tmp)
    end
    return s
end

# The two kernels take the scratch buffer and the solver as separate arguments so
# that Enzyme sees the buffer as active memory and the solver as a constant. The
# pointwise updates are written as plain loops rather than broadcasts: same code
# for the primal, but Enzyme's reverse pass through a loop is several times faster
# than through the broadcast machinery.

# α I(B): infiltration capacity per unit surface water, with biomass clamped at zero.
@inline _infiltration(α, k2, k2w0, B) = (b = max(B, zero(B)); α * (b + k2w0) / (b + k2))

function semi_implicit_step!(s::SimState, p::RietkerkParams, R, dt, tmp::AbstractMatrix, solver)
    O, W, B = s.surface_water, s.soil_water, s.biomass
    α = p.infiltration_rate
    k2 = p.infiltration_half_saturation
    k2w0 = k2 * p.bare_soil_infiltration
    gmax = p.plant_uptake_rate
    k1 = p.uptake_half_saturation
    rw = p.seepage_rate
    c = p.water_use_efficiency
    d = p.mortality_rate

    # Surface water: rain in; infiltration out, implicit; diffusion implicit.
    @inbounds for i in eachindex(tmp, O, B)
        tmp[i] = (O[i] + dt * R) / (1 + dt * _infiltration(α, k2, k2w0, B[i]))
    end
    implicit_diffusion!(O, tmp, dt * p.surface_water_diffusion_coeff, solver)

    # Soil water: infiltration of the new surface water in; uptake and loss out, implicit.
    @inbounds for i in eachindex(tmp, O, W, B)
        b = max(B[i], zero(B[i]))
        w = max(W[i], zero(W[i]))
        tmp[i] = (W[i] + dt * (_infiltration(α, k2, k2w0, B[i]) * O[i])) /
                 (1 + dt * (gmax * b / (w + k1) + rw))
    end
    implicit_diffusion!(W, tmp, dt * p.soil_water_diffusion_coeff, solver)

    # Biomass: growth from the new soil water; mortality implicit.
    @inbounds for i in eachindex(tmp, W, B)
        b = max(B[i], zero(B[i]))
        w = max(W[i], zero(W[i]))
        tmp[i] = (B[i] + dt * (c * gmax * w * b / (w + k1))) / (1 + dt * d)
    end
    implicit_diffusion!(B, tmp, dt * p.biomass_diffusion_coeff, solver)
    return nothing
end

function explicit_step!(s::SimState, p::RietkerkParams, R, dt, tmp::AbstractMatrix)
    O, W, B = s.surface_water, s.soil_water, s.biomass
    α = p.infiltration_rate
    k2 = p.infiltration_half_saturation
    k2w0 = k2 * p.bare_soil_infiltration
    gmax = p.plant_uptake_rate
    k1 = p.uptake_half_saturation
    rw = p.seepage_rate
    c = p.water_use_efficiency
    d = p.mortality_rate
    DO = p.surface_water_diffusion_coeff
    DW = p.soil_water_diffusion_coeff
    DP = p.biomass_diffusion_coeff

    # Each cell's old value is read before it is written, and the Laplacian is
    # taken before the update, so this is forward Euler field by field.
    laplacian!(tmp, O)
    @inbounds for i in eachindex(tmp, O, B)
        O[i] = O[i] + dt * (DO * tmp[i] + R - _infiltration(α, k2, k2w0, B[i]) * O[i])
    end
    laplacian!(tmp, W)
    @inbounds for i in eachindex(tmp, O, W, B)
        b = max(B[i], zero(B[i]))
        w = max(W[i], zero(W[i]))
        W[i] = W[i] + dt * (DW * tmp[i] + _infiltration(α, k2, k2w0, B[i]) * O[i] -
                            (gmax * b / (w + k1) + rw) * W[i])
    end
    laplacian!(tmp, B)
    @inbounds for i in eachindex(tmp, W, B)
        b = max(B[i], zero(B[i]))
        w = max(W[i], zero(W[i]))
        B[i] = B[i] + dt * (DP * tmp[i] + c * gmax * w * b / (w + k1) - d * B[i])
    end
    return nothing
end

"""
    simulate_week!(state, p, rate, dt, nsteps, tmp, solver, semi_implicit) -> nothing

`nsteps` steps at a constant rainfall rate: one forcing week, the unit between the
checkpoints of the Enzyme gradient.
"""
function simulate_week!(s::SimState, p::RietkerkParams, rate, dt, nsteps::Int,
                        tmp::AbstractMatrix, solver, semi_implicit::Bool)
    for _ in 1:nsteps
        if semi_implicit
            semi_implicit_step!(s, p, rate, dt, tmp, solver)
        else
            explicit_step!(s, p, rate, dt, tmp)
        end
    end
    return nothing
end

"""
    simulate_year!(state, p, weekly_precip, cfg[, ws]) -> state

Advance the state by one year under weekly forcing. `weekly_precip[w]` is the rate
(mm/day) of week `w`; each week delivers `7·weekly_precip[w]` mm, split evenly
over `cfg.steps_per_week` sub-steps. Port of `simulate_year_weekly`.
"""
function simulate_year!(s::SimState, p::RietkerkParams, weekly_precip::AbstractVector,
                        cfg::SimConfig, ws::SimWorkspace = workspace(s))
    nweeks = length(weekly_precip)
    nweeks > 0 || throw(ArgumentError("weekly_precip must be non-empty"))
    cfg.steps_per_week > 0 || throw(ArgumentError("steps_per_week must be positive"))
    F = basefloat(eltype(s))
    dt = F(step_size(cfg, nweeks))
    for w in 1:nweeks
        rate = F(week_rate(weekly_precip[w], cfg, nweeks))
        simulate_week!(s, p, rate, dt, cfg.steps_per_week, ws.tmp, ws.solver, cfg.semi_implicit)
    end
    return s
end

"""
    simulate_years!(state, p, weekly_precip, cfg, nyears[, ws]; callback = nothing)

Repeat [`simulate_year!`](@ref) under one forcing profile. `callback(year, state)`
runs after each year, which is how the bifurcation and extrapolation scripts record
mean-biomass trajectories without keeping every field.
"""
function simulate_years!(s::SimState, p::RietkerkParams, weekly_precip::AbstractVector,
                         cfg::SimConfig, nyears::Integer, ws::SimWorkspace = workspace(s);
                         callback = nothing)
    for year in 1:nyears
        simulate_year!(s, p, weekly_precip, cfg, ws)
        callback === nothing || callback(year, s)
    end
    return s
end

"""
    spectral_bound(p, biomass_max) -> Float64

Largest linear decay rate of the explicit scheme, per field

    O: 8 D_O + α,   W: 8 D_W + r_w + g_max B/k₁,   B: 8 D_P + d

(`8D` is the extreme eigenvalue of the 5-point Laplacian; `B` the largest biomass in
play). Explicit Euler needs `dt ≤ 2 / spectral_bound`. Irrelevant for the default
semi-implicit step, which is stable at any `dt`; it is what makes the explicit one
(`semi_implicit = false`) blow up at the steps the fits use.
"""
function spectral_bound(p::RietkerkParams, biomass_max::Real)
    B = Float64(biomass_max)
    return max(8 * p.surface_water_diffusion_coeff + p.infiltration_rate,
               8 * p.soil_water_diffusion_coeff + p.seepage_rate +
               p.plant_uptake_rate * B / p.uptake_half_saturation,
               8 * p.biomass_diffusion_coeff + p.mortality_rate)
end

"""
    max_stable_dt(p, biomass_max) -> Float64

Largest stable explicit-Euler step, `2 / spectral_bound(p, biomass_max)`, in days.
"""
max_stable_dt(p::RietkerkParams, biomass_max::Real) = 2 / spectral_bound(p, biomass_max)

# ===========================================================================
#  Viable random starts
# ===========================================================================

"""
    keeps_vegetation(p, sites, cfg; min_fraction = 0.1, max_fraction = 10) -> Bool

Whether `p` keeps vegetation at a plausible level over a training rollout. `sites`
is a vector of `(initial_biomass, weekly_precip_per_year)` pairs, simulated from
`O = W = 0` as the loss does. True if at the end every site's mean biomass lies
strictly between `min_fraction` and `max_fraction` times its initial mean — neither
collapsed to bare soil nor exploded. Port of `keeps_vegetation`.
"""
function keeps_vegetation(p::RietkerkParams, sites, cfg::SimConfig;
                          min_fraction::Real = 0.1, max_fraction::Real = 10.0)
    for (initial, weekly_per_year) in sites
        s = simstate(initial)
        ws = workspace(s)
        for weekly in weekly_per_year
            simulate_year!(s, p, weekly, cfg, ws)
        end
        start = Statistics.mean(initial)
        final = Statistics.mean(s.biomass)
        (isfinite(final) && final > min_fraction * start && final < max_fraction * start) ||
            return false
    end
    return true
end

"""
    draw_viable_params(rng, reference, sites, cfg; max_draws = 100, decades = 1,
                       bound_decades = 4, min_fraction = 0.1, max_fraction = 10)
        -> (params, draws)

Random start (log-uniform within `decades` of `reference`) that keeps vegetation
alive, and within a factor 10 of its initial level, over the training rollout.
Port of `draw_viable_model`.

Rietkerk's bare state B = 0 is absorbing: from a start where the plants die, later
years contribute almost nothing to the gradient and the fit can only tune the
die-off. Redrawing restricts the prior to parameter sets compatible with the one
thing the data show for certain, that there is vegetation, at the cost of one
forward rollout per draw. Warns and returns the last draw if none qualifies.
"""
function draw_viable_params(rng::Random.AbstractRNG, reference::RietkerkParams, sites,
                            cfg::AbstractDiscretisation; max_draws::Integer = 100, decades::Real = 1.0,
                            bound_decades::Real = 4.0, min_fraction::Real = 0.1,
                            max_fraction::Real = 10.0)
    p = reference
    for draw in 1:max_draws
        p = randparams(rng, reference; decades = decades, bound_decades = bound_decades)
        keeps_vegetation(p, sites, cfg; min_fraction = min_fraction,
                         max_fraction = max_fraction) && return (p, draw)
    end
    @warn "no random start kept vegetation at a plausible level in $max_draws draws; using the last one"
    return (p, max_draws)
end
