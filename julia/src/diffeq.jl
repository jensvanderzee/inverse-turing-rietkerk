"""
    ODEConfig(alg; year_length = 364.0, kwargs...)

Integrate the *continuous* PDE — the method-of-lines ODE below — with the
DifferentialEquations.jl algorithm `alg` and solver options `kwargs` (`abstol`,
`reltol`, ...), instead of the fixed-step scheme of the Python code:

    dO/dt = D_O ΔO + R(t) − α O I(B)
    dW/dt = D_W ΔW + α O I(B) − (g_max B/(W + k₁) + r_w) W
    dB/dt = D_P ΔB + c g_max W B/(W + k₁) − d B

with the same 5-point Laplacian, so the fixed-step scheme is a discretisation of
exactly this system and converges to it as `steps_per_week → ∞`.

An [`InverseProblem`](@ref) built with an `ODEConfig` works with [`loss`](@ref),
[`evaluate`](@ref), [`ForwardDiffBackend`](@ref) (dual numbers through the solver)
and `AdjointODEBackend` (SciMLSensitivity adjoints). Requires an OrdinaryDiffEq
solver package to be loaded (e.g. `using OrdinaryDiffEqStabilizedRK`), which
activates the `InverseTuringDiffEqExt` extension.

Rain is piecewise constant per week, so each week is integrated as its own
segment: an adaptive solver must never step across a jump in the forcing, and a
time-dependent forcing with `tstops` still evaluates the next week's rate in the
last stages of a step that ends on a boundary.

Biomass is in data units (up to 1500), so set `abstol` to the scale of the data
(e.g. `1e-3`–`1e-2`) and control accuracy with `reltol`; the SciML default
`abstol = 1e-6` asks for ten significant digits.
"""
struct ODEConfig{A,K} <: AbstractDiscretisation
    alg::A
    year_length::Float64
    solver_kwargs::K
end

ODEConfig(alg; year_length::Real = DAYS_PER_YEAR, kwargs...) =
    ODEConfig(alg, Float64(year_length), (; kwargs...))

describe(cfg::ODEConfig) = "ODE " * string(nameof(typeof(cfg.alg)))

"""
    rietkerk_rhs!(du, u, p, t)

Right-hand side of the method-of-lines ODE in DifferentialEquations.jl form. `u`
and `du` are `H×W×3` arrays with third axis `(O, W, B)`; `p` is the parameter
vector: the eleven natural parameters in [`PARAM_NAMES`](@ref) order followed by
the rain rate `R` (mm/day), constant over a forcing week. Allocation-free, and
differentiable by Enzyme (which is how SciMLSensitivity's `EnzymeVJP` builds the
adjoint).

The nonlinear terms clamp `B` and (in uptake and growth) `W` at zero exactly as the
fixed-step scheme does, so that scheme converges to this system.
"""
function rietkerk_rhs!(du, u, p, t)
    O = view(u, :, :, 1)
    W = view(u, :, :, 2)
    B = view(u, :, :, 3)
    dO = view(du, :, :, 1)
    dW = view(du, :, :, 2)
    dB = view(du, :, :, 3)
    DO, DW, DP = p[1], p[2], p[3]
    rw, d, α, gmax, c = p[4], p[5], p[6], p[7], p[8]
    k2, w0, k1 = p[9], p[10], p[11]
    R = p[NPARAMS + 1]
    k2w0 = k2 * w0
    laplacian!(dO, O)
    laplacian!(dW, W)
    laplacian!(dB, B)
    @. dO = DO * dO + R - (α * (max(B, 0) + k2w0) / (max(B, 0) + k2)) * O
    @. dW = DW * dW + (α * (max(B, 0) + k2w0) / (max(B, 0) + k2)) * O -
            (gmax * max(B, 0) / (max(W, 0) + k1) + rw) * W
    @. dB = DP * dB + c * gmax * max(W, 0) * max(B, 0) / (max(W, 0) + k1) - d * B
    return nothing
end

"""
    ode_parameters(p, R) -> Vector

Parameter vector of [`rietkerk_rhs!`](@ref): the natural parameters then the rain rate.
"""
ode_parameters(p::RietkerkParams, R) = [paramvector(p); R]

"""
    ode_week_length(cfg, nweeks) -> days

Length of one forcing week of the continuous model.
"""
ode_week_length(cfg::ODEConfig, nweeks::Integer) = cfg.year_length / nweeks

"""
    ode_rate(weekly_rate, cfg, nweeks) -> mm/day

Rain rate that delivers `7·weekly_rate` mm over one week of the continuous model —
the same annual water as the fixed-step scheme, whatever `year_length` is.
"""
ode_rate(weekly_rate::Real, cfg::ODEConfig, nweeks::Integer) = weekly_rate * 7 * nweeks / cfg.year_length

"""
    pack_state(O, W, B) / pack_state(s::SimState) / pack_state(biomass; T)

Stack the three fields into the `H×W×3` layout of the ODE backend; from a biomass
field alone, both water fields start at zero.
"""
function pack_state(O::AbstractMatrix, W::AbstractMatrix, B::AbstractMatrix)
    size(O) == size(W) == size(B) ||
        throw(DimensionMismatch("all three fields must have the same size"))
    u = similar(B, size(B)..., 3)
    u[:, :, 1] .= O
    u[:, :, 2] .= W
    u[:, :, 3] .= B
    return u
end
pack_state(s::SimState) = pack_state(s.surface_water, s.soil_water, s.biomass)
function pack_state(biomass::AbstractMatrix; T::Type = eltype(biomass))
    b = convert.(T, biomass)
    return pack_state(zero(b), zero(b), b)
end

"""Views onto the three fields of a packed state (no copy)."""
unpack_state(u::AbstractArray{<:Any,3}) = (view(u, :, :, 1), view(u, :, :, 2), view(u, :, :, 3))

"""The biomass slice of a packed state."""
biomass_of(u::AbstractArray{<:Any,3}) = view(u, :, :, 3)

"""
    reaction_rate_bound(p, biomass_max) -> Float64

Bound on the spectral radius of the ODE's Jacobian (`8D` from diffusion plus the
reaction rates at biomass up to `biomass_max`), used as the eigenvalue estimate of
the stabilised `ROCK` methods instead of their power iteration.
"""
reaction_rate_bound(p::RietkerkParams, biomass_max::Real) = spectral_bound(p, biomass_max)

# ---------------------------------------------------------------------------
#  Implemented by the extensions
# ---------------------------------------------------------------------------

"""
    solve_week!(u, p, R, cfg::ODEConfig[, integrator]) -> u

Advance the packed state `u` through one forcing week of length
`cfg.year_length / 52` at rain rate `R`. Requires an OrdinaryDiffEq solver package.
"""
function solve_week! end

"""
    simulate_years_ode(p, biomass0, weekly_precip, cfg::ODEConfig, nyears)
        -> (final_u, mean_biomass_per_year)

Roll the continuous model forward `nyears` years under one forcing profile.
Requires an OrdinaryDiffEq solver package.
"""
function simulate_years_ode end

"""
    AdjointODEBackend(; sensealg = nothing, adjoint_alg = nothing)

Gradient of an [`ODEConfig`](@ref) problem by SciMLSensitivity's continuous
adjoint, with Enzyme computing the vector–Jacobian products of
[`rietkerk_rhs!`](@ref). `sensealg` defaults to
`InterpolatingAdjoint(autojacvec = EnzymeVJP())`; `adjoint_alg` (the solver of
the adjoint ODE) defaults to the forward algorithm. Weeks are composed as in the
Enzyme backend: forward pass with weekly checkpoints, then one adjoint solve per
week in reverse. Requires `using SciMLSensitivity` and an OrdinaryDiffEq solver
package (extension `InverseTuringSciMLSensitivityExt`).
"""
struct AdjointODEBackend{S,A} <: AbstractGradientBackend
    sensealg::S
    adjoint_alg::A
end
AdjointODEBackend(; sensealg = nothing, adjoint_alg = nothing) = AdjointODEBackend(sensealg, adjoint_alg)

# Hooks the extensions add methods to; the core only forwards to them.
function ode_trajectory_loss end
function ode_gradient_cache end

_diffeq_loaded() = !isempty(methods(ode_trajectory_loss))
_require_diffeq() = _diffeq_loaded() ||
    throw(ArgumentError("ODEConfig needs an OrdinaryDiffEq solver package, e.g. `using OrdinaryDiffEqStabilizedRK`"))

trajectory_loss(tr::SiteTrajectory, p::RietkerkParams, cfg::ODEConfig, ::Type{T};
                delta::Bool = true) where {T} =
    (_require_diffeq(); ode_trajectory_loss(tr, p, cfg, T, delta))

function gradient_cache(prob::InverseProblem, b::AdjointODEBackend)
    prob.cfg isa ODEConfig ||
        throw(ArgumentError("AdjointODEBackend differentiates the continuous model; build the problem with an ODEConfig"))
    isempty(methods(ode_gradient_cache)) &&
        throw(ArgumentError("AdjointODEBackend needs `using SciMLSensitivity` and an OrdinaryDiffEq solver package"))
    return ode_gradient_cache(prob, b)
end

gradient_cache(::InverseProblem{<:Any,<:Any,<:ODEConfig}, ::EnzymeBackend) =
    throw(ArgumentError("EnzymeBackend differentiates the fixed-step scheme; for an ODEConfig problem use AdjointODEBackend or ForwardDiffBackend"))
