"""
    InverseTuringDiffEqExt

The continuous-model backend: [`ODEConfig`](@ref) problems integrated with any
OrdinaryDiffEq algorithm. Activated by loading a solver package
(`using OrdinaryDiffEqStabilizedRK`, `using OrdinaryDiffEqTsit5`, ...).

Each forcing week is one ODE segment with constant rain. One integrator is built
per rollout and `reinit!`-ed for every week, so the solver caches (several copies
of a 55 000-unknown state for a 131×140 site) are allocated once, not 468 times.
"""
module InverseTuringDiffEqExt

using InverseTuring
using InverseTuring: RietkerkParams, ODEConfig, SiteTrajectory, NPARAMS, rietkerk_rhs!,
                     pack_state, biomass_of, mean_squared_delta_error, mean_squared_error,
                     ode_parameters, ode_week_length, ode_rate
import InverseTuring: solve_week!, simulate_years_ode, ode_trajectory_loss
import SciMLBase
import OrdinaryDiffEqCore
import Statistics

"""
    week_integrator(u, p, cfg; nweeks = 52)

Integrator over one week for the packed state `u` (whose element type may be a
`ForwardDiff.Dual`), keeping only the final state.
"""
function week_integrator(u::AbstractArray{T,3}, p::RietkerkParams, cfg::ODEConfig;
                         nweeks::Integer = 52) where {T}
    pvec = convert(Vector{promote_type(T, eltype(p))}, ode_parameters(p, zero(T)))
    prob = SciMLBase.ODEProblem{true}(rietkerk_rhs!, copy(u), (0.0, ode_week_length(cfg, nweeks)), pvec)
    return SciMLBase.init(prob, cfg.alg; save_everystep = false, save_start = false,
                          save_end = false, cfg.solver_kwargs...)
end

function _advance!(integ, u::AbstractArray, R, tf::Float64)
    integ.p[end] = R
    SciMLBase.reinit!(integ, u; t0 = 0.0, tf = tf, erase_sol = true)
    SciMLBase.solve!(integ)
    SciMLBase.successful_retcode(integ.sol) ||
        throw(ErrorException("ODE solve failed with retcode $(integ.sol.retcode)"))
    copyto!(u, integ.u)
    return u
end

"""
    solve_week!(u, p, R, cfg::ODEConfig[, integrator]) -> u

Advance the packed state `u` by one forcing week at rain rate `R` (mm/day as the
ODE sees it, see `ode_rate`).
"""
function solve_week!(u::AbstractArray{<:Any,3}, p::RietkerkParams, R, cfg::ODEConfig,
                     integ = week_integrator(u, p, cfg); nweeks::Integer = 52)
    return _advance!(integ, u, R, ode_week_length(cfg, nweeks))
end

"""Advance `u` by one year of weekly forcing with the integrator `integ`."""
function ode_year!(u, integ, weekly::AbstractVector, cfg::ODEConfig)
    nweeks = length(weekly)
    tf = ode_week_length(cfg, nweeks)
    for w in 1:nweeks
        _advance!(integ, u, ode_rate(weekly[w], cfg, nweeks), tf)
    end
    return u
end

function simulate_years_ode(p::RietkerkParams, biomass0::AbstractMatrix,
                            weekly_precip::AbstractVector, cfg::ODEConfig, nyears::Integer;
                            callback = nothing)
    u = pack_state(biomass0)
    integ = week_integrator(u, p, cfg; nweeks = length(weekly_precip))
    means = Float64[Statistics.mean(biomass_of(u))]
    for year in 1:nyears
        ode_year!(u, integ, weekly_precip, cfg)
        push!(means, Statistics.mean(biomass_of(u)))
        callback === nothing || callback(year, u)
    end
    return (u, means)
end

function ode_trajectory_loss(tr::SiteTrajectory, p::RietkerkParams, cfg::ODEConfig,
                             ::Type{T}, delta::Bool) where {T}
    u = pack_state(tr.initial_biomass; T = T)
    integ = week_integrator(u, p, cfg; nweeks = length(first(tr.forcings)))
    prev_pred = copy(biomass_of(u))
    prev_target = tr.initial_target
    acc = zero(T)
    for k in eachindex(tr.targets)
        ode_year!(u, integ, tr.forcings[k], cfg)
        target = tr.targets[k]
        B = biomass_of(u)
        acc += delta ? mean_squared_delta_error(B, prev_pred, target, prev_target) :
                       mean_squared_error(B, target)
        copyto!(prev_pred, B)
        prev_target = target
    end
    return acc
end

end # module
