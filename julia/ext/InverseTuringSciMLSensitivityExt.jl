"""
    InverseTuringSciMLSensitivityExt

`AdjointODEBackend`: gradients of [`ODEConfig`](@ref) problems by SciMLSensitivity's
continuous adjoint, with Enzyme computing the vector–Jacobian products of
`rietkerk_rhs!`. Activated by `using SciMLSensitivity` together with an
OrdinaryDiffEq solver package.

The weeks are composed as in the Enzyme backend of the fixed-step scheme: a forward
pass stores the state at every week boundary; the reverse pass re-solves each week
densely from its checkpoint and runs one adjoint solve over it, seeded with the
adjoint of the week's end state and returning that of its start state plus the
parameter gradient. Storing every week's dense solution instead would need tens of
gigabytes; per-week segments are also what the piecewise-constant forcing needs.
"""
module InverseTuringSciMLSensitivityExt

using InverseTuring
using InverseTuring: RietkerkParams, ODEConfig, SiteTrajectory, AdjointODEBackend, NPARAMS,
                     rietkerk_rhs!, pack_state, biomass_of, ode_parameters, ode_week_length,
                     ode_rate, loss_scale, ntransitions, mean_squared_delta_error,
                     mean_squared_error, _paramvector
import InverseTuring: ode_gradient_cache, loss_and_gradient!
import SciMLSensitivity
import SciMLBase
import OrdinaryDiffEqCore

struct AdjointTrajectoryCache
    u::Array{Float64,3}
    λ::Array{Float64,3}
    checkpoints::Vector{Array{Float64,3}}
    biomass::Vector{Matrix{Float64}}            # B_0, B_1, …, B_K
end

function AdjointTrajectoryCache(tr::SiteTrajectory)
    H, W = size(tr)
    nweeks = sum(length, tr.forcings; init = 0)
    AdjointTrajectoryCache(zeros(H, W, 3), zeros(H, W, 3), [zeros(H, W, 3) for _ in 1:nweeks],
                           [zeros(H, W) for _ in 0:ntransitions(tr)])
end

struct AdjointODECache{P,S,A}
    prob::P
    sensealg::S
    adjoint_alg::A
    trajectories::Vector{AdjointTrajectoryCache}
    losses::Vector{Float64}
    grads::Vector{Vector{Float64}}
end

function ode_gradient_cache(prob::InverseProblem, b::AdjointODEBackend)
    sensealg = b.sensealg === nothing ?
               SciMLSensitivity.InterpolatingAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP()) :
               b.sensealg
    adjoint_alg = b.adjoint_alg === nothing ? prob.cfg.alg : b.adjoint_alg
    n = length(prob.trajectories)
    return AdjointODECache(prob, sensealg, adjoint_alg,
                           [AdjointTrajectoryCache(tr) for tr in prob.trajectories],
                           zeros(n), [zeros(NPARAMS) for _ in 1:n])
end

function _week_problem(cfg, u0, pvec, tf)
    f = cfg.sparse_jacobian ?
        SciMLBase.ODEFunction{true}(rietkerk_rhs!; jac_prototype = InverseTuring.rhs_sparsity(size(u0, 1), size(u0, 2))) :
        SciMLBase.ODEFunction{true}(rietkerk_rhs!)
    return SciMLBase.ODEProblem{true}(f, u0, (0.0, tf), pvec)
end

function _advance!(integ, u, R, tf)
    integ.p[end] = R
    SciMLBase.reinit!(integ, u; t0 = 0.0, tf = tf, erase_sol = true)
    SciMLBase.solve!(integ)
    SciMLBase.successful_retcode(integ.sol) ||
        throw(ErrorException("ODE solve failed with retcode $(integ.sol.retcode)"))
    copyto!(u, integ.u)
end

function _add_seed!(λ::Array{Float64,3}, c::AdjointTrajectoryCache, tr::SiteTrajectory,
                    k::Int, delta::Bool)
    s = view(λ, :, :, 3)
    B = c.biomass
    N = length(B[1])
    K = ntransitions(tr)
    Tk = tr.targets[k]
    if delta
        Tkm = k == 1 ? tr.initial_target : tr.targets[k - 1]
        @inbounds for i in eachindex(s)
            s[i] += 2 * ((B[k + 1][i] - B[k][i]) - (Tk[i] - Tkm[i])) / N
        end
        if k < K
            Tn = tr.targets[k + 1]
            @inbounds for i in eachindex(s)
                s[i] -= 2 * ((B[k + 2][i] - B[k + 1][i]) - (Tn[i] - Tk[i])) / N
            end
        end
    else
        @inbounds for i in eachindex(s)
            s[i] += 2 * (B[k + 1][i] - Tk[i]) / N
        end
    end
end

function trajectory_loss_and_gradient!(g::Vector{Float64}, c::AdjointTrajectoryCache,
                                       tr::SiteTrajectory, p::RietkerkParams{Float64},
                                       cfg::ODEConfig, sensealg, adjoint_alg, delta::Bool)
    u = c.u
    fill!(u, 0.0)
    view(u, :, :, 3) .= tr.initial_biomass
    copyto!(c.biomass[1], tr.initial_biomass)
    pvec = ode_parameters(p, 0.0)
    nweeks1 = length(first(tr.forcings))
    integ = SciMLBase.init(_week_problem(cfg, copy(u), copy(pvec), ode_week_length(cfg, nweeks1)),
                           cfg.alg; save_everystep = false, save_start = false, save_end = false,
                           cfg.solver_kwargs...)
    i = 0
    for (k, weekly) in enumerate(tr.forcings)
        nweeks = length(weekly)
        tf = ode_week_length(cfg, nweeks)
        for w in 1:nweeks
            i += 1
            copyto!(c.checkpoints[i], u)
            _advance!(integ, u, ode_rate(weekly[w], cfg, nweeks), tf)
        end
        copyto!(c.biomass[k + 1], biomass_of(u))
    end

    L = 0.0
    for k in eachindex(tr.targets)
        L += delta ?
             mean_squared_delta_error(c.biomass[k + 1], c.biomass[k], tr.targets[k],
                                      k == 1 ? tr.initial_target : tr.targets[k - 1]) :
             mean_squared_error(c.biomass[k + 1], tr.targets[k])
    end

    fill!(g, 0.0)
    λ = c.λ
    fill!(λ, 0.0)
    seed = let λ = λ
        (out, u, p, t, i) -> copyto!(out, λ)
    end
    for k in reverse(eachindex(tr.forcings))
        _add_seed!(λ, c, tr, k, delta)
        weekly = tr.forcings[k]
        nweeks = length(weekly)
        tf = ode_week_length(cfg, nweeks)
        for w in nweeks:-1:1
            pvec[end] = ode_rate(weekly[w], cfg, nweeks)
            sol = SciMLBase.solve(_week_problem(cfg, c.checkpoints[i], copy(pvec), tf), cfg.alg;
                                  save_everystep = true, dense = true, cfg.solver_kwargs...)
            SciMLBase.successful_retcode(sol) ||
                throw(ErrorException("ODE solve failed with retcode $(sol.retcode)"))
            du0, dp = SciMLSensitivity.adjoint_sensitivities(sol, adjoint_alg; sensealg = sensealg,
                                                            t = [tf], dgdu_discrete = seed,
                                                            cfg.solver_kwargs...)
            copyto!(λ, du0)
            for j in 1:NPARAMS
                g[j] += dp[j]
            end
            i -= 1
        end
    end
    return L
end

function loss_and_gradient!(g::AbstractVector, cache::AdjointODECache, θ)
    prob = cache.prob
    p = RietkerkParams(_paramvector(θ))
    trs = prob.trajectories
    run!(i) = (cache.losses[i] = trajectory_loss_and_gradient!(cache.grads[i], cache.trajectories[i],
                                                               trs[i], p, prob.cfg, cache.sensealg,
                                                               cache.adjoint_alg, prob.delta_loss))
    if prob.threaded && length(trs) > 1
        run!(1)
        Threads.@threads for i in 2:length(trs)
            run!(i)
        end
    else
        foreach(run!, eachindex(trs))
    end
    scale = loss_scale(prob)
    fill!(g, 0.0)
    for gi in cache.grads
        g .+= gi
    end
    g .*= scale
    return sum(cache.losses) * scale
end

end # module
