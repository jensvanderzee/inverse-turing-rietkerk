"""
    AbstractGradientBackend

How [`loss_and_gradient`](@ref) differentiates the loss. All backends return the
gradient with respect to the natural parameters θ (the optimiser, which works on
log θ, multiplies by θ itself).

| backend                  | differentiates                           | cost per gradient          |
|:-------------------------|:-----------------------------------------|:---------------------------|
| [`EnzymeBackend`](@ref)  | the fixed-step scheme, reverse mode      | ~5 rollouts, O(weeks) memory |
| [`ForwardDiffBackend`](@ref) | the fixed-step scheme, forward mode  | ~(1 + 11/chunk) rollouts of dual numbers |
| [`FiniteDiffBackend`](@ref) | the fixed-step scheme, central differences | 22 rollouts, for checks only |
| `AdjointODEBackend`      | the continuous PDE (DifferentialEquations.jl + SciMLSensitivity), see `ext/` | solver dependent |

The first three give the gradient of exactly the loss PyTorch differentiates.
"""
abstract type AbstractGradientBackend end

"""
    EnzymeBackend()

Reverse-mode AD with Enzyme through the fixed-step scheme — a discrete adjoint, as
PyTorch's autograd computes it. The forward pass stores the state at every week
boundary; the reverse pass restores each week, recomputes the states at its step
boundaries, and lets Enzyme differentiate one step at a time — the recomputation
schedule of PyTorch's `gradient_checkpointing=True`. It stores three fields per
week, ~200 MB for a 131×140 site over nine years. The diffusion solve (FFTW/BLAS)
has a hand-written rule (`src/enzyme_rules.jl`); everything else is Enzyme's.

Enzyme differentiates single steps rather than whole weeks on purpose: with
Enzyme 0.13 on Julia 1.12, differentiating a runtime-length loop around the step
function corrupted the heap after a few thousand calls (the collector then
segfaults), while single steps have run over a million calls without a fault.
"""
struct EnzymeBackend <: AbstractGradientBackend end

"""
    ForwardDiffBackend(; chunk = 11)

Forward-mode AD (ForwardDiff) through the fixed-step scheme. Every field carries
`chunk` partials; the diffusion solve transforms value and partials in one batched
FFTW call. Allocation-light and simple, but its cost grows with the number of
parameters, where reverse mode's does not.
"""
struct ForwardDiffBackend <: AbstractGradientBackend
    chunk::Int
end
ForwardDiffBackend(; chunk::Integer = NPARAMS) = ForwardDiffBackend(chunk)

"""
    FiniteDiffBackend(; relstep = 1e-6)

Central differences in log θ (step `relstep` relative to each parameter). Only for
checking the other backends: 22 rollouts per gradient and O(relstep²) error.
"""
struct FiniteDiffBackend <: AbstractGradientBackend
    relstep::Float64
end
FiniteDiffBackend(; relstep::Real = 1e-6) = FiniteDiffBackend(relstep)

"""
    gradient_cache(prob, backend = EnzymeBackend()) -> cache

Preallocate everything a backend needs for repeated gradients of `prob` (states,
checkpoints, FFTW plans, dual-number buffers). Training builds one per run.
"""
function gradient_cache end

"""
    loss_and_gradient!(g, cache, θ) -> loss

Loss at the natural parameters `θ` (vector or [`RietkerkParams`](@ref)), with its
gradient written to `g`.
"""
function loss_and_gradient! end

"""
    loss_and_gradient(prob, θ; backend = EnzymeBackend()) -> (loss, gradient)

One-off loss and gradient with respect to the natural parameters. Build a
[`gradient_cache`](@ref) for repeated calls.
"""
function loss_and_gradient(prob::InverseProblem, θ; backend::AbstractGradientBackend = EnzymeBackend())
    g = zeros(NPARAMS)
    L = loss_and_gradient!(g, gradient_cache(prob, backend), θ)
    return L, g
end

_paramvector(θ::RietkerkParams) = paramvector(convert(RietkerkParams{Float64}, θ))
_paramvector(θ::AbstractVector) = collect(Float64, θ)

# ===========================================================================
#  Enzyme: discrete adjoint with weekly checkpoints
# ===========================================================================

struct EnzymeTrajectoryCache{S}
    state::SimState{Float64,Matrix{Float64}}
    shadow::SimState{Float64,Matrix{Float64}}
    tmp::Matrix{Float64}
    dtmp::Matrix{Float64}
    solver::S
    checkpoints::Vector{SimState{Float64,Matrix{Float64}}}   # every week boundary
    steps::Vector{SimState{Float64,Matrix{Float64}}}         # step boundaries of one week
    biomass::Vector{Matrix{Float64}}            # B_0, B_1, …, B_K
end

_zerostate(H, W) = SimState(zeros(H, W), zeros(H, W), zeros(H, W))

function EnzymeTrajectoryCache(tr::SiteTrajectory, steps_per_week::Integer)
    H, W = size(tr)
    nweeks = sum(length, tr.forcings; init = 0)
    EnzymeTrajectoryCache(_zerostate(H, W), _zerostate(H, W), zeros(H, W), zeros(H, W),
                          DiffusionSolver{Float64}(H, W),
                          [_zerostate(H, W) for _ in 1:nweeks],
                          [_zerostate(H, W) for _ in 1:steps_per_week],
                          [zeros(H, W) for _ in 0:ntransitions(tr)])
end

struct EnzymeCache{P<:InverseProblem,C<:EnzymeTrajectoryCache}
    prob::P
    trajectories::Vector{C}
    losses::Vector{Float64}
    grads::Vector{Vector{Float64}}
    compiled::Base.RefValue{Bool}
end

function gradient_cache(prob::InverseProblem, ::EnzymeBackend)
    caches = [EnzymeTrajectoryCache(tr, prob.cfg.steps_per_week) for tr in prob.trajectories]
    n = length(caches)
    return EnzymeCache(prob, caches, zeros(n), [zeros(NPARAMS) for _ in 1:n], Ref(false))
end

"""
Vector–Jacobian product of one step: on entry `c.shadow` holds the adjoint of the
state after the step and `c.state` the state before it; on exit `c.shadow` holds
the adjoint before the step. Returns the parameter adjoint.
"""
function step_vjp!(c::EnzymeTrajectoryCache, p::RietkerkParams{Float64}, rate::Float64,
                   dt::Float64, semi_implicit::Bool)
    fill!(c.dtmp, 0.0)
    if semi_implicit
        c.solver isa DiffusionSolver && (c.solver.nsaved[] = 0)
        res = Enzyme.autodiff(Enzyme.Reverse, semi_implicit_step!, Enzyme.Const,
                              Enzyme.Duplicated(c.state, c.shadow), Enzyme.Active(p),
                              Enzyme.Const(rate), Enzyme.Const(dt),
                              Enzyme.Duplicated(c.tmp, c.dtmp), Enzyme.Const(c.solver))
    else
        res = Enzyme.autodiff(Enzyme.Reverse, explicit_step!, Enzyme.Const,
                              Enzyme.Duplicated(c.state, c.shadow), Enzyme.Active(p),
                              Enzyme.Const(rate), Enzyme.Const(dt),
                              Enzyme.Duplicated(c.tmp, c.dtmp))
    end
    return res[1][2]
end

function _fill_shadow_seed!(c::EnzymeTrajectoryCache, tr::SiteTrajectory, k::Int, delta::Bool)
    # ∂L/∂B_k of L = Σ_k mean(e_k²), on top of what later years already sent back.
    B = c.biomass
    N = length(B[1])
    s = c.shadow.biomass
    K = ntransitions(tr)
    if delta
        # e_k = (B_k − B_{k−1}) − (T_k − T_{k−1}) enters e_k² and e_{k+1}².
        Tk = tr.targets[k]
        Tkm = k == 1 ? tr.initial_target : tr.targets[k - 1]
        @inbounds for i in eachindex(s)
            ek = (B[k + 1][i] - B[k][i]) - (Tk[i] - Tkm[i])
            s[i] += 2 * ek / N
        end
        if k < K
            Tn = tr.targets[k + 1]
            @inbounds for i in eachindex(s)
                en = (B[k + 2][i] - B[k + 1][i]) - (Tn[i] - Tk[i])
                s[i] -= 2 * en / N
            end
        end
    else
        Tk = tr.targets[k]
        @inbounds for i in eachindex(s)
            s[i] += 2 * (B[k + 1][i] - Tk[i]) / N
        end
    end
    return nothing
end

"""
Loss and natural-parameter gradient of one site's summed per-transition MSE: a
forward pass that checkpoints every week boundary, then a reverse sweep that
recomputes each week's step boundaries from its checkpoint and runs one Enzyme VJP
per step, seeding the biomass adjoint with ∂L/∂B_k at every year end.
"""
function trajectory_loss_and_gradient!(g::Vector{Float64}, c::EnzymeTrajectoryCache,
                                       tr::SiteTrajectory, p::RietkerkParams{Float64},
                                       cfg::SimConfig, delta::Bool)
    s = c.state
    fill!(s.surface_water, 0.0)
    fill!(s.soil_water, 0.0)
    copyto!(s.biomass, tr.initial_biomass)
    copyto!(c.biomass[1], tr.initial_biomass)
    spw = cfg.steps_per_week
    semi = cfg.semi_implicit

    i = 0
    for (k, weekly) in enumerate(tr.forcings)
        nweeks = length(weekly)
        dt = step_size(cfg, nweeks)
        for w in 1:nweeks
            i += 1
            copystate!(c.checkpoints[i], s)
            simulate_week!(s, p, week_rate(weekly[w], cfg, nweeks), dt, spw, c.tmp, c.solver, semi)
        end
        copyto!(c.biomass[k + 1], s.biomass)
    end

    L = 0.0
    for k in eachindex(tr.targets)
        L += delta ?
             mean_squared_delta_error(c.biomass[k + 1], c.biomass[k], tr.targets[k],
                                      k == 1 ? tr.initial_target : tr.targets[k - 1]) :
             mean_squared_error(c.biomass[k + 1], tr.targets[k])
    end

    fill!(g, 0.0)
    fill!(c.shadow.surface_water, 0.0)
    fill!(c.shadow.soil_water, 0.0)
    fill!(c.shadow.biomass, 0.0)
    for k in reverse(eachindex(tr.forcings))
        _fill_shadow_seed!(c, tr, k, delta)
        weekly = tr.forcings[k]
        nweeks = length(weekly)
        dt = step_size(cfg, nweeks)
        for w in nweeks:-1:1
            rate = week_rate(weekly[w], cfg, nweeks)
            # Recompute the states at the week's step boundaries from its checkpoint.
            copystate!(c.steps[1], c.checkpoints[i])
            for j in 1:(spw - 1)
                copystate!(c.steps[j + 1], c.steps[j])
                simulate_week!(c.steps[j + 1], p, rate, dt, 1, c.tmp, c.solver, semi)
            end
            for j in spw:-1:1
                copystate!(s, c.steps[j])
                dp = step_vjp!(c, p, rate, dt, semi)
                for q in 1:NPARAMS
                    g[q] += getfield(dp, q)
                end
            end
            i -= 1
        end
    end
    return L
end

function loss_and_gradient!(g::AbstractVector, cache::EnzymeCache, θ)
    prob = cache.prob
    p = RietkerkParams(_paramvector(θ))
    trs = prob.trajectories
    run!(i) = (cache.losses[i] = trajectory_loss_and_gradient!(cache.grads[i], cache.trajectories[i],
                                                               trs[i], p, prob.cfg, prob.delta_loss))
    if prob.threaded && length(trs) > 1
        # Compile the derivative on one thread before several ask for it at once.
        first = 1
        if !cache.compiled[]
            run!(1)
            cache.compiled[] = true
            first = 2
        end
        Threads.@threads for i in first:length(trs)
            run!(i)
        end
    else
        foreach(run!, eachindex(trs))
        cache.compiled[] = true
    end
    scale = loss_scale(prob)
    fill!(g, 0.0)
    for gi in cache.grads          # fixed order: thread-count independent
        g .+= gi
    end
    g .*= scale
    return sum(cache.losses) * scale
end

# ===========================================================================
#  ForwardDiff
# ===========================================================================

struct ForwardDiffCache{P<:InverseProblem,F,R,C}
    prob::P
    objective::F
    result::R
    config::C
end

function gradient_cache(prob::InverseProblem, b::ForwardDiffBackend)
    objective = let prob = prob
        θ -> loss(θ, prob)
    end
    x = paramvector(SYNTHETIC_TRUTH)
    return ForwardDiffCache(prob, objective, DiffResults.GradientResult(x),
                            ForwardDiff.GradientConfig(objective, x, ForwardDiff.Chunk{b.chunk}()))
end

function loss_and_gradient!(g::AbstractVector, cache::ForwardDiffCache, θ)
    res = ForwardDiff.gradient!(cache.result, cache.objective, _paramvector(θ), cache.config)
    copyto!(g, DiffResults.gradient(res))
    return DiffResults.value(res)
end

# ===========================================================================
#  Finite differences (checks only)
# ===========================================================================

struct FiniteDiffCache{P<:InverseProblem}
    prob::P
    relstep::Float64
end

gradient_cache(prob::InverseProblem, b::FiniteDiffBackend) = FiniteDiffCache(prob, b.relstep)

function loss_and_gradient!(g::AbstractVector, cache::FiniteDiffCache, θ)
    x = _paramvector(θ)
    L = loss(x, cache.prob)
    h = cache.relstep
    for i in 1:NPARAMS
        up = copy(x)
        dn = copy(x)
        up[i] *= exp(h)
        dn[i] *= exp(-h)
        # central difference in log θ, converted to ∂L/∂θ
        g[i] = (loss(up, cache.prob) - loss(dn, cache.prob)) / (2h * x[i])
    end
    return L
end
