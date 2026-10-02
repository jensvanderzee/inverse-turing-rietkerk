"""
    SiteTrajectory{T}

One site's contribution to the inverse problem:

- `initial_biomass`: field the rollout starts from (with `O = W = 0`).
- `initial_target`: observed field the first predicted change is measured against.
  The real data use the first image for both; the synthetic experiments start
  from the clean equilibrium and compare against it too.
- `forcings[k]`: weekly rainfall rates (mm/day) driving transition `k`.
- `targets[k]`: observed biomass at the end of transition `k`.
"""
struct SiteTrajectory{T,F<:AbstractVector}
    initial_biomass::Matrix{T}
    initial_target::Matrix{T}
    forcings::Vector{F}
    targets::Vector{Matrix{T}}

    function SiteTrajectory(initial_biomass::Matrix{T}, initial_target::Matrix{T},
                            forcings::Vector{F}, targets::Vector{Matrix{T}}) where {T,F<:AbstractVector}
        length(forcings) == length(targets) ||
            throw(DimensionMismatch("need one forcing per target, got $(length(forcings)) and $(length(targets))"))
        size(initial_target) == size(initial_biomass) ||
            throw(DimensionMismatch("initial target and initial biomass differ in size"))
        for t in targets
            size(t) == size(initial_biomass) ||
                throw(DimensionMismatch("target size $(size(t)) != initial biomass size $(size(initial_biomass))"))
        end
        new{T,F}(initial_biomass, initial_target, forcings, targets)
    end
end

ntransitions(tr::SiteTrajectory) = length(tr.targets)
Base.size(tr::SiteTrajectory) = size(tr.initial_biomass)

"""
    SiteTrajectory(series::SiteSeries)

Consecutive observations of a real site: transition `k` runs from year `k` to year
`k+1` under year `k`'s weekly rainfall, as in `realdata_train_invPDE.compute_loss`.
"""
function SiteTrajectory(series::SiteSeries{T}) where {T}
    obs = series.observations
    length(obs) >= 2 ||
        throw(ArgumentError("site $(series.name) has $(length(obs)) observations; need at least 2"))
    forcings = [o.weekly_precipitation for o in obs[1:(end - 1)]]
    targets = [copy(o.biomass) for o in obs[2:end]]
    return SiteTrajectory(copy(obs[1].biomass), copy(obs[1].biomass), forcings, targets)
end

"""
    viability_sites(trajectories) -> Vector

The `(initial_biomass, weekly_precip_per_year)` pairs that
[`draw_viable_params`](@ref) screens random starts on: exactly the rollout the loss
performs.
"""
viability_sites(trajectories::AbstractVector{<:SiteTrajectory}) =
    [(tr.initial_biomass, tr.forcings) for tr in trajectories]

"""
    InverseProblem(trajectories, cfg; average = true, delta_loss = true,
                   threaded = Threads.nthreads() > 1)

Everything a fit needs: observed trajectories, time discretisation, and how the
per-transition MSEs combine.

- `average = true`: divide by the number of transitions (`realdata_train_invPDE.py`);
  `false`: sum (`train_invPDE_synthetic_batch*.py`). This rescales the gradient,
  which Adam mostly normalises away but gradient clipping does not.
- `delta_loss = true`: score the year-on-year change in biomass (`use_delta_loss`,
  used by every fit); `false`: score the absolute field.
- `threaded`: evaluate sites in parallel. Sites are the only parallelism a single
  loss offers; partial sums are reduced in a fixed order, so the result does not
  depend on the thread count. Turn it off when the outer loop is already threaded.
"""
struct InverseProblem{T,F,C<:AbstractDiscretisation}
    trajectories::Vector{SiteTrajectory{T,F}}
    cfg::C
    average::Bool
    threaded::Bool
    delta_loss::Bool
end

InverseProblem(trajectories::AbstractVector{<:SiteTrajectory}, cfg::AbstractDiscretisation;
               average::Bool = true, threaded::Bool = Threads.nthreads() > 1,
               delta_loss::Bool = true) =
    # The comprehension narrows the element type, e.g. of a vector built by
    # `push!`-ing into `SiteTrajectory[]`.
    InverseProblem([tr for tr in trajectories], cfg, average, threaded, delta_loss)

"""
    InverseProblem(sites::AbstractVector{<:SiteSeries}, cfg; kwargs...)

The real-data fit: one trajectory per site, averaged.
"""
InverseProblem(sites::AbstractVector{<:SiteSeries}, cfg::AbstractDiscretisation; kwargs...) =
    InverseProblem([SiteTrajectory(s) for s in sites], cfg; kwargs...)

"""
    with_discretisation(prob, cfg) -> InverseProblem

The same data and loss, posed with another discretisation — e.g. the fixed-step
problem as an [`ODEConfig`](@ref) problem on the continuous PDE.
"""
with_discretisation(prob::InverseProblem, cfg::AbstractDiscretisation) =
    InverseProblem(prob.trajectories, cfg, prob.average, prob.threaded, prob.delta_loss)

"""
    rethread(prob, threaded) -> InverseProblem

Copy of `prob` with the site-level threading flag flipped.
"""
rethread(prob::InverseProblem, threaded::Bool) =
    InverseProblem(prob.trajectories, prob.cfg, prob.average, threaded, prob.delta_loss)

ntransitions(prob::InverseProblem) = sum(ntransitions, prob.trajectories; init = 0)
Base.length(prob::InverseProblem) = length(prob.trajectories)

"""Factor the summed per-transition MSEs are multiplied by."""
loss_scale(prob::InverseProblem) = prob.average && ntransitions(prob) > 0 ? 1 / ntransitions(prob) : 1.0

function Base.show(io::IO, prob::InverseProblem{T}) where {T}
    print(io, "InverseProblem{$T}(", length(prob.trajectories), " sites, ",
          ntransitions(prob), " transitions, ", describe(prob.cfg), ", ",
          prob.average ? "mean" : "sum", " ",
          prob.delta_loss ? "delta" : "absolute", " loss)")
end

"""
    mean_squared_delta_error(pred, prev_pred, target, prev_target)

MSE between the change the model predicts over a transition and the change
observed. Fitting on differences removes each site's static spatial pattern from
the objective, so the parameters are constrained by how the vegetation moves.
"""
function mean_squared_delta_error(pred::AbstractMatrix, prev_pred::AbstractMatrix,
                                  target::AbstractMatrix, prev_target::AbstractMatrix)
    acc = zero(promote_type(eltype(pred), eltype(target)))
    @inbounds @simd for i in eachindex(pred, prev_pred, target, prev_target)
        d = (pred[i] - prev_pred[i]) - (target[i] - prev_target[i])
        acc += d * d
    end
    return acc / length(pred)
end

"""
    mean_squared_error(pred, target)

Plain MSE on absolute fields (`use_delta_loss = False`).
"""
function mean_squared_error(pred::AbstractMatrix, target::AbstractMatrix)
    acc = zero(promote_type(eltype(pred), eltype(target)))
    @inbounds @simd for i in eachindex(pred, target)
        d = pred[i] - target[i]
        acc += d * d
    end
    return acc / length(pred)
end

"""
    loss(θ, prob) -> Real

The objective: roll every site forward under its observed forcing and accumulate
the per-transition MSE (on year-on-year differences by default). `θ` is the
11-vector of natural parameters in [`PARAM_NAMES`](@ref) order, or a
[`RietkerkParams`](@ref). The state is allocated at `eltype(θ)`, so a vector of
`ForwardDiff.Dual`s differentiates the whole rollout.
"""
function loss(θ::AbstractVector, prob::InverseProblem)
    p = RietkerkParams(θ)
    T = eltype(θ)
    trajectories = prob.trajectories
    partials = Vector{T}(undef, length(trajectories))
    if prob.threaded && length(trajectories) > 1
        Threads.@threads for i in eachindex(trajectories)
            partials[i] = trajectory_loss(trajectories[i], p, prob.cfg, T; delta = prob.delta_loss)
        end
    else
        for i in eachindex(trajectories)
            partials[i] = trajectory_loss(trajectories[i], p, prob.cfg, T; delta = prob.delta_loss)
        end
    end
    return sum(partials) * loss_scale(prob)   # fixed order: thread-count independent
end

loss(p::RietkerkParams, prob::InverseProblem) = loss(paramvector(p), prob)

"""
    trajectory_loss(tr, p, cfg, T; delta = true) -> T

Summed per-transition MSE over one site, with the state at element type `T`.
"""
function trajectory_loss(tr::SiteTrajectory, p::RietkerkParams, cfg::SimConfig,
                         ::Type{T}; delta::Bool = true) where {T}
    state = simstate(tr.initial_biomass; T = T)
    ws = workspace(state)
    prev_pred = copy(state.biomass)
    prev_target = tr.initial_target
    acc = zero(T)
    for k in eachindex(tr.targets)
        simulate_year!(state, p, tr.forcings[k], cfg, ws)
        target = tr.targets[k]
        acc += delta ? mean_squared_delta_error(state.biomass, prev_pred, target, prev_target) :
                       mean_squared_error(state.biomass, target)
        copyto!(prev_pred, state.biomass)
        prev_target = target
    end
    return acc
end

"""
    evaluate(p, series, cfg) -> NamedTuple

Held-out diagnostics for one site, as `evaluate_model_on_site` in
`realdata_test_invPDE.py`: delta-MSE and delta-MAE averaged over transitions, and
the mean over transitions of the pixelwise Pearson correlation between predicted
and observed biomass (0 when either field is constant or not finite, as there).
"""
function evaluate(p::RietkerkParams, series::SiteSeries{T}, cfg::SimConfig) where {T}
    obs = series.observations
    length(obs) >= 2 || return (mse = NaN, mae = NaN, correlation = NaN, num_transitions = 0)
    state = simstate(obs[1].biomass)
    ws = workspace(state)
    prev_pred = similar(state.biomass)
    total_mse = 0.0
    total_mae = 0.0
    corrs = Float64[]
    for k in 1:(length(obs) - 1)
        copyto!(prev_pred, state.biomass)
        simulate_year!(state, p, obs[k].weekly_precipitation, cfg, ws)
        mse = 0.0
        mae = 0.0
        @inbounds for i in eachindex(state.biomass)
            d = Float64((state.biomass[i] - prev_pred[i]) - (obs[k + 1].biomass[i] - obs[k].biomass[i]))
            mse += d * d
            mae += abs(d)
        end
        n = length(state.biomass)
        total_mse += mse / n
        total_mae += mae / n
        push!(corrs, pearson(state.biomass, obs[k + 1].biomass))
    end
    nt = length(obs) - 1
    return (mse = total_mse / nt, mae = total_mae / nt,
            correlation = Statistics.mean(corrs), num_transitions = nt)
end

"""
Pearson correlation over all pixels; 0 when either standard deviation is not
positive (which includes NaN), as `np.std(...) > 0` decides in Python.
"""
function pearson(a::AbstractArray, b::AbstractArray)
    x = vec(Float64.(a))
    y = vec(Float64.(b))
    sa = Statistics.std(x; corrected = false)
    sb = Statistics.std(y; corrected = false)
    (sa > 0 && sb > 0) || return 0.0
    return Statistics.cor(x, y)
end
