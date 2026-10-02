"""
    TrainConfig(; kwargs...)

Hyper-parameters of one fit. The optimiser works on `x = log θ`, as `invRietkerk`
stores its parameters, so `learning_rate` is a relative step.

Defaults are those of `realdata_train_invPDE.py`; use
[`SYNTHETIC_TRAIN_CONFIG`](@ref) and [`SYNTHETIC_1SITE_TRAIN_CONFIG`](@ref) for the
synthetic experiments.

| field           | meaning                                                         |
|:----------------|:----------------------------------------------------------------|
| `epochs`        | gradient steps                                                  |
| `learning_rate` | Adam step size in log θ                                         |
| `beta1`, `beta2`, `eps` | Adam constants (`betas=(0.9, 0.95)` in the real-data script) |
| `lr_decay`      | multiplicative decay per epoch (`StepLR(step_size=1, gamma)`)   |
| `grad_clip`     | max 2-norm of the log-space gradient, or `nothing`              |
| `bound_decades` | box: this many decades either side of the reference (W₀ ≤ 1)    |
| `init_decades`  | random starts: log-uniform within this many decades             |
| `save_interval` | parameter snapshot every this many epochs                       |
| `print_interval`, `verbose` | progress printing                                   |
"""
Base.@kwdef struct TrainConfig
    epochs::Int = 7500
    learning_rate::Float64 = 0.01
    beta1::Float64 = 0.9
    beta2::Float64 = 0.95
    eps::Float64 = 1e-8
    lr_decay::Float64 = 0.999
    grad_clip::Union{Nothing,Float64} = 10.0
    bound_decades::Float64 = 4.0
    init_decades::Float64 = 1.0
    save_interval::Int = 10
    print_interval::Int = 10
    verbose::Bool = true
end

"""
    TrainConfig(base; kwargs...)

Copy `base` with selected fields replaced.
"""
function TrainConfig(base::TrainConfig; kwargs...)
    fields = (; (f => getfield(base, f) for f in fieldnames(TrainConfig))...)
    return TrainConfig(; merge(fields, NamedTuple(kwargs))...)
end

"""Hyper-parameters of `realdata_train_invPDE.py`."""
const REALDATA_TRAIN_CONFIG = TrainConfig()

"""Hyper-parameters of `train_invPDE_synthetic_batch.py`: no clipping, slower decay."""
const SYNTHETIC_TRAIN_CONFIG = TrainConfig(epochs = 10_000, lr_decay = 0.9999, grad_clip = nothing)

"""Hyper-parameters of `train_invPDE_synthetic_batch_1site.py`: as the four-site run with `betas=(0.95, 0.99)`."""
const SYNTHETIC_1SITE_TRAIN_CONFIG = TrainConfig(SYNTHETIC_TRAIN_CONFIG; beta1 = 0.95, beta2 = 0.99)

"""
    TrainResult

Outcome of a fit: fitted and initial parameters, loss and gradient-norm curves,
parameter snapshots (every `save_interval` epochs, taken after the step, plus the
last epoch), wall-clock time, and how many random draws the start took.

`converged` is false when the run stopped at a non-finite loss or gradient; the
non-finite loss is the last entry of `loss_history`, as in the synthetic scripts.
"""
struct TrainResult
    params::RietkerkParams{Float64}
    initial_params::RietkerkParams{Float64}
    loss_history::Vector{Float64}
    parameter_history::Vector{Dict{String,Float64}}
    epochs_run::Int
    final_loss::Float64
    elapsed_seconds::Float64
    converged::Bool
    seed::Union{Nothing,Int}
    init_draws::Int
    gradnorm_history::Vector{Float64}
end

function Base.show(io::IO, r::TrainResult)
    print(io, "TrainResult(final_loss = ", @sprintf("%.6g", r.final_loss),
          ", epochs = ", r.epochs_run, ", ", @sprintf("%.1f", r.elapsed_seconds), "s",
          r.converged ? "" : ", DIVERGED", ")")
end

function _snapshot(epoch::Integer, θ::AbstractVector)
    d = Dict{String,Float64}("epoch" => Float64(epoch))
    for i in 1:NPARAMS
        d[PARAM_NAMES[i]] = θ[i]
    end
    return d
end

"""
    train(prob, p0; cfg = REALDATA_TRAIN_CONFIG, reference, backend = EnzymeBackend(),
          seed = nothing, init_draws = 1, checkpoint_path = nothing,
          checkpoint_every = 10, callback = nothing) -> TrainResult

Fit the eleven coefficients from the start `p0`. Each epoch reproduces the PyTorch
loop: loss and gradient, the gradient taken to log space (∂L/∂log θ = θ ∂L/∂θ),
clipped to `grad_clip`, an Adam step on log θ, learning-rate decay, projection of
log θ onto the bounds of `reference`, then a snapshot every `save_interval` epochs.

`backend` picks the derivative (see [`AbstractGradientBackend`](@ref)). With
`checkpoint_path`, the whole training state is written there (JSON, write then
rename) every `checkpoint_every` epochs and an existing file is resumed from, so a
killed run continues where it stopped, as the Python scripts do.

`callback(epoch, θ, loss, gradnorm)` runs at every snapshot, with θ in natural units.
"""
function train(prob::InverseProblem, p0::RietkerkParams; cfg::TrainConfig = REALDATA_TRAIN_CONFIG,
               reference::RietkerkParams,
               backend::AbstractGradientBackend = EnzymeBackend(),
               seed::Union{Nothing,Integer} = nothing, init_draws::Integer = 1,
               checkpoint_path::Union{Nothing,AbstractString} = nothing,
               checkpoint_every::Integer = 10, callback = nothing)
    lo, hi = log_bounds(reference; decades = cfg.bound_decades)
    initial = convert(RietkerkParams{Float64}, p0)
    x = clamp.(log.(paramvector(initial)), lo, hi)
    opt = AdamState(NPARAMS; lr = cfg.learning_rate, beta1 = cfg.beta1, beta2 = cfg.beta2,
                    eps = cfg.eps)
    loss_history = Float64[]
    gradnorm_history = Float64[]
    parameter_history = Dict{String,Float64}[]
    start_epoch = 0
    prior_elapsed = 0.0

    if checkpoint_path !== nothing && isfile(checkpoint_path)
        ck = JSON.parsefile(checkpoint_path; allownan = true)
        x = Float64.(ck["log_params"])
        opt.m .= Float64.(ck["adam_m"])
        opt.v .= Float64.(ck["adam_v"])
        opt.t = Int(ck["adam_t"])
        opt.lr = Float64(ck["lr"])
        initial = RietkerkParams(Dict{String,Any}(ck["initial_params"]))
        init_draws = Int(ck["init_draws"])
        loss_history = Float64.(ck["loss_history"])
        gradnorm_history = Float64.(ck["gradnorm_history"])
        parameter_history = [Dict{String,Float64}(String(k) => Float64(v) for (k, v) in s)
                             for s in ck["parameter_history"]]
        start_epoch = Int(ck["next_epoch"])
        prior_elapsed = Float64(ck["elapsed_seconds"])
        cfg.verbose && println("Resuming from $checkpoint_path at epoch $start_epoch")
    end

    cache = gradient_cache(prob, backend)
    g = zeros(NPARAMS)
    θ = exp.(x)
    converged = true
    t0 = time()

    function save_checkpoint(next_epoch)
        save_json(checkpoint_path * ".tmp", Dict{String,Any}(
            "log_params" => x, "adam_m" => opt.m, "adam_v" => opt.v, "adam_t" => opt.t,
            "lr" => opt.lr, "initial_params" => paramdict(initial), "init_draws" => init_draws,
            "loss_history" => loss_history, "gradnorm_history" => gradnorm_history,
            "parameter_history" => parameter_history, "next_epoch" => next_epoch,
            "elapsed_seconds" => prior_elapsed + time() - t0, "seed" => seed))
        mv(checkpoint_path * ".tmp", checkpoint_path; force = true)
    end

    for epoch in start_epoch:(cfg.epochs - 1)
        L = loss_and_gradient!(g, cache, θ)
        g .*= θ                                   # ∂L/∂log θ
        if !isfinite(L) || any(!isfinite, g)
            push!(loss_history, L)
            cfg.verbose && @warn "non-finite loss or gradient; stopping" epoch loss = L
            converged = false
            break
        end
        push!(loss_history, L)
        push!(gradnorm_history, sqrt(sum(abs2, g)))   # before clipping
        cfg.grad_clip === nothing || clip_global_norm!(g, cfg.grad_clip)
        adam_step!(opt, x, g)
        decay_lr!(opt, cfg.lr_decay)
        x .= clamp.(x, lo, hi)
        θ .= exp.(x)

        if epoch % cfg.save_interval == 0
            push!(parameter_history, _snapshot(epoch, θ))
            callback === nothing || callback(epoch, θ, L, gradnorm_history[end])
        end
        if cfg.verbose && (epoch % cfg.print_interval == 0 || epoch == cfg.epochs - 1)
            @printf("Epoch %5d  loss %.6g  |∇| %.3g  lr %.4g\n", epoch, L, gradnorm_history[end], opt.lr)
            flush(stdout)
        end
        if checkpoint_path !== nothing && (epoch + 1) % checkpoint_every == 0
            save_checkpoint(epoch + 1)
        end
    end

    # A last snapshot at the final epoch, as the Python scripts append it — labelled
    # with the epoch that produced it rather than with `epochs - 1` when the run
    # stopped early.
    last_epoch = length(loss_history) - 1
    if converged && last_epoch >= 0 &&
       (isempty(parameter_history) || parameter_history[end]["epoch"] != last_epoch)
        push!(parameter_history, _snapshot(last_epoch, θ))
    end

    return TrainResult(RietkerkParams(copy(θ)), initial, loss_history, parameter_history,
                       length(loss_history), isempty(loss_history) ? NaN : loss_history[end],
                       prior_elapsed + time() - t0, converged,
                       seed === nothing ? nothing : Int(seed), Int(init_draws), gradnorm_history)
end

"""
    train(prob; cfg, reference, seed, backend, checkpoint_path, ...) -> TrainResult

Fit from a random start: drawn log-uniformly within `cfg.init_decades` of
`reference` and redrawn until it keeps vegetation alive over the training rollout
([`draw_viable_params`](@ref)). `seed` makes the start reproducible (within Julia).
A run with an existing checkpoint resumes from it instead of drawing.
"""
function train(prob::InverseProblem; cfg::TrainConfig = REALDATA_TRAIN_CONFIG,
               reference::RietkerkParams, seed::Union{Nothing,Integer} = nothing,
               checkpoint_path::Union{Nothing,AbstractString} = nothing, kwargs...)
    if checkpoint_path !== nothing && isfile(checkpoint_path)
        return train(prob, reference; cfg = cfg, reference = reference, seed = seed,
                     checkpoint_path = checkpoint_path, kwargs...)
    end
    rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(seed)
    p0, draws = draw_viable_params(rng, reference, viability_sites(prob.trajectories), prob.cfg;
                                   decades = cfg.init_decades, bound_decades = cfg.bound_decades)
    cfg.verbose && println("Random start accepted after $draws draw(s)")
    return train(prob, p0; cfg = cfg, reference = reference, seed = seed, init_draws = draws,
                 checkpoint_path = checkpoint_path, kwargs...)
end

"""
    train_many(prob, seeds; cfg, reference, backend = EnzymeBackend(), parallel = :runs,
               checkpoint_dir = nothing) -> Vector{TrainResult}

Fit from several random restarts (the published workflow trains 10 per site set
and filters them, because the objective is non-convex).

- `parallel = :runs`: one thread per restart, each fit serial over sites — the
  better use of cores when there are more restarts than threads.
- `:sites`: restarts one after another, threads across sites within each loss.
- `:none`: fully serial.

Results come back in `seeds` order regardless. With `checkpoint_dir`, run `i`
checkpoints to `checkpoint_dir/run_<seed>.json` and resumes from it.
"""
function train_many(prob::InverseProblem, seeds::AbstractVector{<:Integer};
                    cfg::TrainConfig = REALDATA_TRAIN_CONFIG, reference::RietkerkParams,
                    backend::AbstractGradientBackend = EnzymeBackend(),
                    parallel::Symbol = :runs,
                    checkpoint_dir::Union{Nothing,AbstractString} = nothing)
    parallel in (:runs, :sites, :none) ||
        throw(ArgumentError("parallel must be :runs, :sites or :none, got :$parallel"))
    checkpoint_dir === nothing || mkpath(checkpoint_dir)
    ckpath(s) = checkpoint_dir === nothing ? nothing : joinpath(checkpoint_dir, "run_$(s).json")
    results = Vector{TrainResult}(undef, length(seeds))
    if parallel === :runs && Threads.nthreads() > 1 && length(seeds) > 1
        inner = rethread(prob, false)
        quiet = TrainConfig(cfg; verbose = false)
        # Compile the gradient once before several threads ask for it.
        loss_and_gradient(inner, reference; backend = backend)
        Threads.@threads for i in eachindex(seeds)
            results[i] = train(inner; cfg = quiet, reference = reference, seed = seeds[i],
                               backend = backend, checkpoint_path = ckpath(seeds[i]))
        end
    else
        outer = parallel === :sites ? rethread(prob, true) : rethread(prob, false)
        for i in eachindex(seeds)
            cfg.verbose && @info "fitting run $i/$(length(seeds))" seed = seeds[i]
            results[i] = train(outer; cfg = cfg, reference = reference, seed = seeds[i],
                               backend = backend, checkpoint_path = ckpath(seeds[i]))
        end
    end
    return results
end
